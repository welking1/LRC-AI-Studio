#!/usr/bin/env python3
"""LRC Studio - a zero-dependency local LRC timing tool.

The core editor runs entirely in the browser.  This tiny server only serves the
single-page app and exposes an optional /api/align endpoint when the user has
installed the reference project's local Whisper dependencies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import os
import socket
import sys
import tempfile
import threading
import traceback
import webbrowser
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
INDEX_FILE = ROOT / "index.html"
APP_NAME = "LRC Studio"
APP_VERSION = "1.0.0"


def find_free_port(start: int = 8765) -> int:
    """Return the first free TCP port from start onwards."""
    for port in range(start, 65536):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("找不到可用端口")


def ai_available() -> bool:
    # stable-ts's faster-whisper adapter and its PyAV decoder are both needed
    # for the optional local alignment path.
    return all(importlib.util.find_spec(name) is not None for name in ("stable_whisper", "faster_whisper", "av"))


def decode_audio_for_alignment(audio_path: str):
    """Decode audio with PyAV instead of invoking an external ffmpeg.exe.

    stable-ts's file-path loader historically launches the ffmpeg command-line
    program.  That is the usual source of WinError 2 on a clean Windows PC.
    faster-whisper already ships a PyAV based decoder, so passing its 16 kHz
    NumPy waveform to stable-ts keeps the AI path self-contained.
    """
    try:
        from faster_whisper.audio import decode_audio
    except Exception as exc:
        raise RuntimeError(
            "缺少 PyAV 音频解码组件。请在启动 LRC Studio 的同一个 Python 环境执行："
            "python -m pip install -U av faster-whisper stable-ts"
        ) from exc
    try:
        return decode_audio(audio_path, sampling_rate=16000, split_stereo=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "音频解码器启动失败。请在启动 LRC Studio 的同一个 Python 环境执行："
            "python -m pip install -U av faster-whisper stable-ts，然后重启程序。"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"无法读取音频文件，请尝试转换为 WAV 或 MP3：{exc}") from exc


def json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def format_time(seconds: float) -> str:
    """Format seconds as an LRC tag, e.g. [01:23.45]."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    # Avoid 60.00 after rounding (rare, but annoying in an LRC file).
    if remaining >= 59.995:
        minutes += 1
        remaining = 0.0
    return f"[{minutes:02d}:{remaining:05.2f}]"


def clean_text(value: str) -> str:
    return "".join(value.replace("\n", "").replace("\r", "").split())


def split_lyrics(raw: str) -> tuple[list[str], list[str]]:
    """Split the reference project's optional staff/header lines from lyrics."""
    staff: list[str] = []
    sung: list[str] = []
    raw_lines = raw.splitlines()
    has_separator = any(line.strip() == "---" for line in raw_lines)
    started = not has_separator
    staff_prefixes = ("作词", "作曲", "编曲", "演唱", "制作", "来源", "词", "曲")

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue
        if line == "---":
            started = True
            continue
        if not started:
            staff.append(line)
            continue
        # When no explicit separator is provided, treat common credit lines as
        # non-sung notes only if they occur before the first actual lyric line.
        if not sung and any(line.startswith(prefix) and (":" in line or "：" in line) for prefix in staff_prefixes):
            staff.append(line)
            continue
        sung.append(line)
    return staff, sung


def generate_aligned_lrc(audio_path: str, raw_lyrics: str, title: str = "", artist: str = "", album: str = "") -> dict[str, str]:
    """Optional Whisper alignment adapted from LRCMaker-AI-Backend.

    This function is intentionally imported lazily.  The basic LRC Studio app
    works with Python's standard library only; users who want AI alignment can
    install requirements-ai.txt and place/cache a Whisper model locally.
    """
    if not ai_available():
        raise RuntimeError("本地 AI 组件未安装。基础手动打轴无需安装任何插件；如需 AI 对齐，请先安装 requirements-ai.txt。")

    import stable_whisper  # type: ignore

    model_dir = ROOT / "models" / "faster-whisper-small"
    cache_dir = Path.home() / ".lrc_studio_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target: str = str(model_dir) if model_dir.exists() and any(model_dir.iterdir()) else os.environ.get("LRC_AI_MODEL", "small")

    print("🧠 正在进行本地 Whisper 强制对齐…", flush=True)
    device = os.environ.get("LRC_AI_DEVICE", "cpu")
    # CPU does not have efficient float16 kernels. int8 avoids the
    # CTranslate2 warning and is usually faster on ordinary Windows PCs.
    compute_type = os.environ.get(
        "LRC_AI_COMPUTE_TYPE",
        "int8" if device.lower() == "cpu" else "float16",
    )
    model = stable_whisper.load_faster_whisper(
        target,
        device=device,
        compute_type=compute_type,
        download_root=str(cache_dir),
    )
    staff_lines, sung_lines = split_lyrics(raw_lyrics)
    if not sung_lines:
        raise ValueError("没有找到有效歌词。请至少输入一行歌词。")

    # Do not pass the file path to stable-ts: its path loader invokes an
    # external ffmpeg.exe on Windows.  PyAV is bundled with faster-whisper and
    # works without adding anything to the system PATH.
    audio_waveform = decode_audio_for_alignment(audio_path)
    result = model.align(audio_waveform, "\n".join(sung_lines), language=os.environ.get("LRC_LANGUAGE", "zh"))
    all_words: list[Any] = []
    for segment in result.segments:
        words = getattr(segment, "words", None) or []
        if words:
            all_words.extend(words)
        else:
            # Some model versions do not return word timestamps.  Keep a
            # segment-level fallback so standard LRC can still be generated.
            all_words.append(type("Word", (), {"word": segment.text, "start": segment.start, "end": segment.end})())

    standard: list[str] = []
    enhanced: list[str] = []
    if title:
        standard.append(f"[ti:{title}]")
        enhanced.append(f"[ti:{title}]")
    if artist:
        standard.append(f"[ar:{artist}]")
        enhanced.append(f"[ar:{artist}]")
    if album:
        standard.append(f"[al:{album}]")
        enhanced.append(f"[al:{album}]")

    if staff_lines:
        first_start = float(getattr(all_words[0], "start", 0.0)) if all_words else 0.0
        intro = max(0.0, first_start - 0.5)
        interval = intro / max(1, len(staff_lines))
        for index, staff in enumerate(staff_lines):
            tag = f"{format_time(index * interval)}{staff}"
            standard.append(tag)
            enhanced.append(tag)

    word_index = 0
    previous_end = -1.0
    for line in sung_lines:
        if word_index < len(all_words):
            start = float(getattr(all_words[word_index], "start", 0.0))
        else:
            start = max(0.0, previous_end + 0.1)
        if previous_end >= 0 and start - previous_end > 3.0:
            gap = f"{format_time(previous_end + 0.2)} "
            standard.append(gap)
            enhanced.append(gap)

        target_len = len(clean_text(line))
        current_len = 0
        current_words: list[Any] = []
        current_end = start
        while word_index < len(all_words) and current_len < target_len:
            word = all_words[word_index]
            current_words.append(word)
            current_len += len(clean_text(str(getattr(word, "word", ""))))
            current_end = float(getattr(word, "end", current_end))
            word_index += 1

        standard.append(f"{format_time(start)}{line}")
        enhanced_line = format_time(start)
        for word in current_words:
            word_text = clean_text(str(getattr(word, "word", "")))
            if word_text:
                enhanced_line += f"<{format_time(float(getattr(word, 'start', start)))}>{word_text}"
        enhanced.append(enhanced_line)
        previous_end = current_end

    if previous_end >= 0:
        end_tag = f"{format_time(previous_end + 1.0)} "
        standard.append(end_tag)
        enhanced.append(end_tag)

    return {"standard_lrc": "\n".join(standard), "enhanced_lrc": "\n".join(enhanced)}


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """Parse FormData without adding python-multipart as a dependency."""
    header = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8")
    message = BytesParser(policy=default).parsebytes(header + body)
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename:
            files[name] = (filename, payload)
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[name] = payload.decode(charset, errors="replace")
    return fields, files


class LRCHandler(BaseHTTPRequestHandler):
    server_version = "LRCStudio/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the terminal useful without dumping every static asset request.
        if self.path.startswith("/api/"):
            super().log_message(format, *args)

    def send_bytes(self, payload: bytes, content_type: str = "application/octet-stream", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, data: Any, status: int = 200) -> None:
        self.send_bytes(json_bytes(data), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/ping":
            self.send_json({
                "status": "ok",
                "app": APP_NAME,
                "version": APP_VERSION,
                "ai_available": ai_available(),
            })
            return
        if path in ("/", "/index.html"):
            try:
                self.send_bytes(INDEX_FILE.read_bytes(), "text/html; charset=utf-8")
            except OSError as exc:
                self.send_json({"error": str(exc)}, 500)
            return
        if path == "/favicon.ico":
            self.send_bytes(b"", "image/x-icon", 204)
            return
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/align":
            self.send_json({"code": 404, "message": "Not found", "data": None}, 404)
            return

        if not ai_available():
            self.send_json({
                "code": 503,
                "message": "本地 AI 组件未安装。基础手动打轴无需安装任何插件；如需 AI 对齐，请先安装 requirements-ai.txt。",
                "data": None,
            }, 503)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1024 * 1024 * 1024:
                raise ValueError("上传的音频为空或超过 1GB 限制")
            body = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            fields, files = parse_multipart(content_type, body)
            if "audio" not in files:
                raise ValueError("未收到音频文件")
            filename, content = files["audio"]
            suffix = Path(filename).suffix or ".audio"
            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                result = generate_aligned_lrc(
                    tmp_path,
                    fields.get("lyrics", ""),
                    fields.get("ti", ""),
                    fields.get("ar", ""),
                    fields.get("al", ""),
                )
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            self.send_json({"code": 200, "message": "success", "data": result})
        except Exception as exc:  # AI is optional; return a friendly message.
            print(f"❌ AI 对齐失败: {exc}", file=sys.stderr, flush=True)
            if os.environ.get("LRC_DEBUG"):
                traceback.print_exc()
            self.send_json({"code": 500, "message": str(exc), "data": None}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description="LRC Studio 本地歌词打轴工具")
    parser.add_argument("--port", type=int, default=0, help="端口，默认从 8765 开始自动寻找")
    parser.add_argument("--host", default=os.environ.get("LRC_HOST", "0.0.0.0"), help="监听地址")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args()

    port = args.port or find_free_port(8765)
    server = ThreadingHTTPServer((args.host, port), LRCHandler)
    local_url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 58)
    print(f"  🎵 {APP_NAME} 已启动")
    print(f"  浏览器打开: {local_url}")
    print("  所有基础功能在浏览器本地完成，可断网使用")
    if ai_available():
        print("  ✅ 检测到可选 Whisper 组件，AI 对齐按钮可用")
    else:
        print("  ℹ️ 未安装可选 Whisper 组件，基础打轴模式完全可用")
    print("  按 Ctrl+C 停止服务")
    print("=" * 58 + "\n", flush=True)

    if not args.no_browser:
        threading.Timer(0.35, lambda: webbrowser.open(local_url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 LRC Studio 已关闭")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
