# LRC Studio v1.0.2

一个参考 [Flare-Sky/LRCMaker-AI-Backend](https://github.com/Flare-Sky/LRCMaker-AI-Backend) 工作流、但不依赖浏览器插件的本地 LRC 歌词生成工具。

## 特点

- **无需油猴、浏览器扩展、账号或第三方网站**
- 核心手动打轴功能仅使用 Python 标准库，**无需安装任何 Python 插件/依赖**
- 音频、歌词、波形分析全部在本机浏览器中完成，不上传文件
- 支持 MP3、WAV、M4A、AAC、OGG、FLAC、OPUS 等浏览器可解码的音频格式
- 支持粘贴纯文本、导入 `.txt` / `.lrc`
- 可视化波形、播放/暂停、后退/前进、播放速度调整
- `Space` 播放，`Enter` 标记当前行；方向键微调时间
- 自动平均分布、按时间排序、逐行编辑、导出 UTF-8 LRC
- 支持歌曲标题、歌手、专辑、作词、作曲信息
- 可勾选「写入歌曲信息」，默认开启；关闭后只导出歌词时间轴
- 可选接入本地 Whisper 自动对齐，弹窗内显示实时进度（来自本地 AI 任务与控制台回调）；不安装 AI 组件不影响基础使用
- 歌词时间轴区提供「清空歌词」按钮，完成一首后可快速开始下一首
- 解析歌词和导出 LRC 时保留时间轴中的空行；空行时间取前后歌词时间戳的中间值

## 最简单的启动方式

### Windows

双击 `run.bat`。如果系统没有关联 Python，可在命令提示符中运行：

```bat
py app.py
```

### macOS / Linux

```bash
chmod +x run.sh
./run.sh
```

或者：

```bash
python3 app.py
```

程序会自动打开 `http://127.0.0.1:8765`。如果端口被占用，会自动寻找下一个可用端口。关闭终端窗口即可停止。

## 使用流程

1. 左侧拖入歌曲音频。
2. 粘贴歌词，点击「解析歌词，开始打轴」，或点击右上角「导入歌词」。
3. 点击「开始连续打轴」进入连续打轴，按空格播放；唱到每行开头时按 Enter。
4. 点击某一行可以试听该行；使用 `←` / `→` 微调 0.1 秒，`Shift + ←` / `Shift + →` 微调 1 秒。
5. 确认是否勾选「写入歌曲信息」，再点击「导出 LRC」。勾选时会使用 `[00:00.00]歌曲标题`、`ARTISTS:`、`Albums:`、`Lyrics:`、`Music:` 五行头部，并根据第一行歌词前的时间平均分配头部时间；取消勾选时只导出歌词时间轴。

已有 LRC 也可以直接导入，之后只需拖动播放位置或修改每行时间。完成一首后，点击「歌词时间轴」右上角的「清空歌词」并确认，即可载入下一首歌词；歌曲信息和音频文件会保留，方便更换或继续编辑。

## 可选：本地 Whisper AI 对齐

基础功能不需要 AI 依赖。如果想使用参考项目里的「AI 自动对齐」思路：

```bash
python -m pip install -r requirements-ai.txt
python app.py
```

首次使用模型时，`stable-ts` / `faster-whisper` 可能会下载 Whisper 模型；之后可以将模型放在项目目录的 `models/faster-whisper-small/`，或通过 `LRC_AI_MODEL` 指向本地模型目录，从而离线运行。AI 推理需要较多 CPU、内存和磁盘空间，长歌曲处理时间会明显增加。

AI 对齐会优先使用 `faster-whisper` 自带的 PyAV 解码器，不要求把 `ffmpeg.exe` 加入 Windows 的 PATH。升级过旧环境时，可以执行：

```bash
python -m pip install -U av faster-whisper stable-ts
```

> 说明：为了让基础版本真正做到「下载后即可用」，本项目没有把数百 MB 的模型塞进代码仓库。AI 是可选增强功能，手动打轴不依赖它。

### Windows `WinError 2` / 找不到指定的文件

如果日志里出现：

```text
[WinError 2] 系统找不到指定的文件
```

通常是旧版 `stable-ts` 直接寻找系统 `ffmpeg.exe`。当前版本已经改为通过 PyAV 读取音频，并且 CPU 默认使用 `int8`，不会再因为 float16 警告而失败。请确认使用的是更新后的 `app.py`，然后在同一个 Python 环境执行：

```bash
python -m pip install -U av faster-whisper stable-ts
```

安装完成后关闭旧的 LRC Studio 窗口，重新运行 `run.bat` 或 `python app.py`。`float16` 被自动转成 `float32` 的提示本身只是性能警告，不是导致失败的原因。

## 项目结构

```text
lrc-studio/
├── app.py                 # 零依赖本地服务 + 可选 AI API
├── index.html             # 自包含前端，无 CDN、无外部脚本
├── requirements-ai.txt    # 可选 Whisper 依赖
├── run.bat                # Windows 一键启动
├── run.sh                 # macOS/Linux 一键启动
└── README.md
```

## 隐私与限制

- 基础模式没有任何上传接口，音频和歌词只存在当前浏览器页面内。
- 关闭页面后未导出的时间轴不会自动保存；歌曲资料字段会保存在浏览器本地存储中。
- 波形和播放能力取决于浏览器对音频格式的支持。
- 导出文件使用自定义可读头部：标题、歌手、Albums、`Lyrics:`、`Music:`，后接逐行时间标签；基础模式不伪造逐字时间。若播放器只接受 `[ti:]` 等标准元数据标签，可再按播放器要求转换头部。若需要逐字 KTV LRC，请使用可选 Whisper AI 对齐，并在导出前人工校准。

## 许可证

本项目的实现代码可自由修改使用。参考思路来自上方开源项目；如分发其衍生代码，请遵循原项目的 MIT 许可证与版权声明。
