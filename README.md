# Text Material to Promotional Video

项目分成四个脚本：

1. `generate_video_prompt.py`：读取材料 txt，调用 text model，输出分段视频提示词和每段配音稿。
2. `generate_video.py`：读取视频提示词 txt，调用 video model。分段提示词会按顺序生成多个 5 秒视频片段，配音稿不会发送给视频模型。
3. `generate_voiceover.py`：读取分段配音稿，用 OmniVoice 生成每段视频对应的语音。
4. `combine_videos.py`：把生成的视频片段按顺序合成一个完整视频。

`.env` 中需要配置：

```env
OPENAI_API_KEY=...
TEXT_API_URL=https://api.modelarts-maas.com/openai/v1
TEXT_MODEL=kimi-k2.6
VIDEO_API_URL=https://api.modelarts-maas.com/v1/video/generations
VIDEO_MODEL=Wan2.2-T2V-A14B
```

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

如果需要 GPU 推理，建议按本机 CUDA 版本从 PyTorch 官方源安装对应的 `torch` 和 `torchaudio` wheel。

合成视频需要系统中已安装 `ffmpeg`，并且命令行可直接执行 `ffmpeg`。

## 1. 生成视频提示词

```bash
python3 generate_video_prompt.py materials/material.txt \
  --output outputs/ai_meeting_prompt.txt \
  --voiceover-output outputs/ai_meeting_voiceover.txt \
  --save-raw-response outputs/text_response.json
```

输出：

```text
outputs/ai_meeting_prompt.txt
```

提示词文件会使用以下格式分段，每段对应一个 5 秒视频。段数不做硬性限制，会根据材料长度、信息密度和核心信息点动态决定。

```text
=== SEGMENT 1 ===
VIDEO_PROMPT:
第一段视频提示词
VOICEOVER:
第一段配音稿
=== SEGMENT 2 ===
VIDEO_PROMPT:
第二段视频提示词
VOICEOVER:
第二段配音稿
```

如果传入 `--voiceover-output`，会额外保存一个只包含配音稿的 txt，供后续语音生成流程使用。

## 2. 读取提示词并生成视频

先 dry run，检查实际请求体。分段提示词会保存为聚合 JSON：

```bash
python3 generate_video.py outputs/ai_meeting_prompt.txt \
  --request-output outputs/video_request.json \
  --dry-run
```

确认后调用视频接口并下载最终视频：

```bash
python3 generate_video.py outputs/ai_meeting_prompt.txt \
  --request-output outputs/video_request.json \
  --response-output outputs/video_response.json \
  --output outputs/video.mp4
```

如果提示词包含多段，脚本会生成：

```text
outputs/video_parts/segment_001.mp4
outputs/video_parts/segment_002.mp4
...
```

旧的单段提示词 txt 仍会按原流程只生成一个 `outputs/video.mp4`。

## 3. 生成每段语音

使用参考音频克隆同一个声音：

```bash
python3 generate_voiceover.py outputs/ai_meeting_voiceover.txt \
  --ref-audio ref.wav \
  --ref-text "参考音频对应的文字" \
  --output-dir outputs/audio_parts
```

也可以不用参考音频，使用声音设计：

```bash
python3 generate_voiceover.py outputs/ai_meeting_voiceover.txt \
  --instruct "female, warm tone, low pitch" \
  --output-dir outputs/audio_parts
```

脚本会生成：

```text
outputs/audio_parts/segment_001.wav
outputs/audio_parts/segment_002.wav
...
```

每段默认固定为 5 秒，和视频片段对齐。失败后可加 `--skip-existing` 跳过已经生成的音频。

## 4. 合成带语音的视频

把每段视频和同名语音先封装，再按顺序合成：

```bash
python3 combine_videos.py \
  --input-dir outputs/video_parts \
  --audio-dir outputs/audio_parts \
  --output outputs/video.mp4
```

脚本会优先按同名文件匹配：

```text
outputs/video_parts/segment_001.mp4 + outputs/audio_parts/segment_001.wav
outputs/video_parts/segment_002.mp4 + outputs/audio_parts/segment_002.wav
```

如果不传 `--audio-dir`，则保持旧逻辑，只拼接视频片段：

```bash
python3 combine_videos.py \
  --input-dir outputs/video_parts \
  --output outputs/video.mp4
```

默认使用 ffmpeg 直接拼接封装后的片段，不重新编码视频。如果片段编码参数不一致导致失败，可加 `--reencode`：

```bash
python3 combine_videos.py \
  --input-dir outputs/video_parts \
  --audio-dir outputs/audio_parts \
  --output outputs/video.mp4 \
  --reencode
```

## 常用参数

覆盖模型：

```bash
python3 generate_video_prompt.py materials/material.txt --model kimi-k2.6
python3 generate_video.py outputs/ai_meeting_prompt.txt --model Wan2.2-T2V-A14B
```

调整视频参数：

```bash
python3 generate_video.py outputs/ai_meeting_prompt.txt \
  --size 1280x720 \
  --duration 5 \
  --extra-json '{"seed":123}'
```

网络不稳定时增加下载重试：

```bash
python3 generate_video.py outputs/ai_meeting_prompt.txt \
  --download-retries 5 \
  --download-retry-interval 10
```

如果中途失败，保留已经下载好的片段后可跳过已有片段继续：

```bash
python3 generate_video.py outputs/ai_meeting_prompt.txt \
  --skip-existing \
  --download-retries 5 \
  --download-retry-interval 10
```

指定片段输出目录：

```bash
python3 generate_video.py outputs/ai_meeting_prompt.txt \
  --parts-dir outputs/ai_meeting_parts \
  --output outputs/ai_meeting.mp4
```
