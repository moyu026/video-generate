# Text Material to Promotional Video

项目分成三个脚本：

1. `generate_video_prompt.py`：读取材料 txt，调用 text model，输出分段视频提示词 txt。
2. `generate_video.py`：读取视频提示词 txt，调用 video model。分段提示词会按顺序生成多个 5 秒视频片段。
3. `combine_videos.py`：把生成的视频片段按顺序合成一个完整视频。

`.env` 中需要配置：

```env
OPENAI_API_KEY=...
TEXT_API_URL=https://api.modelarts-maas.com/openai/v1
TEXT_MODEL=kimi-k2.6
VIDEO_API_URL=https://api.modelarts-maas.com/v1/video/generations
VIDEO_MODEL=Wan2.2-T2V-A14B
```

## 1. 生成视频提示词

```bash
python3 generate_video_prompt.py materials/material.txt \
  --output outputs/ai_meeting_prompt.txt \
  --save-raw-response outputs/text_response.json
```

输出：

```text
outputs/ai_meeting_prompt.txt
```

提示词文件会使用以下格式分段，每段对应一个 5 秒视频。段数不做硬性限制，会根据材料长度、信息密度和核心信息点动态决定。

```text
=== SEGMENT 1 ===
第一段视频提示词
=== SEGMENT 2 ===
第二段视频提示词
```

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

## 3. 合成视频片段

默认按目录中文件名排序合成：

```bash
python3 combine_videos.py \
  --input-dir outputs/video_parts \
  --output outputs/video.mp4
```

默认使用 ffmpeg 直接拼接，不重新编码。如果片段编码参数不一致导致失败，可加 `--reencode`：

```bash
python3 combine_videos.py \
  --input-dir outputs/video_parts \
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
