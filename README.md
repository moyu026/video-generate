# Text Material to Promotional Video

项目分成两个脚本：

1. `generate_video_prompt.py`：读取材料 txt，调用 text model，输出视频提示词 txt。
2. `generate_video.py`：读取视频提示词 txt，调用 video model，等待任务完成并下载视频文件。

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
python3 generate_video_prompt.py materials/ai_meeting_assistant.txt \
  --output outputs/ai_meeting_prompt.txt \
  --save-raw-response outputs/text_response.json
```

输出：

```text
outputs/ai_meeting_prompt.txt
```

## 2. 读取提示词并生成视频

先 dry run，检查实际请求体：

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

## 常用参数

覆盖模型：

```bash
python3 generate_video_prompt.py materials/ai_meeting_assistant.txt --model kimi-k2.6
python3 generate_video.py outputs/ai_meeting_prompt.txt --model Wan2.2-T2V-A14B
```

调整视频参数：

```bash
python3 generate_video.py outputs/ai_meeting_prompt.txt \
  --size 1280x720 \
  --duration 5 \
  --extra-json '{"seed":123}'
```
