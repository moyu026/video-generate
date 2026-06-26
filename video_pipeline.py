import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


TEXT_SYSTEM_PROMPT = """你是宣传视频导演、文生视频提示词工程师和广告配音文案。
把用户提供的文字材料改写成多段可直接用于文生视频模型的中文提示词，并为每段生成对应配音稿。每段生成 5 秒视频，后续会按顺序拼接成完整宣传片。

要求：
1. 只输出分段正文，不要解释、Markdown 或代码块。
2. 保留材料里的品牌、产品、卖点、受众和传播意图。
3. 根据材料长度、信息密度和卖点数量动态设计连续段落，不限制段落数量，不要为了控制段数而压缩重要信息。
4. 按核心信息点自然分段：每个核心场景、卖点、转折、使用步骤或记忆点通常单独成段；材料很短时可以只生成少量段落，材料很长时可以生成更多段落。
5. 每段都必须是一个独立、完整、适合生成 5 秒视频的文生视频提示词。
6. 段落之间要有连续的叙事和视觉节奏：开场吸引、痛点或场景、产品能力、使用效果、收束记忆点。
7. 每段补足画面、人物、动作、场景、光线、镜头运动、质感和节奏，避免依赖上一段才能理解。
8. 每段配音稿必须对应本段画面，适合 5 秒内自然读完，语气像广告旁白，简洁、有节奏，不要写舞台说明。
9. 配音稿不要包含“旁白：”“镜头：”等标签，不要包含括号说明，不要要求屏幕显示文字。
10. 成片感要像广告短片，不要像产品说明书。
11. 视频提示词尽量避免生成任何文字类画面元素，包括字幕、标题、招牌、海报、包装文案、屏幕文字、手机界面文字、电脑界面文字、白板文字、文件文字、品牌标语等；需要表达信息时用人物动作、产品外观、图标化元素、环境和配音表达。
12. 除非用户明确提供参考 logo 图片并要求出现 logo，否则不要让模型生成 logo 或文字标识；如果必须出现 logo，只描述“参考图中的 logo 保持清晰完整”，不要让模型重新生成额外文字。
13. 避免字幕、水印、乱码文字、错误文字、伪文字、logo 变形、低清、畸形手部、多余肢体。

输出格式必须严格如下，用分隔行标记每段，每段都必须包含 VIDEO_PROMPT 和 VOICEOVER：
=== SEGMENT 1 ===
VIDEO_PROMPT:
第一段 5 秒视频提示词
VOICEOVER:
第一段 5 秒配音稿
=== SEGMENT 2 ===
VIDEO_PROMPT:
第二段 5 秒视频提示词
VOICEOVER:
第二段 5 秒配音稿
"""


TEXT_USER_PROMPT_TEMPLATE = """请根据以下材料，生成多段宣传短视频的文生视频提示词和每段配音稿。每段对应 5 秒视频，最终按顺序拼接。

材料：
{material}
"""


DEFAULT_NEGATIVE_PROMPT = (
    "低清晰度，模糊，抖动，噪点，过曝，欠曝，画面撕裂，畸形人物，畸形手指，"
    "多余肢体，字幕，标题文字，屏幕文字，手机界面文字，电脑界面文字，招牌文字，"
    "海报文字，包装文字，文件文字，白板文字，错误文字，乱码文字，伪文字，水印，"
    "logo 变形，廉价感，卡通化，过度锐化"
)


def raise_for_status_with_body(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:2000]
        raise requests.HTTPError(
            f"{exc}; response body: {body}",
            response=response,
        ) from exc


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_env(*names: str) -> dict[str, str]:
    values = {}
    missing = []

    for name in names:
        value = os.getenv(name)
        if value:
            values[name] = value
        else:
            missing.append(name)

    if missing:
        raise RuntimeError(f"缺少环境变量: {', '.join(missing)}")
    return values


def read_text_file(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{label}为空: {path}")
    return text


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


VIDEO_PROMPT_LABELS = {"VIDEO_PROMPT:", "视频提示词:"}
VOICEOVER_LABELS = {"VOICEOVER:", "配音稿:"}


def split_prompt_segment_texts(prompt_text: str) -> list[str]:
    segments = []
    current_lines = []
    saw_marker = False

    for raw_line in prompt_text.splitlines():
        line = raw_line.strip()
        if line.upper().startswith("=== SEGMENT ") and line.endswith("==="):
            if saw_marker and current_lines:
                segment = "\n".join(current_lines).strip()
                if segment:
                    segments.append(segment)
            saw_marker = True
            current_lines = []
            continue
        current_lines.append(raw_line)

    if current_lines:
        segment = "\n".join(current_lines).strip()
        if segment:
            segments.append(segment)

    if not saw_marker:
        prompt = prompt_text.strip()
        return [prompt] if prompt else []
    return segments


def parse_prompt_segment(segment_text: str) -> dict[str, str]:
    video_lines = []
    voiceover_lines = []
    current_field = "video_prompt"

    for raw_line in segment_text.splitlines():
        line = raw_line.strip()
        normalized_line = line.upper().replace("：", ":")
        if normalized_line in VIDEO_PROMPT_LABELS:
            current_field = "video_prompt"
            continue
        if normalized_line in VOICEOVER_LABELS:
            current_field = "voiceover"
            continue

        if current_field == "voiceover":
            voiceover_lines.append(raw_line)
        else:
            video_lines.append(raw_line)

    return {
        "video_prompt": "\n".join(video_lines).strip(),
        "voiceover": "\n".join(voiceover_lines).strip(),
    }


def parse_prompt_segments(prompt_text: str) -> list[dict[str, str]]:
    parsed_segments = []
    for segment_text in split_prompt_segment_texts(prompt_text):
        segment = parse_prompt_segment(segment_text)
        if segment["video_prompt"] or segment["voiceover"]:
            parsed_segments.append(segment)
    return parsed_segments


def parse_video_prompt_segments(prompt_text: str) -> list[str]:
    return [
        segment["video_prompt"]
        for segment in parse_prompt_segments(prompt_text)
        if segment["video_prompt"]
    ]


def parse_voiceover_segments(prompt_text: str) -> list[str]:
    return [
        segment["voiceover"]
        for segment in parse_prompt_segments(prompt_text)
        if segment["voiceover"]
    ]


def parse_voiceover_file_segments(text: str) -> list[str]:
    voiceovers = parse_voiceover_segments(text)
    if voiceovers:
        return voiceovers
    return [segment.strip() for segment in split_prompt_segment_texts(text) if segment.strip()]


def format_voiceover_segments(voiceovers: list[str]) -> str:
    lines = []
    for index, voiceover in enumerate(voiceovers, start=1):
        lines.append(f"=== SEGMENT {index} ===")
        lines.append(voiceover)
    return "\n".join(lines)


def generate_prompt_with_text_model(
    *,
    api_key: str,
    text_api_url: str,
    text_model: str,
    material: str,
    temperature: float,
    timeout: int,
) -> tuple[str, dict]:
    url = f"{text_api_url.rstrip('/')}/chat/completions"
    payload = {
        "model": text_model,
        "messages": [
            {"role": "system", "content": TEXT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": TEXT_USER_PROMPT_TEMPLATE.format(material=material),
            },
        ],
        "temperature": temperature,
    }

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    raise_for_status_with_body(response)

    data = response.json()
    prompt = data["choices"][0]["message"]["content"].strip().strip("`").strip()
    if not prompt:
        raise RuntimeError("文本模型返回了空提示词")
    return prompt, data


def build_video_payload(
    *,
    video_model: str,
    prompt: str,
    negative_prompt: str,
    size: str,
    duration: int | None,
    extra: dict,
) -> dict:
    payload = {
        "model": video_model,
        "input": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        },
        "parameters": {
            "size": size,
        },
    }
    if duration is not None:
        payload["parameters"]["duration"] = duration
    payload.update(extra)
    return payload


def generate_video_with_video_model(
    *,
    api_key: str,
    video_api_url: str,
    payload: dict,
    timeout: int,
) -> dict:
    response = requests.post(
        video_api_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    raise_for_status_with_body(response)
    return response.json()


def get_video_task(
    *,
    api_key: str,
    video_api_url: str,
    task_id: str,
    timeout: int,
) -> dict:
    url = f"{video_api_url.rstrip('/')}/{task_id}"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    raise_for_status_with_body(response)
    return response.json()


def wait_for_video_result(
    *,
    api_key: str,
    video_api_url: str,
    task_id: str,
    poll_interval: int,
    max_wait: int,
    timeout: int,
) -> dict:
    deadline = time.time() + max_wait

    while True:
        task = get_video_task(
            api_key=api_key,
            video_api_url=video_api_url,
            task_id=task_id,
            timeout=timeout,
        )
        status = str(task.get("status", "")).lower()
        print(f"任务状态: {status or 'unknown'}")

        if status in {"succeeded", "success", "completed", "finished"}:
            return task
        if status in {"failed", "fail", "error", "canceled", "cancelled"}:
            raise RuntimeError(f"视频任务失败: {json.dumps(task, ensure_ascii=False)}")
        if time.time() >= deadline:
            raise TimeoutError(f"等待视频任务超时: {task_id}")

        time.sleep(poll_interval)


def extract_result_url(task: dict) -> str:
    candidates = [
        task.get("result_url"),
        task.get("url"),
        task.get("video_url"),
        task.get("output_url"),
        task.get("content", {}).get("result_url")
        if isinstance(task.get("content"), dict)
        else None,
    ]

    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate

    content = task.get("content")
    if isinstance(content, dict):
        for value in content.values():
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value

    raise RuntimeError(f"未在任务结果中找到视频 URL: {json.dumps(task, ensure_ascii=False)}")


def download_file(
    url: str,
    output_path: Path,
    timeout: int,
    attempts: int = 3,
    retry_interval: int = 5,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = max(1, attempts)

    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                raise_for_status_with_body(response)
                with output_path.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            return
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            output_path.unlink(missing_ok=True)
            if attempt >= attempts:
                break
            print(f"下载失败，{retry_interval}s 后重试 ({attempt}/{attempts}): {exc}")
            time.sleep(retry_interval)

    raise RuntimeError(f"下载文件失败，已重试 {attempts} 次: {url}") from last_error


def output_path_from_url(url: str, fallback: Path) -> Path:
    suffix = Path(urlparse(url).path).suffix
    if suffix and fallback.suffix != suffix:
        return fallback.with_suffix(suffix)
    return fallback


def parse_json_object(value: str | None) -> dict:
    if not value:
        return {}

    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("额外参数必须是 JSON object")
    return data
