import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


TEXT_SYSTEM_PROMPT = """你是宣传视频导演和文生视频提示词工程师。
把用户提供的文字材料改写成一条可直接用于文生视频模型的中文提示词。

要求：
1. 只输出提示词正文，不要标题、解释、Markdown、分镜编号或代码块。
2. 保留材料里的品牌、产品、卖点、受众和传播意图。
3. 补足画面、人物、动作、场景、光线、镜头运动、质感和节奏。
4. 成片感要像广告短片，不要像产品说明书。
5. 避免字幕、水印、乱码文字、logo 变形、低清、畸形手部、多余肢体。
"""


TEXT_USER_PROMPT_TEMPLATE = """请根据以下材料，生成一条 8-12 秒宣传短视频的文生视频提示词。

材料：
{material}
"""


DEFAULT_NEGATIVE_PROMPT = (
    "低清晰度，模糊，抖动，噪点，过曝，欠曝，画面撕裂，畸形人物，畸形手指，"
    "多余肢体，错误文字，乱码字幕，水印，logo 变形，廉价感，卡通化，过度锐化"
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


def download_file(url: str, output_path: Path, timeout: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        raise_for_status_with_body(response)
        with output_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


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
