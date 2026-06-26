#!/usr/bin/env python3
import argparse
import base64
import copy
import mimetypes
import os
import time
from pathlib import Path

from video_pipeline import (
    DEFAULT_NEGATIVE_PROMPT,
    download_file,
    extract_result_url,
    generate_video_with_video_model,
    load_env,
    output_path_from_url,
    parse_json_object,
    read_text_file,
    require_env,
    wait_for_video_result,
    write_json_file,
)


DEFAULT_PROMPT = (
    "参考图片中的主体保持清晰完整，围绕主体生成自然真实的短视频运动，"
    "镜头缓慢推进，光线柔和，画面稳定，真实质感，5秒广告短片风格"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用本地参考图调用图生视频模型。")
    parser.add_argument("image_path", type=Path, help="本地参考图片路径，例如 assets/logo.png")
    parser.add_argument("--env", type=Path, default=Path(".env"), help=".env 路径")
    parser.add_argument("--prompt", default=None, help="图生视频提示词")
    parser.add_argument("--prompt-file", type=Path, default=None, help="从 txt 文件读取提示词")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/image_to_video.mp4"),
        help="输出视频文件路径",
    )
    parser.add_argument(
        "--request-output",
        type=Path,
        default=Path("outputs/image_to_video_request.json"),
        help="保存脱敏请求 JSON 的路径",
    )
    parser.add_argument(
        "--response-output",
        type=Path,
        default=Path("outputs/image_to_video_response.json"),
        help="保存最终任务响应 JSON 的路径",
    )
    parser.add_argument("--api-url", default=None, help="覆盖 .env 中的图生视频 API URL")
    parser.add_argument("--model", default=None, help="覆盖 .env 中的图生视频模型")
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument(
        "--image-field",
        default="img_url",
        help="图片字段名；当前 ModelArts Wan I2V 验证可用的是 img_url",
    )
    parser.add_argument(
        "--image-location",
        choices=("input", "top-level", "parameters"),
        default="input",
        help="图片字段放置位置；当前验证可用的是 input",
    )
    parser.add_argument(
        "--image-format",
        choices=("data-uri", "base64"),
        default="data-uri",
        help="图片编码格式；当前验证可用的是 data-uri",
    )
    parser.add_argument(
        "--extra-json",
        default=None,
        help='额外合并到请求体的 JSON object，例如 \'{"parameters":{"seed":123}}\'',
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--max-wait", type=int, default=1800)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--download-retry-interval", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="只生成请求 JSON，不调用接口")
    return parser.parse_args()


def first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt and args.prompt_file:
        raise ValueError("--prompt 和 --prompt-file 只能二选一")
    if args.prompt_file:
        return read_text_file(args.prompt_file, "提示词文件")
    return args.prompt or DEFAULT_PROMPT


def encode_image(image_path: Path, image_format: str) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"参考图片不存在: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"参考图片路径不是文件: {image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    if image_format == "base64":
        return encoded

    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    return f"data:{mime_type};base64,{encoded}"


def merge_dict(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def build_payload(
    *,
    model: str,
    prompt: str,
    negative_prompt: str,
    image_field: str,
    image_location: str,
    image_value: str,
    size: str,
    duration: int,
    extra: dict,
) -> dict:
    payload = {
        "model": model,
        "input": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        },
        "parameters": {
            "size": size,
            "duration": duration,
        },
    }

    if image_location == "input":
        payload["input"][image_field] = image_value
    elif image_location == "parameters":
        payload["parameters"][image_field] = image_value
    else:
        payload[image_field] = image_value

    return merge_dict(payload, extra)


def redact_image(payload: dict, image_field: str, image_location: str) -> dict:
    redacted = copy.deepcopy(payload)
    container = redacted
    if image_location == "input":
        container = redacted.get("input", {})
    elif image_location == "parameters":
        container = redacted.get("parameters", {})

    image_value = container.get(image_field) if isinstance(container, dict) else None
    if isinstance(image_value, str):
        container[image_field] = f"<local image encoded, {len(image_value)} chars>"
    return redacted


def main() -> int:
    args = parse_args()
    load_env(args.env)
    env = require_env("OPENAI_API_KEY")

    api_url = args.api_url or first_env(
        "IMAGE_TO_VIDEO_API_URL",
        "IMAGET_TO_VIDEO_API_URL",
        "VIDEO_API_URL",
    )
    model = args.model or first_env(
        "IMAGE_TO_VIDEO_MODEL",
        "IMAGET_TO_VIDEO_MODEL",
        "VIDEO_MODEL",
    )
    if not api_url:
        raise RuntimeError("缺少图生视频 API URL：请配置 IMAGE_TO_VIDEO_API_URL 或 IMAGET_TO_VIDEO_API_URL")
    if not model:
        raise RuntimeError("缺少图生视频模型：请配置 IMAGE_TO_VIDEO_MODEL 或 IMAGET_TO_VIDEO_MODEL")

    prompt = read_prompt(args)
    image_value = encode_image(args.image_path, args.image_format)
    payload = build_payload(
        model=model,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        image_field=args.image_field,
        image_location=args.image_location,
        image_value=image_value,
        size=args.size,
        duration=args.duration,
        extra=parse_json_object(args.extra_json),
    )
    write_json_file(args.request_output, redact_image(payload, args.image_field, args.image_location))
    print(f"已保存脱敏请求 JSON: {args.request_output}")

    if args.dry_run:
        print("dry run：未调用图生视频接口")
        return 0

    started_at = time.time()
    create_response = generate_video_with_video_model(
        api_key=env["OPENAI_API_KEY"],
        video_api_url=api_url,
        payload=payload,
        timeout=args.timeout,
    )
    task_id = create_response.get("task_id") or create_response.get("id")
    if not task_id:
        raise RuntimeError(f"图生视频接口未返回 task_id/id: {create_response}")

    print(f"图生视频任务 ID: {task_id}")
    final_response = wait_for_video_result(
        api_key=env["OPENAI_API_KEY"],
        video_api_url=api_url,
        task_id=task_id,
        poll_interval=args.poll_interval,
        max_wait=args.max_wait,
        timeout=args.timeout,
    )
    write_json_file(args.response_output, final_response)
    print(f"已保存最终任务响应: {args.response_output}")

    result_url = extract_result_url(final_response)
    output_path = output_path_from_url(result_url, args.output)
    download_file(
        result_url,
        output_path,
        timeout=args.timeout,
        attempts=args.download_retries,
        retry_interval=args.download_retry_interval,
    )
    print(f"已下载视频文件: {output_path}")
    print(f"耗时: {time.time() - started_at:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
