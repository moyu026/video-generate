#!/usr/bin/env python3
import argparse
import base64
import mimetypes
import os
import time
from pathlib import Path

from video_pipeline import (
    DEFAULT_NEGATIVE_PROMPT,
    build_video_payload,
    download_file,
    extract_result_url,
    generate_video_with_video_model,
    load_env,
    output_path_from_url,
    parse_prompt_segments,
    parse_json_object,
    read_text_file,
    require_env,
    wait_for_video_result,
    write_json_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取视频提示词 txt，并调用视频生成模型。")
    parser.add_argument("prompt_path", type=Path, help="视频提示词 txt 路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/video.mp4"),
        help="输出视频文件路径",
    )
    parser.add_argument(
        "--request-output",
        type=Path,
        default=Path("outputs/video_request.json"),
        help="保存实际发送的视频请求 JSON 路径",
    )
    parser.add_argument("--env", type=Path, default=Path(".env"), help=".env 路径")
    parser.add_argument("--model", default=None, help="覆盖 .env 中的 VIDEO_MODEL")
    parser.add_argument("--image-model", default=None, help="覆盖 .env 中的图生视频模型")
    parser.add_argument("--image-api-url", default=None, help="覆盖 .env 中的图生视频 API URL")
    parser.add_argument(
        "--image-segment",
        action="append",
        default=[],
        metavar="N=IMAGE_PATH",
        help="指定某段使用参考图走图生视频，可重复传入，例如 --image-segment 6=assets/logo.png",
    )
    parser.add_argument(
        "--image-field",
        default="img_url",
        help="图生视频图片字段名；当前 ModelArts Wan I2V 验证可用的是 img_url",
    )
    parser.add_argument(
        "--image-format",
        choices=("data-uri", "base64"),
        default="data-uri",
        help="图生视频图片编码格式；当前验证可用的是 data-uri",
    )
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--duration", type=int, default=5, help="视频时长；当前接口支持 3 或 5")
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help="多段提示词生成时的视频片段输出目录；默认使用 <output文件名>_parts",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--extra-json",
        default=None,
        help='额外合并到视频请求体的 JSON object，例如 \'{"seed":123}\'',
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--download-retries", type=int, default=3, help="下载视频失败时的重试次数")
    parser.add_argument("--download-retry-interval", type=int, default=5, help="下载重试间隔秒数")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="多段生成时跳过已存在且非空的片段文件，便于失败后继续生成",
    )
    parser.add_argument("--poll-interval", type=int, default=10, help="轮询任务状态间隔秒数")
    parser.add_argument("--max-wait", type=int, default=1800, help="最长等待视频生成秒数")
    parser.add_argument(
        "--response-output",
        type=Path,
        default=Path("outputs/video_response.json"),
        help="保存最终任务响应 JSON 路径",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成请求 JSON，不调用视频接口",
    )
    return parser.parse_args()


def first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def parse_image_segments(values: list[str]) -> dict[int, Path]:
    image_segments = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--image-segment 格式必须是 N=IMAGE_PATH: {value}")

        index_text, image_path_text = value.split("=", 1)
        try:
            index = int(index_text)
        except ValueError as exc:
            raise ValueError(f"--image-segment 段号必须是整数: {value}") from exc

        if index < 1:
            raise ValueError(f"--image-segment 段号必须从 1 开始: {value}")

        image_path = Path(image_path_text)
        if not image_path.exists():
            raise FileNotFoundError(f"第 {index} 段参考图不存在: {image_path}")
        if not image_path.is_file():
            raise ValueError(f"第 {index} 段参考图不是文件: {image_path}")
        image_segments[index] = image_path
    return image_segments


def encode_image(image_path: Path, image_format: str) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    if image_format == "base64":
        return encoded

    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    return f"data:{mime_type};base64,{encoded}"


def build_image_video_payload(
    *,
    image_model: str,
    prompt: str,
    negative_prompt: str,
    image_field: str,
    image_value: str,
    size: str,
    duration: int | None,
    extra: dict,
) -> dict:
    payload = {
        "model": image_model,
        "input": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            image_field: image_value,
        },
        "parameters": {
            "size": size,
        },
    }
    if duration is not None:
        payload["parameters"]["duration"] = duration
    payload.update(extra)
    return payload


def redact_payload(payload: dict, image_field: str) -> dict:
    input_data = payload.get("input")
    if not isinstance(input_data, dict) or image_field not in input_data:
        return payload

    redacted = {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in payload.items()
    }
    image_value = redacted["input"].get(image_field)
    if isinstance(image_value, str):
        redacted["input"][image_field] = f"<local image encoded, {len(image_value)} chars>"
    return redacted


def main() -> int:
    args = parse_args()
    load_env(args.env)

    env = require_env("OPENAI_API_KEY", "VIDEO_API_URL", "VIDEO_MODEL")
    video_model = args.model or env["VIDEO_MODEL"]
    image_segments = parse_image_segments(args.image_segment)
    image_api_url = args.image_api_url or first_env(
        "IMAGE_TO_VIDEO_API_URL",
        "IMAGET_TO_VIDEO_API_URL",
        "VIDEO_API_URL",
    )
    image_model = args.image_model or first_env(
        "IMAGE_TO_VIDEO_MODEL",
        "IMAGET_TO_VIDEO_MODEL",
        "VIDEO_MODEL",
    )
    if image_segments and not image_api_url:
        raise RuntimeError("使用 --image-segment 时缺少图生视频 API URL")
    if image_segments and not image_model:
        raise RuntimeError("使用 --image-segment 时缺少图生视频模型")

    prompt_text = read_text_file(args.prompt_path, "视频提示词文件")
    prompt_segments = [
        segment for segment in parse_prompt_segments(prompt_text) if segment["video_prompt"]
    ]
    if not prompt_segments:
        raise RuntimeError(f"视频提示词文件没有可用段落: {args.prompt_path}")
    invalid_image_segments = sorted(index for index in image_segments if index > len(prompt_segments))
    if invalid_image_segments:
        raise ValueError(
            "指定的图生视频段号超出提示词段落总数 "
            f"({len(prompt_segments)}): {', '.join(str(index) for index in invalid_image_segments)}"
        )

    extra = parse_json_object(args.extra_json)
    segment_jobs = []
    for index, segment in enumerate(prompt_segments, start=1):
        image_path = image_segments.get(index)
        if image_path:
            payload = build_image_video_payload(
                image_model=image_model,
                prompt=segment["video_prompt"],
                negative_prompt=args.negative_prompt,
                image_field=args.image_field,
                image_value=encode_image(image_path, args.image_format),
                size=args.size,
                duration=args.duration,
                extra=extra,
            )
            segment_jobs.append(
                {
                    "index": index,
                    "mode": "image-to-video",
                    "api_url": image_api_url,
                    "image_path": str(image_path),
                    "payload": payload,
                }
            )
            continue

        payload = build_video_payload(
            video_model=video_model,
            prompt=segment["video_prompt"],
            negative_prompt=args.negative_prompt,
            size=args.size,
            duration=args.duration,
            extra=extra,
        )
        segment_jobs.append(
            {
                "index": index,
                "mode": "text-to-video",
                "api_url": env["VIDEO_API_URL"],
                "payload": payload,
            }
        )

    request_data = (
        redact_payload(segment_jobs[0]["payload"], args.image_field)
        if len(segment_jobs) == 1
        else {
            "segment_duration": args.duration,
            "segments": [
                {
                    "index": job["index"],
                    "mode": job["mode"],
                    "image_path": job.get("image_path"),
                    "voiceover": prompt_segments[index - 1].get("voiceover", ""),
                    "payload": redact_payload(job["payload"], args.image_field),
                }
                for index, job in enumerate(segment_jobs, start=1)
            ],
        }
    )
    write_json_file(args.request_output, request_data)
    print(f"已保存视频请求 JSON: {args.request_output}")
    print(f"共解析到 {len(prompt_segments)} 段提示词，每段生成 {args.duration}s 视频")
    if image_segments:
        image_segment_text = ", ".join(str(index) for index in sorted(image_segments))
        print(f"图生视频段落: {image_segment_text}")

    if args.dry_run:
        print("dry run：未调用视频接口")
        return 0

    started_at = time.time()

    final_responses = []
    output_paths = []
    parts_dir = args.parts_dir or args.output.with_suffix("").with_name(f"{args.output.stem}_parts")

    for job in segment_jobs:
        index = job["index"]
        fallback = args.output if len(segment_jobs) == 1 else parts_dir / f"segment_{index:03d}.mp4"
        if len(segment_jobs) > 1 and args.skip_existing and fallback.exists() and fallback.stat().st_size > 0:
            output_paths.append(str(fallback))
            print(f"第 {index}/{len(segment_jobs)} 段已存在，跳过: {fallback}")
            continue

        print(f"开始生成第 {index}/{len(segment_jobs)} 段 ({job['mode']})")
        create_response = generate_video_with_video_model(
            api_key=env["OPENAI_API_KEY"],
            video_api_url=job["api_url"],
            payload=job["payload"],
            timeout=args.timeout,
        )
        task_id = create_response.get("task_id")
        if not task_id:
            raise RuntimeError(f"视频接口未返回 task_id: {create_response}")

        print(f"第 {index} 段视频任务 ID: {task_id}")
        final_response = wait_for_video_result(
            api_key=env["OPENAI_API_KEY"],
            video_api_url=job["api_url"],
            task_id=task_id,
            poll_interval=args.poll_interval,
            max_wait=args.max_wait,
            timeout=args.timeout,
        )

        result_url = extract_result_url(final_response)
        output_path = output_path_from_url(result_url, fallback)
        final_responses.append(
            {
                "index": index,
                "mode": job["mode"],
                "image_path": job.get("image_path"),
                "task_id": task_id,
                "create_response": create_response,
                "final_response": final_response,
                "output_path": str(output_path),
            }
        )
        response_data = (
            final_responses[0]["final_response"]
            if len(segment_jobs) == 1
            else {"segments": final_responses, "output_paths": output_paths}
        )
        write_json_file(args.response_output, response_data)

        download_file(
            result_url,
            output_path,
            timeout=args.timeout,
            attempts=args.download_retries,
            retry_interval=args.download_retry_interval,
        )
        output_paths.append(str(output_path))
        if len(segment_jobs) > 1:
            write_json_file(
                args.response_output,
                {"segments": final_responses, "output_paths": output_paths},
            )
        print(f"已下载第 {index} 段视频文件: {output_path}")

    response_data = (
        final_responses[0]["final_response"]
        if len(final_responses) == 1
        else {"segments": final_responses, "output_paths": output_paths}
    )
    write_json_file(args.response_output, response_data)
    print(f"已保存最终任务响应: {args.response_output}")

    if len(output_paths) > 1:
        print(f"视频片段目录: {parts_dir}")
        print(f"可使用 combine_videos.py 合成: python3 combine_videos.py --input-dir {parts_dir} --output {args.output}")
    else:
        print(f"已下载视频文件: {output_paths[0]}")
    print(f"耗时: {time.time() - started_at:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
