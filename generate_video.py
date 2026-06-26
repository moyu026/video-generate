#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

from video_project import (
    DEFAULT_NEGATIVE_PROMPT,
    build_video_payload,
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
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--duration", type=int, default=5, help="视频时长；当前接口支持 3 或 5")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--extra-json",
        default=None,
        help='额外合并到视频请求体的 JSON object，例如 \'{"seed":123}\'',
    )
    parser.add_argument("--timeout", type=int, default=300)
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


def main() -> int:
    args = parse_args()
    load_env(args.env)

    env = require_env("OPENAI_API_KEY", "VIDEO_API_URL", "VIDEO_MODEL")
    video_model = args.model or env["VIDEO_MODEL"]
    prompt = read_text_file(args.prompt_path, "视频提示词文件")

    payload = build_video_payload(
        video_model=video_model,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        size=args.size,
        duration=args.duration,
        extra=parse_json_object(args.extra_json),
    )
    write_json_file(args.request_output, payload)
    print(f"已保存视频请求 JSON: {args.request_output}")

    if args.dry_run:
        print("dry run：未调用视频接口")
        return 0

    started_at = time.time()
    create_response = generate_video_with_video_model(
        api_key=env["OPENAI_API_KEY"],
        video_api_url=env["VIDEO_API_URL"],
        payload=payload,
        timeout=args.timeout,
    )
    task_id = create_response.get("task_id")
    if not task_id:
        raise RuntimeError(f"视频接口未返回 task_id: {create_response}")

    print(f"视频任务 ID: {task_id}")
    final_response = wait_for_video_result(
        api_key=env["OPENAI_API_KEY"],
        video_api_url=env["VIDEO_API_URL"],
        task_id=task_id,
        poll_interval=args.poll_interval,
        max_wait=args.max_wait,
        timeout=args.timeout,
    )
    write_json_file(args.response_output, final_response)
    print(f"已保存最终任务响应: {args.response_output}")

    result_url = extract_result_url(final_response)
    output_path = output_path_from_url(result_url, args.output)
    download_file(result_url, output_path, timeout=args.timeout)
    print(f"已下载视频文件: {output_path}")
    print(f"耗时: {time.time() - started_at:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
