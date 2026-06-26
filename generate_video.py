#!/usr/bin/env python3
import argparse
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


def main() -> int:
    args = parse_args()
    load_env(args.env)

    env = require_env("OPENAI_API_KEY", "VIDEO_API_URL", "VIDEO_MODEL")
    video_model = args.model or env["VIDEO_MODEL"]
    prompt_text = read_text_file(args.prompt_path, "视频提示词文件")
    prompt_segments = [
        segment for segment in parse_prompt_segments(prompt_text) if segment["video_prompt"]
    ]
    if not prompt_segments:
        raise RuntimeError(f"视频提示词文件没有可用段落: {args.prompt_path}")

    extra = parse_json_object(args.extra_json)
    payloads = [
        build_video_payload(
            video_model=video_model,
            prompt=segment["video_prompt"],
            negative_prompt=args.negative_prompt,
            size=args.size,
            duration=args.duration,
            extra=extra,
        )
        for segment in prompt_segments
    ]

    request_data = (
        payloads[0]
        if len(payloads) == 1
        else {
            "segment_duration": args.duration,
            "segments": [
                {
                    "index": index,
                    "voiceover": prompt_segments[index - 1].get("voiceover", ""),
                    "payload": payload,
                }
                for index, payload in enumerate(payloads, start=1)
            ],
        }
    )
    write_json_file(args.request_output, request_data)
    print(f"已保存视频请求 JSON: {args.request_output}")
    print(f"共解析到 {len(prompt_segments)} 段提示词，每段生成 {args.duration}s 视频")

    if args.dry_run:
        print("dry run：未调用视频接口")
        return 0

    started_at = time.time()

    final_responses = []
    output_paths = []
    parts_dir = args.parts_dir or args.output.with_suffix("").with_name(f"{args.output.stem}_parts")

    for index, payload in enumerate(payloads, start=1):
        fallback = args.output if len(payloads) == 1 else parts_dir / f"segment_{index:03d}.mp4"
        if len(payloads) > 1 and args.skip_existing and fallback.exists() and fallback.stat().st_size > 0:
            output_paths.append(str(fallback))
            print(f"第 {index}/{len(payloads)} 段已存在，跳过: {fallback}")
            continue

        print(f"开始生成第 {index}/{len(payloads)} 段")
        create_response = generate_video_with_video_model(
            api_key=env["OPENAI_API_KEY"],
            video_api_url=env["VIDEO_API_URL"],
            payload=payload,
            timeout=args.timeout,
        )
        task_id = create_response.get("task_id")
        if not task_id:
            raise RuntimeError(f"视频接口未返回 task_id: {create_response}")

        print(f"第 {index} 段视频任务 ID: {task_id}")
        final_response = wait_for_video_result(
            api_key=env["OPENAI_API_KEY"],
            video_api_url=env["VIDEO_API_URL"],
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
                "task_id": task_id,
                "create_response": create_response,
                "final_response": final_response,
                "output_path": str(output_path),
            }
        )
        response_data = (
            final_responses[0]["final_response"]
            if len(payloads) == 1
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
        if len(payloads) > 1:
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
