#!/usr/bin/env python3
import argparse
from pathlib import Path

from video_pipeline import (
    format_voiceover_segments,
    generate_prompt_with_text_model,
    load_env,
    parse_voiceover_segments,
    read_text_file,
    require_env,
    write_json_file,
    write_text_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把材料 txt 生成视频提示词 txt。")
    parser.add_argument("material_path", type=Path, help="输入材料 txt 路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/video_prompt.txt"),
        help="输出视频提示词 txt 路径",
    )
    parser.add_argument("--env", type=Path, default=Path(".env"), help=".env 路径")
    parser.add_argument("--model", default=None, help="覆盖 .env 中的 TEXT_MODEL")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--save-raw-response",
        type=Path,
        default=None,
        help="可选：保存文本模型原始 JSON 响应",
    )
    parser.add_argument(
        "--voiceover-output",
        type=Path,
        default=None,
        help="可选：把每段配音稿单独保存到 txt，便于后续生成语音",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env(args.env)

    env = require_env("OPENAI_API_KEY", "TEXT_API_URL", "TEXT_MODEL")
    text_model = args.model or env["TEXT_MODEL"]
    material = read_text_file(args.material_path, "材料文件")

    prompt, raw_response = generate_prompt_with_text_model(
        api_key=env["OPENAI_API_KEY"],
        text_api_url=env["TEXT_API_URL"],
        text_model=text_model,
        material=material,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    write_text_file(args.output, prompt)
    print(f"已生成视频提示词: {args.output}")

    if args.voiceover_output:
        voiceovers = parse_voiceover_segments(prompt)
        if not voiceovers:
            raise RuntimeError("文本模型返回内容中没有找到配音稿")
        write_text_file(args.voiceover_output, format_voiceover_segments(voiceovers))
        print(f"已保存配音稿: {args.voiceover_output}")

    if args.save_raw_response:
        write_json_file(args.save_raw_response, raw_response)
        print(f"已保存文本模型原始响应: {args.save_raw_response}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
