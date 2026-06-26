#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from video_pipeline import (
    parse_voiceover_file_segments,
    read_text_file,
    write_json_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取分段配音稿，并用 OmniVoice 生成每段语音。")
    parser.add_argument("voiceover_path", type=Path, help="分段配音稿 txt 路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audio_parts"),
        help="输出语音片段目录",
    )
    parser.add_argument("--model", default="k2-fsa/OmniVoice", help="OmniVoice 模型名或本地路径")
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=None,
        help="参考音频路径；提供后使用 voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="参考音频对应文本；不传时 OmniVoice 会尝试自动识别",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help='声音设计描述；例如 "female, low pitch, warm tone"',
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="每段语音固定时长秒数；小于等于 0 时不固定",
    )
    parser.add_argument("--speed", type=float, default=None, help="语速；设置 duration 时通常不需要")
    parser.add_argument("--num-step", type=int, default=None, help="扩散步数；例如 16 更快，32 质量更稳")
    parser.add_argument(
        "--device",
        default="auto",
        help='推理设备；默认 auto，也可指定 "cuda:0"、"cpu"、"mps"、"xpu"',
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="模型 dtype",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="OmniVoice 默认输出采样率",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("outputs/audio_manifest.json"),
        help="保存每段配音稿和音频路径的 JSON",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已存在且非空的音频文件，便于失败后继续生成",
    )
    return parser.parse_args()


def resolve_device(torch_module, device: str) -> str:
    if device != "auto":
        return device
    if torch_module.cuda.is_available():
        return "cuda:0"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    if hasattr(torch_module, "xpu") and torch_module.xpu.is_available():
        return "xpu"
    return "cpu"


def resolve_dtype(torch_module, dtype: str, device: str):
    if dtype == "float16":
        return torch_module.float16
    if dtype == "bfloat16":
        return torch_module.bfloat16
    if dtype == "float32":
        return torch_module.float32
    return torch_module.float16 if device.startswith(("cuda", "xpu")) else torch_module.float32


def load_omnivoice():
    try:
        import soundfile as sf
        import torch
        from omnivoice import OmniVoice
    except ImportError as exc:
        raise RuntimeError(
            "缺少 OmniVoice 依赖。请先安装：pip install omnivoice soundfile"
        ) from exc
    return OmniVoice, sf, torch


def build_generate_kwargs(args: argparse.Namespace, text: str) -> dict:
    kwargs = {"text": text}
    if args.ref_audio:
        kwargs["ref_audio"] = str(args.ref_audio)
        if args.ref_text:
            kwargs["ref_text"] = args.ref_text
    elif args.instruct:
        kwargs["instruct"] = args.instruct

    if args.duration > 0:
        kwargs["duration"] = args.duration
    elif args.speed is not None:
        kwargs["speed"] = args.speed

    if args.num_step is not None:
        kwargs["num_step"] = args.num_step
    return kwargs


def main() -> int:
    args = parse_args()
    if args.ref_audio and args.instruct:
        raise ValueError("--ref-audio 和 --instruct 只能二选一")
    if args.ref_audio and not args.ref_audio.exists():
        raise FileNotFoundError(f"参考音频不存在: {args.ref_audio}")

    text = read_text_file(args.voiceover_path, "配音稿文件")
    voiceovers = parse_voiceover_file_segments(text)
    if not voiceovers:
        raise RuntimeError(f"配音稿文件没有可用段落: {args.voiceover_path}")

    OmniVoice, sf, torch = load_omnivoice()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype, device)

    print(f"加载 OmniVoice 模型: {args.model}")
    print(f"设备: {device}, dtype: {dtype}")
    model = OmniVoice.from_pretrained(args.model, device_map=device, dtype=dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, voiceover in enumerate(voiceovers, start=1):
        output_path = args.output_dir / f"segment_{index:03d}.wav"
        if args.skip_existing and output_path.exists() and output_path.stat().st_size > 0:
            print(f"第 {index}/{len(voiceovers)} 段已存在，跳过: {output_path}")
            manifest.append(
                {
                    "index": index,
                    "text": voiceover,
                    "output_path": str(output_path),
                    "skipped": True,
                }
            )
            continue

        print(f"开始生成第 {index}/{len(voiceovers)} 段语音: {voiceover}")
        generate_kwargs = build_generate_kwargs(args, voiceover)
        audio = model.generate(**generate_kwargs)
        if not audio:
            raise RuntimeError(f"OmniVoice 第 {index} 段返回了空音频")
        sf.write(output_path, audio[0], args.sample_rate)
        manifest.append(
            {
                "index": index,
                "text": voiceover,
                "output_path": str(output_path),
                "generate_kwargs": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in generate_kwargs.items()
                },
                "skipped": False,
            }
        )
        write_json_file(args.manifest_output, {"segments": manifest})
        print(f"已生成第 {index} 段语音: {output_path}")

    write_json_file(args.manifest_output, {"segments": manifest})
    print(f"已保存语音清单: {args.manifest_output}")
    print(json.dumps({"audio_dir": str(args.output_dir), "segments": len(manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
