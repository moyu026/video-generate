#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按顺序合成多个视频片段。")
    parser.add_argument(
        "videos",
        nargs="*",
        type=Path,
        help="要合成的视频片段路径；不传时使用 --input-dir 下按文件名排序的视频文件",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="视频片段目录，按文件名排序后合成",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/video.mp4"),
        help="合成后的视频输出路径",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg 命令路径")
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="重新编码输出；默认直接拼接，要求片段编码参数一致",
    )
    return parser.parse_args()


def collect_videos(input_dir: Path | None, videos: list[Path]) -> list[Path]:
    if videos:
        return videos
    if not input_dir:
        raise ValueError("请提供视频路径，或使用 --input-dir 指定片段目录")
    if not input_dir.exists():
        raise FileNotFoundError(f"片段目录不存在: {input_dir}")

    paths = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    return sorted(paths)


def ffmpeg_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", r"'\''")


def build_ffmpeg_command(
    *,
    ffmpeg: str,
    list_path: Path,
    output_path: Path,
    reencode: bool,
) -> list[str]:
    command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
    ]
    if reencode:
        command.extend(["-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart"])
    else:
        command.extend(["-c", "copy"])
    command.append(str(output_path))
    return command


def main() -> int:
    args = parse_args()
    videos = collect_videos(args.input_dir, args.videos)
    if not videos:
        raise RuntimeError("没有找到可合成的视频片段")

    for path in videos:
        if not path.exists():
            raise FileNotFoundError(f"视频片段不存在: {path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        delete=False,
    ) as list_file:
        list_path = Path(list_file.name)
        for video in videos:
            list_file.write(f"file '{ffmpeg_concat_path(video)}'\n")

    try:
        command = build_ffmpeg_command(
            ffmpeg=args.ffmpeg,
            list_path=list_path,
            output_path=args.output,
            reencode=args.reencode,
        )
        subprocess.run(command, check=True)
    finally:
        list_path.unlink(missing_ok=True)

    print(f"已合成 {len(videos)} 个视频片段: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
