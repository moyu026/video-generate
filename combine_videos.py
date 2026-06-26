#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


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
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="语音片段目录；提供后会先为每段视频匹配同名音频并封装",
    )
    parser.add_argument(
        "--muxed-dir",
        type=Path,
        default=Path("outputs/muxed_parts"),
        help="带音频视频片段的临时输出目录",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg 命令路径")
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="重新编码输出；默认直接拼接，要求片段编码参数一致",
    )
    parser.add_argument(
        "--keep-video-audio",
        action="store_true",
        help="封装语音时保留原视频音轨；默认用生成语音替换原音轨",
    )
    parser.add_argument(
        "--shortest",
        action="store_true",
        help="封装语音时以较短流结束；默认保留完整视频时长",
    )
    parser.add_argument(
        "--skip-existing-muxed",
        action="store_true",
        help="跳过已存在且非空的带音频片段",
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


def collect_audios(audio_dir: Path) -> list[Path]:
    if not audio_dir.exists():
        raise FileNotFoundError(f"语音片段目录不存在: {audio_dir}")
    paths = [
        path
        for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    ]
    return sorted(paths)


def match_audio_for_video(video: Path, audios_by_stem: dict[str, Path], audios: list[Path], index: int) -> Path:
    audio = audios_by_stem.get(video.stem)
    if audio:
        return audio
    if index < len(audios):
        return audios[index]
    raise FileNotFoundError(f"没有找到第 {index + 1} 个视频对应的语音: {video}")


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


def mux_video_audio(
    *,
    ffmpeg: str,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    keep_video_audio: bool,
    shortest: bool,
) -> None:
    output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
    ]
    if keep_video_audio:
        command.extend(["-map", "0:a?", "-map", "1:a:0"])
    else:
        command.extend(["-map", "1:a:0"])
    command.extend(["-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart"])
    if shortest:
        command.append("-shortest")
    command.append(str(output_path))
    subprocess.run(command, check=True)


def mux_segments_with_audio(
    *,
    ffmpeg: str,
    videos: list[Path],
    audio_dir: Path,
    muxed_dir: Path,
    keep_video_audio: bool,
    shortest: bool,
    skip_existing: bool,
) -> list[Path]:
    audios = collect_audios(audio_dir)
    if len(audios) < len(videos):
        raise RuntimeError(f"语音片段数量不足: 视频 {len(videos)} 个，语音 {len(audios)} 个")

    audios_by_stem = {audio.stem: audio for audio in audios}
    muxed_paths = []
    for index, video in enumerate(videos):
        audio = match_audio_for_video(video, audios_by_stem, audios, index)
        output_path = muxed_dir / f"{video.stem}.mp4"
        if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
            print(f"带音频片段已存在，跳过: {output_path}")
            muxed_paths.append(output_path)
            continue

        print(f"封装第 {index + 1}/{len(videos)} 段: {video} + {audio}")
        mux_video_audio(
            ffmpeg=ffmpeg,
            video_path=video,
            audio_path=audio,
            output_path=output_path,
            keep_video_audio=keep_video_audio,
            shortest=shortest,
        )
        muxed_paths.append(output_path)
    return muxed_paths


def main() -> int:
    args = parse_args()
    videos = collect_videos(args.input_dir, args.videos)
    if not videos:
        raise RuntimeError("没有找到可合成的视频片段")

    for path in videos:
        if not path.exists():
            raise FileNotFoundError(f"视频片段不存在: {path}")

    concat_videos = videos
    if args.audio_dir:
        concat_videos = mux_segments_with_audio(
            ffmpeg=args.ffmpeg,
            videos=videos,
            audio_dir=args.audio_dir,
            muxed_dir=args.muxed_dir,
            keep_video_audio=args.keep_video_audio,
            shortest=args.shortest,
            skip_existing=args.skip_existing_muxed,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        delete=False,
    ) as list_file:
        list_path = Path(list_file.name)
        for video in concat_videos:
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

    print(f"已合成 {len(concat_videos)} 个视频片段: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
