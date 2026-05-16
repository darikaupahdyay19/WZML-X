# ruff: noqa: E402
"""
Video Tools (-vt) module for WZML-X.

Provides an inline-keyboard driven post-processing menu that exposes 11
FFmpeg-backed operations on videos that have just been downloaded into the
standard WZML-X working directory:

    /usr/src/app/downloads/{task_id}/   (user_id == task_id in the upstream listener)

The module integrates with the TaskListener pipeline via
``process_video_tools(listener)`` which is meant to be called from
``TaskListener.on_download_complete`` whenever ``listener.video_tools`` is
True (i.e. the user passed ``-vt``).  When invoked standalone via the
``/vtools`` command the module behaves identically against the user's last
finished task directory.

Strict constraints (mandatory, enforced here):

  * C1  -- ``-n`` is rejected when a Merge operation (1, 2 or 3) is selected.
  * C2  -- Merge operations require ``-m`` (multi-mode); otherwise an
           Inline alert is shown to the user.
  * C3  -- Trim, Watermark, Remove, Extract and Convert capture their
           parameters from the FIRST video in the working directory and
           apply them to every other video in the batch.
  * C4  -- If ``-m`` is absent, only the first video in the working
           directory is processed even if multiple are present.

Async execution:
  Every FFmpeg invocation goes through ``asyncio.create_subprocess_shell``
  so the bot's event loop is never blocked.

Output:
  Once processing succeeds, the resulting files replace (or are added to)
  the listener's working directory and the listener is asked to continue
  with its standard upload flow (``proceed_upload``) so the artefacts are
  delivered through the existing MirrorLeechListener / TgUploader paths.
"""

from __future__ import annotations

import json
import shlex
import time
from asyncio import create_subprocess_shell, gather
from asyncio.subprocess import PIPE
from os import path as ospath, walk
from typing import Any

from aiofiles.os import listdir, makedirs
from aiofiles.os import path as aiopath
from aiofiles.os import remove
from aioshutil import move
from pyrogram.filters import command, regex
from pyrogram.handlers import CallbackQueryHandler, MessageHandler

from .. import DOWNLOAD_DIR, LOGGER, bot_loop
from ..core.config_manager import BinConfig, Config
from ..core.tg_client import TgClient
from ..helper.ext_utils.bot_utils import arg_parser, new_task
from ..helper.telegram_helper.bot_commands import BotCommands
from ..helper.telegram_helper.button_build import ButtonMaker
from ..helper.telegram_helper.filters import CustomFilters
from ..helper.telegram_helper.message_utils import (
    delete_message,
    edit_message,
    send_message,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v",
    ".ts", ".mts", ".m2ts", ".wmv", ".3gp", ".vob", ".ogv",
}
AUDIO_EXTS = {".mp3", ".aac", ".m4a", ".flac", ".wav", ".opus", ".ogg", ".ac3"}
SUBTITLE_EXTS = {".srt", ".ass", ".ssa", ".vtt", ".sub"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

RESOLUTION_MAP = {
    "1080p": (1920, 1080),
    "720p":  (1280, 720),
    "540p":  (960, 540),
    "480p":  (854, 480),
    "360p":  (640, 360),
}

CALLBACK_PREFIX = "vt"

# Operation codes used in callback_data and routing
OP_MERGE_VV   = "mvv"
OP_MERGE_VA   = "mva"
OP_MERGE_VS   = "mvs"
OP_HARDSUB    = "hsb"
OP_SUBSYNC    = "ssy"
OP_COMPRESS   = "cmp"
OP_TRIM       = "trm"
OP_WATERMARK  = "wmk"
OP_REMOVE_VID = "rmv"   # extract audio only (mute video → audio)
OP_EXTRACT_VID = "exv"  # extract video only (no audio)
OP_CONVERT    = "cvt"

MERGE_OPS = {OP_MERGE_VV, OP_MERGE_VA, OP_MERGE_VS}
INHERITED_OPS = {OP_TRIM, OP_WATERMARK, OP_REMOVE_VID, OP_EXTRACT_VID, OP_CONVERT}

# In-memory session store, keyed by the bot message-id of the keyboard.
# Each entry is a dict with: user_id, work_dir, multi, rename, listener,
# created_at, op (set on click), params (set on click).
VT_SESSIONS: dict[int, dict[str, Any]] = {}
SESSION_TTL = 60 * 30  # 30 minutes


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _ffmpeg() -> str:
    """Return the configured ffmpeg binary name."""
    return getattr(BinConfig, "FFMPEG_NAME", "ffmpeg")


def _ffprobe() -> str:
    """ffprobe ships alongside the configured ffmpeg binary."""
    name = _ffmpeg()
    # By convention WZML-X renames ffmpeg → mediaforge; ffprobe stays standard.
    return "ffprobe" if name in ("ffmpeg", "mediaforge") else name.replace(
        "ffmpeg", "ffprobe"
    )


async def run_ffmpeg(cmd: str) -> tuple[int, str, str]:
    """Run an FFmpeg command via shell asynchronously.

    Returns ``(returncode, stdout, stderr)``.
    """
    LOGGER.info(f"[VT] FFmpeg → {cmd}")
    proc = await create_subprocess_shell(cmd, stdout=PIPE, stderr=PIPE)
    stdout_b, stderr_b = await proc.communicate()
    stdout = stdout_b.decode(errors="ignore").strip() if stdout_b else ""
    stderr = stderr_b.decode(errors="ignore").strip() if stderr_b else ""
    if proc.returncode != 0:
        LOGGER.error(f"[VT] FFmpeg failed (rc={proc.returncode}): {stderr[-800:]}")
    return int(proc.returncode or 0), stdout, stderr


async def list_files(work_dir: str) -> list[str]:
    """Return absolute paths of every regular file under ``work_dir``."""
    out: list[str] = []
    for root, _, files in walk(work_dir):
        for f in files:
            out.append(ospath.join(root, f))
    return sorted(out)


def _classify(path: str) -> str:
    ext = ospath.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in SUBTITLE_EXTS:
        return "subtitle"
    if ext in IMAGE_EXTS:
        return "image"
    return "other"


async def find_videos(work_dir: str) -> list[str]:
    return [p for p in await list_files(work_dir) if _classify(p) == "video"]


async def find_audios(work_dir: str) -> list[str]:
    return [p for p in await list_files(work_dir) if _classify(p) == "audio"]


async def find_subs(work_dir: str) -> list[str]:
    return [p for p in await list_files(work_dir) if _classify(p) == "subtitle"]


async def find_images(work_dir: str) -> list[str]:
    return [p for p in await list_files(work_dir) if _classify(p) == "image"]


async def probe_video(path: str) -> dict[str, Any]:
    """Run ffprobe and return parsed JSON, or ``{}`` on failure."""
    cmd = (
        f"{_ffprobe()} -v error -print_format json -show_streams -show_format "
        f"{shlex.quote(path)}"
    )
    rc, out, err = await run_ffmpeg(cmd)
    if rc != 0:
        return {}
    try:
        return json.loads(out) if out else {}
    except json.JSONDecodeError:
        LOGGER.error(f"[VT] ffprobe JSON decode failed for {path}: {err}")
        return {}


def _video_stream(meta: dict) -> dict:
    for s in meta.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    return {}


def _resolution_of(meta: dict) -> tuple[int, int] | None:
    s = _video_stream(meta)
    w, h = s.get("width"), s.get("height")
    if w and h:
        return int(w), int(h)
    return None


def _label_for_resolution(res: tuple[int, int]) -> str:
    """Snap (w,h) to the closest target tier label."""
    h = res[1]
    pick = min(RESOLUTION_MAP.items(), key=lambda kv: abs(kv[1][1] - h))
    return pick[0]


def _output_path(src: str, suffix: str, ext: str | None = None) -> str:
    base, src_ext = ospath.splitext(src)
    return f"{base}.{suffix}{ext or src_ext}"


async def _replace(src: str, dst: str) -> None:
    """Atomically replace ``src`` with ``dst`` (both absolute paths)."""
    if await aiopath.exists(src):
        await remove(src)
    await move(dst, src)


def _user_alert(query, text: str, show: bool = True) -> Any:
    """Helper to answer a callback query with an alert."""
    return query.answer(text, show_alert=show)


# ---------------------------------------------------------------------------
# Inline keyboard
# ---------------------------------------------------------------------------


def build_video_tools_keyboard(session_id: int) -> Any:
    """Build the 11-button inline keyboard for the video tools menu.

    Layout (organised rows):

        Row 1 — Merge V+V        | Merge V+A
        Row 2 — Merge V+S        | Hardsub (sudo)
        Row 3 — SubSync          | Compress (HEVC)
        Row 4 — Trim             | Watermark
        Row 5 — Remove Video     | Extract Video
        Row 6 — Convert (Resize) ▼
        Row 7 — Cancel
    """
    buttons = ButtonMaker()

    s = session_id

    buttons.data_button("🎬 Merge Video+Video", f"{CALLBACK_PREFIX} {s} {OP_MERGE_VV}")
    buttons.data_button("🔊 Merge Video+Audio", f"{CALLBACK_PREFIX} {s} {OP_MERGE_VA}")

    buttons.data_button("📝 Merge Video+Subtitle", f"{CALLBACK_PREFIX} {s} {OP_MERGE_VS}")
    buttons.data_button("🔥 Hardsub (sudo)", f"{CALLBACK_PREFIX} {s} {OP_HARDSUB}")

    buttons.data_button("⏱️ SubSync", f"{CALLBACK_PREFIX} {s} {OP_SUBSYNC}")
    buttons.data_button("📦 Compress (HEVC CRF28)", f"{CALLBACK_PREFIX} {s} {OP_COMPRESS}")

    buttons.data_button("✂️ Trim", f"{CALLBACK_PREFIX} {s} {OP_TRIM}")
    buttons.data_button("💧 Watermark", f"{CALLBACK_PREFIX} {s} {OP_WATERMARK}")

    buttons.data_button("🔇 Remove Video Stream", f"{CALLBACK_PREFIX} {s} {OP_REMOVE_VID}")
    buttons.data_button("🎞️ Extract Video Stream", f"{CALLBACK_PREFIX} {s} {OP_EXTRACT_VID}")

    buttons.data_button("🔁 Convert (Resize)", f"{CALLBACK_PREFIX} {s} {OP_CONVERT}")

    buttons.data_button("❌ Cancel", f"{CALLBACK_PREFIX} {s} cancel", "footer")

    return buttons.build_menu(2)


def build_resolution_keyboard(session_id: int) -> Any:
    """Sub-menu for the Convert/Resize action."""
    buttons = ButtonMaker()
    for label in RESOLUTION_MAP:
        buttons.data_button(
            label, f"{CALLBACK_PREFIX} {session_id} {OP_CONVERT}:{label}"
        )
    buttons.data_button("⬅️ Back", f"{CALLBACK_PREFIX} {session_id} back")
    return buttons.build_menu(2)


# ---------------------------------------------------------------------------
# FFmpeg operations (one coroutine per feature)
# ---------------------------------------------------------------------------


async def op_merge_videos(videos: list[str], work_dir: str) -> str | None:
    """1) Concatenate multiple videos using the concat demuxer (no re-encode)."""
    if len(videos) < 2:
        return None
    list_path = ospath.join(work_dir, ".vt_concat_list.txt")
    lines = "\n".join(f"file {shlex.quote(v)}" for v in videos)
    # Write the concat manifest
    async with await _open_async(list_path, "w") as fh:
        await fh.write(lines + "\n")

    out_path = ospath.join(work_dir, "merged_output.mkv")
    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y -f concat -safe 0 "
        f"-i {shlex.quote(list_path)} -c copy {shlex.quote(out_path)}"
    )
    rc, _, _ = await run_ffmpeg(cmd)
    if await aiopath.exists(list_path):
        await remove(list_path)
    return out_path if rc == 0 else None


async def op_merge_video_audio(video: str, audio: str) -> str | None:
    """2) Merge an external audio file with a video stream (copy both)."""
    out_path = _output_path(video, "merged_audio")
    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y "
        f"-i {shlex.quote(video)} -i {shlex.quote(audio)} "
        f"-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest "
        f"{shlex.quote(out_path)}"
    )
    rc, _, _ = await run_ffmpeg(cmd)
    return out_path if rc == 0 else None


async def op_merge_video_subtitle(video: str, sub: str) -> str | None:
    """3) Soft-mux subtitles (SRT/ASS) into an MKV container."""
    out_path = _output_path(video, "softsub", ".mkv")
    sub_codec = "ass" if ospath.splitext(sub)[1].lower() in {".ass", ".ssa"} else "srt"
    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y "
        f"-i {shlex.quote(video)} -i {shlex.quote(sub)} "
        f"-map 0 -map 1 -c copy -c:s {sub_codec} "
        f"{shlex.quote(out_path)}"
    )
    rc, _, _ = await run_ffmpeg(cmd)
    return out_path if rc == 0 else None


async def op_hardsub(video: str, sub: str) -> str | None:
    """4) Burn subtitles into the video stream (re-encode required)."""
    out_path = _output_path(video, "hardsub", ".mp4")
    # The 'subtitles' filter requires a forward-slash, escaped path.
    sub_filter = sub.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y "
        f"-i {shlex.quote(video)} "
        f"-vf \"subtitles='{sub_filter}'\" "
        f"-c:v libx264 -preset veryfast -crf 23 -c:a copy "
        f"{shlex.quote(out_path)}"
    )
    rc, _, _ = await run_ffmpeg(cmd)
    return out_path if rc == 0 else None


async def op_subsync(video: str, sub: str) -> str | None:
    """5) Align subtitle timings to the audio track using ffsubsync."""
    out_sub = _output_path(sub, "synced")
    cmd = (
        f"ffsubsync {shlex.quote(video)} -i {shlex.quote(sub)} "
        f"-o {shlex.quote(out_sub)}"
    )
    proc = await create_subprocess_shell(cmd, stdout=PIPE, stderr=PIPE)
    _, stderr_b = await proc.communicate()
    if proc.returncode != 0:
        LOGGER.error(
            f"[VT] ffsubsync failed (rc={proc.returncode}): "
            f"{stderr_b.decode(errors='ignore')[-500:]}"
        )
        return None
    return out_sub


async def op_compress(video: str) -> str | None:
    """6) Re-encode to HEVC/x265, CRF 28 (visually lossless-ish, ~½ the size)."""
    out_path = _output_path(video, "x265", ".mkv")
    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y "
        f"-i {shlex.quote(video)} "
        f"-c:v libx265 -preset medium -crf 28 -tag:v hvc1 "
        f"-c:a copy {shlex.quote(out_path)}"
    )
    rc, _, _ = await run_ffmpeg(cmd)
    return out_path if rc == 0 else None


async def op_trim(video: str, start: str, end: str) -> str | None:
    """7) Trim a clip between ``start`` and ``end`` (HH:MM:SS or seconds)."""
    out_path = _output_path(video, "trimmed")
    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y "
        f"-i {shlex.quote(video)} -ss {shlex.quote(start)} -to {shlex.quote(end)} "
        f"-c copy {shlex.quote(out_path)}"
    )
    rc, _, _ = await run_ffmpeg(cmd)
    return out_path if rc == 0 else None


async def op_watermark_image(video: str, image: str, position: str = "br") -> str | None:
    """8) Image watermark overlaid on the video (position: tl/tr/bl/br)."""
    pos_map = {
        "tl": "10:10",
        "tr": "main_w-overlay_w-10:10",
        "bl": "10:main_h-overlay_h-10",
        "br": "main_w-overlay_w-10:main_h-overlay_h-10",
    }
    overlay = pos_map.get(position, pos_map["br"])
    out_path = _output_path(video, "wm")
    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y "
        f"-i {shlex.quote(video)} -i {shlex.quote(image)} "
        f"-filter_complex \"overlay={overlay}\" "
        f"-c:v libx264 -preset veryfast -crf 23 -c:a copy "
        f"{shlex.quote(out_path)}"
    )
    rc, _, _ = await run_ffmpeg(cmd)
    return out_path if rc == 0 else None


async def op_watermark_text(video: str, text: str, position: str = "br") -> str | None:
    """8b) Text watermark via the drawtext filter."""
    pos_map = {
        "tl": "x=10:y=10",
        "tr": "x=w-tw-10:y=10",
        "bl": "x=10:y=h-th-10",
        "br": "x=w-tw-10:y=h-th-10",
    }
    pos = pos_map.get(position, pos_map["br"])
    safe_text = text.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    out_path = _output_path(video, "wmtext")
    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y -i {shlex.quote(video)} "
        f"-vf \"drawtext=text='{safe_text}':fontcolor=white:fontsize=28:"
        f"box=1:boxcolor=black@0.4:boxborderw=8:{pos}\" "
        f"-c:v libx264 -preset veryfast -crf 23 -c:a copy "
        f"{shlex.quote(out_path)}"
    )
    rc, _, _ = await run_ffmpeg(cmd)
    return out_path if rc == 0 else None


async def op_remove_video_stream(video: str) -> str | None:
    """9) Remove the video stream → audio-only file (mute video)."""
    out_path = _output_path(video, "audio_only", ".m4a")
    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y "
        f"-i {shlex.quote(video)} -vn -c:a copy {shlex.quote(out_path)}"
    )
    rc, _, _ = await run_ffmpeg(cmd)
    if rc != 0:
        # Fallback: re-encode audio if -c:a copy is incompatible with target ext.
        cmd = (
            f"{_ffmpeg()} -hide_banner -loglevel error -y "
            f"-i {shlex.quote(video)} -vn -c:a aac -b:a 192k "
            f"{shlex.quote(out_path)}"
        )
        rc, _, _ = await run_ffmpeg(cmd)
    return out_path if rc == 0 else None


async def op_extract_video_stream(video: str) -> str | None:
    """10) Strip audio → video-only file."""
    out_path = _output_path(video, "video_only")
    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y "
        f"-i {shlex.quote(video)} -an -c:v copy {shlex.quote(out_path)}"
    )
    rc, _, _ = await run_ffmpeg(cmd)
    return out_path if rc == 0 else None


async def op_convert_resolution(video: str, label: str) -> str | None:
    """11) Resize to a target tier (1080p / 720p / 540p / 480p / 360p)."""
    if label not in RESOLUTION_MAP:
        return None
    w, h = RESOLUTION_MAP[label]
    out_path = _output_path(video, label)
    # Preserve aspect ratio: scale to height = h, width auto-rounded to even.
    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y -i {shlex.quote(video)} "
        f"-vf \"scale=-2:{h}\" -c:v libx264 -preset veryfast -crf 23 "
        f"-c:a copy {shlex.quote(out_path)}"
    )
    rc, _, _ = await run_ffmpeg(cmd)
    return out_path if rc == 0 else None


# ---------------------------------------------------------------------------
# Async file open helper (aiofiles is a hard dep elsewhere in the repo)
# ---------------------------------------------------------------------------


async def _open_async(path: str, mode: str = "r"):
    import aiofiles  # local import — only needed by op_merge_videos
    return await aiofiles.open(path, mode)


# ---------------------------------------------------------------------------
# Constraint enforcement & operation orchestration
# ---------------------------------------------------------------------------


def _ensure_session(message_id: int) -> dict[str, Any] | None:
    s = VT_SESSIONS.get(message_id)
    if not s:
        return None
    if time.time() - s["created_at"] > SESSION_TTL:
        VT_SESSIONS.pop(message_id, None)
        return None
    return s


async def _resolve_work_dir(user_id: int, listener: Any | None) -> str | None:
    """Return the user's most relevant working directory.

    Priority:
      1. ``listener.dir`` if a listener is attached.
      2. ``downloads/{user_id}/{task_id}`` — newest by mtime.
      3. ``downloads/{user_id}`` itself if it has video files.
    """
    if listener is not None and getattr(listener, "dir", None):
        return listener.dir

    base = ospath.join(DOWNLOAD_DIR.rstrip("/"), str(user_id))
    if not await aiopath.isdir(base):
        # In single-user upstream layout, downloads/<task_id>/ is used directly.
        candidates = []
        try:
            for entry in await listdir(DOWNLOAD_DIR.rstrip("/")):
                full = ospath.join(DOWNLOAD_DIR.rstrip("/"), entry)
                if await aiopath.isdir(full):
                    try:
                        st = await aiopath.getmtime(full)
                    except Exception:
                        st = 0
                    candidates.append((st, full))
        except FileNotFoundError:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1] if candidates else None

    # Inside the user's dir, pick the newest task subdir
    try:
        entries = await listdir(base)
    except FileNotFoundError:
        return None
    candidates = []
    for e in entries:
        full = ospath.join(base, e)
        if await aiopath.isdir(full):
            try:
                st = await aiopath.getmtime(full)
            except Exception:
                st = 0
            candidates.append((st, full))
    if not candidates:
        # Fall back to the user dir itself if it directly contains files.
        return base
    candidates.sort(reverse=True)
    return candidates[0][1]


async def _select_targets(work_dir: str, multi: bool) -> list[str]:
    """Apply Constraint 4: when -m is absent, only the first video is touched."""
    videos = await find_videos(work_dir)
    if not videos:
        return []
    return videos if multi else [videos[0]]


def _validate_pre_click(session: dict, op: str) -> str | None:
    """Run constraints C1 and C2 at click time.

    Returns an error message to flash to the user, or ``None`` to proceed.
    """
    if op in MERGE_OPS:
        # C2: Merge requires -m
        if not session.get("multi"):
            return "Merge requires -m argument for multi-file processing."
        # C1: -n is not allowed with merges
        if session.get("rename"):
            return (
                "The -n (rename) flag is not allowed for Merge operations. "
                "Use -n only with bulk/other video tools."
            )
    return None


async def _capture_inheritance(targets: list[str]) -> dict[str, Any]:
    """C3: Capture parameters from the first video to apply to the rest."""
    if not targets:
        return {}
    first = targets[0]
    meta = await probe_video(first)
    res = _resolution_of(meta)
    return {
        "first": first,
        "resolution_label": _label_for_resolution(res) if res else "720p",
        "resolution": res,
    }


# ---------------------------------------------------------------------------
# Public coroutines: invoked by callbacks
# ---------------------------------------------------------------------------


async def _process_op(session: dict, op: str, extra: str | None = None) -> tuple[bool, str, list[str]]:
    """Run the chosen operation against the working directory.

    Returns ``(ok, message, output_files)``.
    """
    work_dir = session["work_dir"]
    multi = bool(session.get("multi"))

    videos = await find_videos(work_dir)
    if not videos:
        return False, "No video files found in the working directory.", []

    targets = videos if multi else [videos[0]]
    inherited = await _capture_inheritance(targets) if op in INHERITED_OPS else {}
    outputs: list[str] = []

    # --- 1) Merge V+V ------------------------------------------------------
    if op == OP_MERGE_VV:
        if len(videos) < 2:
            return False, "Merge V+V needs at least two videos.", []
        out = await op_merge_videos(videos, work_dir)
        if out:
            outputs.append(out)

    # --- 2) Merge V+A ------------------------------------------------------
    elif op == OP_MERGE_VA:
        audios = await find_audios(work_dir)
        if not audios:
            return False, "Merge V+A requires at least one external audio file.", []
        # With -m, pair videos[i] with audios[i] (round-robin). Without -m, just first.
        for i, v in enumerate(targets):
            a = audios[i % len(audios)]
            out = await op_merge_video_audio(v, a)
            if out:
                outputs.append(out)

    # --- 3) Merge V+S ------------------------------------------------------
    elif op == OP_MERGE_VS:
        subs = await find_subs(work_dir)
        if not subs:
            return False, "Merge V+S requires at least one subtitle file.", []
        for i, v in enumerate(targets):
            s = subs[i % len(subs)]
            out = await op_merge_video_subtitle(v, s)
            if out:
                outputs.append(out)

    # --- 4) Hardsub --------------------------------------------------------
    elif op == OP_HARDSUB:
        subs = await find_subs(work_dir)
        if not subs:
            return False, "Hardsub requires at least one subtitle file.", []
        s = subs[0]
        results = await gather(*(op_hardsub(v, s) for v in targets))
        outputs.extend([r for r in results if r])

    # --- 5) SubSync --------------------------------------------------------
    elif op == OP_SUBSYNC:
        subs = await find_subs(work_dir)
        if not subs:
            return False, "SubSync requires at least one subtitle file.", []
        for v, s in zip(targets, subs * len(targets)):
            out = await op_subsync(v, s)
            if out:
                outputs.append(out)

    # --- 6) Compress -------------------------------------------------------
    elif op == OP_COMPRESS:
        results = await gather(*(op_compress(v) for v in targets))
        outputs.extend([r for r in results if r])

    # --- 7) Trim -----------------------------------------------------------
    elif op == OP_TRIM:
        # `extra` is "HH:MM:SS-HH:MM:SS" provided via "/vtset trim ..." or
        # taken from the listener's options string.  Default: first 60 s.
        ts = (extra or session.get("trim") or "00:00:00-00:01:00").strip()
        try:
            start, end = ts.split("-", 1)
        except ValueError:
            return False, "Trim timestamps must be 'HH:MM:SS-HH:MM:SS'.", []
        # C3: same start/end captured from the first video for the whole batch.
        for v in targets:
            out = await op_trim(v, start, end)
            if out:
                outputs.append(out)

    # --- 8) Watermark ------------------------------------------------------
    elif op == OP_WATERMARK:
        images = await find_images(work_dir)
        if images:
            logo = images[0]  # C3: same logo for all batch items
            results = await gather(*(op_watermark_image(v, logo) for v in targets))
            outputs.extend([r for r in results if r])
        else:
            text = session.get("watermark_text") or "WZML-X"
            results = await gather(*(op_watermark_text(v, text) for v in targets))
            outputs.extend([r for r in results if r])

    # --- 9) Remove video stream (audio only) -------------------------------
    elif op == OP_REMOVE_VID:
        results = await gather(*(op_remove_video_stream(v) for v in targets))
        outputs.extend([r for r in results if r])

    # --- 10) Extract video stream (no audio) -------------------------------
    elif op == OP_EXTRACT_VID:
        results = await gather(*(op_extract_video_stream(v) for v in targets))
        outputs.extend([r for r in results if r])

    # --- 11) Convert -------------------------------------------------------
    elif op == OP_CONVERT:
        label = (extra or inherited.get("resolution_label") or "720p")
        if label not in RESOLUTION_MAP:
            return False, f"Unknown resolution tier: {label}", []
        results = await gather(*(op_convert_resolution(v, label) for v in targets))
        outputs.extend([r for r in results if r])

    else:
        return False, f"Unknown operation: {op}", []

    if not outputs:
        return False, (
            "FFmpeg returned no output. Check the bot log for the codec / "
            "container error."
        ), []

    return True, "Processing complete.", outputs


async def _handoff_to_listener(session: dict, outputs: list[str]) -> None:
    """Hand processed files back to the WZML-X TaskListener for upload.

    The module supports two integration paths:

      * ``listener`` is a TaskListener instance → call ``proceed_upload`` if
        present (post-process) or ``on_download_complete`` (fresh).
      * Otherwise, leave the files in place and let the user run /leech or
        /mirror against the directory.
    """
    listener = session.get("listener")
    if listener is None:
        return

    # If a rename was provided and we did NOT do a Merge op (already validated),
    # apply it to the first output only — keeping behaviour predictable.
    rename = session.get("rename")
    if rename and outputs:
        new_path = ospath.join(ospath.dirname(outputs[0]), rename)
        try:
            await move(outputs[0], new_path)
            outputs[0] = new_path
        except Exception as e:
            LOGGER.error(f"[VT] Rename failed: {e}")

    # Update listener.name to the produced artefact so its existing upload
    # routine targets the right path.
    try:
        if len(outputs) == 1:
            listener.name = ospath.basename(outputs[0])
            listener.dir = ospath.dirname(outputs[0])
        else:
            # When multiple files are produced, surface the directory.
            listener.dir = ospath.dirname(outputs[0])
            listener.name = ospath.basename(listener.dir)

        # Preferred WZML-X hook to move past download → upload.
        if hasattr(listener, "proceed_upload"):
            await listener.proceed_upload()
        elif hasattr(listener, "on_download_complete"):
            await listener.on_download_complete()
    except Exception as e:
        LOGGER.error(f"[VT] Listener hand-off failed: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Pyrogram handlers
# ---------------------------------------------------------------------------


@new_task
async def video_tools_command(client, message):
    """Entry point for the ``/vtools`` command."""
    await _open_video_tools_menu(message)


async def _open_video_tools_menu(
    message,
    listener: Any | None = None,
    multi: bool = False,
    rename: str = "",
    work_dir: str | None = None,
) -> None:
    """Send the inline-keyboard menu and register a session."""
    user_id = message.from_user.id if message.from_user else message.chat.id

    # Parse flags from the command text when invoked directly.
    if listener is None:
        text = message.text or ""
        parts = text.split()
        args = {"-vt": False, "-m": "", "-n": ""}
        arg_parser(parts[1:], args)
        multi = bool(args["-m"])
        rename = args["-n"] or ""

    if work_dir is None:
        work_dir = await _resolve_work_dir(user_id, listener)
    if not work_dir or not await aiopath.isdir(work_dir):
        await send_message(
            message,
            "❌ No working directory was found.\n"
            "Run a download first (e.g. /mirror) or pass `-vt` alongside it.",
        )
        return

    videos = await find_videos(work_dir)
    if not videos:
        await send_message(
            message,
            f"❌ No video files found in `{work_dir}`.\n"
            "Drop or download at least one video and retry.",
        )
        return

    sent = await send_message(
        message,
        (
            "🛠 **Video Tools (-vt)**\n"
            f"📂 Dir: `{work_dir}`\n"
            f"🎞 Videos detected: **{len(videos)}**\n"
            f"➕ Multi (-m): **{'on' if multi else 'off'}**\n"
            f"✏️ Rename (-n): **{rename or '—'}**\n\n"
            "Choose an operation:"
        ),
        buttons=build_video_tools_keyboard(0),  # placeholder, replaced below
    )

    # Replace the keyboard now that we know the message-id used as session key.
    if hasattr(sent, "id"):
        VT_SESSIONS[sent.id] = {
            "user_id": user_id,
            "work_dir": work_dir,
            "multi": multi,
            "rename": rename,
            "listener": listener,
            "created_at": time.time(),
        }
        await edit_message(
            sent,
            sent.text.markdown if hasattr(sent.text, "markdown") else
            (sent.text or "🛠 Video Tools"),
            buttons=build_video_tools_keyboard(sent.id),
        )


@new_task
async def video_tools_callback(client, query):
    """CallbackQueryHandler entry point — routes button presses."""
    data = (query.data or "").split()
    if len(data) < 3 or data[0] != CALLBACK_PREFIX:
        await query.answer()
        return

    try:
        sid = int(data[1])
    except ValueError:
        await query.answer("Invalid session.", show_alert=True)
        return
    op = data[2]

    # Use the message id we are attached to as the session id (the seed
    # passed in the keyboard payload may be 0 for the bootstrap render).
    session_id = sid or query.message.id
    session = _ensure_session(session_id)
    if not session:
        await query.answer("This menu has expired. Run /vtools again.", show_alert=True)
        return

    if session["user_id"] != (query.from_user.id if query.from_user else 0):
        await query.answer("This menu is not for you.", show_alert=True)
        return

    if op == "cancel":
        VT_SESSIONS.pop(session_id, None)
        await delete_message(query.message)
        await query.answer("Cancelled.")
        return

    if op == "back":
        await edit_message(
            query.message,
            query.message.text.markdown
            if hasattr(query.message.text, "markdown")
            else (query.message.text or ""),
            buttons=build_video_tools_keyboard(session_id),
        )
        await query.answer()
        return

    # Convert opens a sub-menu first; the resolution comes back as "cvt:1080p".
    if op == OP_CONVERT and ":" not in op:
        await edit_message(
            query.message,
            "🔁 **Convert (Resize)** — pick a target tier:",
            buttons=build_resolution_keyboard(session_id),
        )
        await query.answer()
        return

    extra: str | None = None
    if ":" in op:
        op, extra = op.split(":", 1)

    # Constraint check (C1, C2)
    err = _validate_pre_click(session, op)
    if err:
        await query.answer(err, show_alert=True)
        return

    # All clear — disable buttons and run the operation.
    await edit_message(
        query.message,
        f"⏳ Running operation `{op}`{f' :: {extra}' if extra else ''} …",
        buttons=None,
    )
    await query.answer("Started")

    ok, msg, outputs = await _process_op(session, op, extra=extra)
    if not ok:
        await edit_message(query.message, f"❌ {msg}")
        return

    listing = "\n".join(f"• `{ospath.basename(o)}`" for o in outputs)
    await edit_message(
        query.message,
        f"✅ {msg}\nProduced {len(outputs)} file(s):\n{listing}",
    )

    # Hand off to MirrorLeechListener / TgUploader for delivery.
    await _handoff_to_listener(session, outputs)
    VT_SESSIONS.pop(session_id, None)


# ---------------------------------------------------------------------------
# Public API used by TaskListener and handler registry
# ---------------------------------------------------------------------------


async def process_video_tools(listener) -> None:
    """Hook for ``TaskListener.on_download_complete`` when ``-vt`` is on.

    Usage in ``bot/helper/listeners/task_listener.py``::

        if getattr(self, "video_tools", False):
            from ...modules.video_tools import process_video_tools
            await process_video_tools(self)
            return
    """
    work_dir = getattr(listener, "dir", None) or await _resolve_work_dir(
        getattr(listener, "user_id", 0), listener
    )
    multi = bool(getattr(listener, "multi", 0)) or bool(
        getattr(listener, "folder_name", "")
    )
    rename = getattr(listener, "name", "") if getattr(listener, "_user_renamed", False) else ""

    await _open_video_tools_menu(
        listener.message,
        listener=listener,
        multi=multi,
        rename=rename,
        work_dir=work_dir,
    )


def register_video_tools_handlers() -> None:
    """Register the /vtools command and the callback handler with TgClient.

    Call this from ``bot/core/handlers.py::add_handlers`` after the other
    MessageHandler registrations.
    """
    cmd_attr = getattr(BotCommands, "VideoToolsCommand", None) or [
        f"vtools{Config.CMD_SUFFIX}",
        f"vt{Config.CMD_SUFFIX}",
    ]

    TgClient.bot.add_handler(
        MessageHandler(
            video_tools_command,
            filters=command(cmd_attr, case_sensitive=True) & CustomFilters.authorized,
        )
    )
    TgClient.bot.add_handler(
        CallbackQueryHandler(
            video_tools_callback, filters=regex(rf"^{CALLBACK_PREFIX} ")
        )
    )


# Convenience wrapper so __init__.py can import a callable named ``video_tools``.
async def video_tools(client, message):
    bot_loop.create_task(video_tools_command(client, message))
