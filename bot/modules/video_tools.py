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

import asyncio
import json
import re
import shlex
import time
from asyncio import create_subprocess_shell, gather
from asyncio.subprocess import PIPE
from os import path as ospath, walk
from typing import Any, Awaitable, Callable

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

# Common ISO 639-2/B (3-letter) language codes used for Merge V+A and V+S
# metadata tagging. The flag emoji is purely cosmetic on the button.
COMMON_LANGS: list[tuple[str, str, str]] = [
    ("🇬🇧", "eng", "English"),
    ("🇯🇵", "jpn", "Japanese"),
    ("🇮🇳", "hin", "Hindi"),
    ("🇧🇩", "ben", "Bengali"),
    ("🇪🇸", "spa", "Spanish"),
    ("🇫🇷", "fre", "French"),
    ("🇩🇪", "ger", "German"),
    ("🇨🇳", "chi", "Chinese"),
    ("🇰🇷", "kor", "Korean"),
    ("🇷🇺", "rus", "Russian"),
    ("🇸🇦", "ara", "Arabic"),
    ("🇵🇹", "por", "Portuguese"),
    ("🇮🇹", "ita", "Italian"),
    ("🇹🇷", "tur", "Turkish"),
    ("🇮🇩", "ind", "Indonesian"),
    ("🇹🇭", "tha", "Thai"),
]

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
OP_CUSTOM_EX  = "cex"   # custom multi-stream extraction

# Sub-actions for the custom-extract sub-menu
CEX_TOGGLE = "cextog"   # toggle a single stream selection
CEX_RUN    = "cexrun"   # run extraction with the current selection
CEX_ALL    = "cexall"   # select all streams of a kind ("a" / "s" / "v")
CEX_NONE   = "cexnone"  # clear selection

# Sub-actions for the merge-V+A / V+S language picker
MRG_LANG = "mlang"     # user picked a language for the merge op

# Sub-actions for the interactive Trim flow
TRM_PRESET = "trmpre"   # quick preset (e.g. trmpre:0:60)
TRM_INPUT  = "trminp"   # user is sending custom timestamps as a chat reply
TRM_CANCEL = "trmcan"   # cancel a running trim job

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
    """Build the inline keyboard for the video tools menu.

    Layout (organised rows):

        Row 1 — Merge V+V        | Merge V+A
        Row 2 — Merge V+S        | Hardsub (sudo)
        Row 3 — SubSync          | Compress (HEVC)
        Row 4 — Trim             | Watermark
        Row 5 — Remove Video     | Extract Video
        Row 6 — Custom Extract   | Convert (Resize)
        Footer — Cancel
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

    buttons.data_button("🎯 Custom Extract Streams", f"{CALLBACK_PREFIX} {s} {OP_CUSTOM_EX}")
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


def build_merge_lang_keyboard(session_id: int, op: str) -> Any:
    """Pick a language tag for the upcoming Merge V+A or V+S operation.

    The chosen ISO-639-2 code travels through the callback as the ``extra``
    component, e.g. ``vt <sid> mlang:mva:eng`` or ``…:mvs:jpn``.  Picking
    "Skip" sends an empty extra so the merge runs with ``language=und``.
    """
    buttons = ButtonMaker()
    for flag, code, name in COMMON_LANGS:
        buttons.data_button(
            f"{flag} {name}",
            f"{CALLBACK_PREFIX} {session_id} {MRG_LANG}:{op}:{code}",
        )
    buttons.data_button(
        "🌐 Undefined (skip)",
        f"{CALLBACK_PREFIX} {session_id} {MRG_LANG}:{op}:und",
        "footer",
    )
    buttons.data_button("⬅️ Back", f"{CALLBACK_PREFIX} {session_id} back", "footer")
    return buttons.build_menu(b_cols=2, f_cols=2)


# ---------------------------------------------------------------------------
# Trim picker + progress reporting + cancellable ffmpeg runner
# ---------------------------------------------------------------------------


# Per-session state used while a long-running trim (or any future cancellable
# op) is in flight. Keys:
#   "trim_proc"    -> asyncio subprocess handle (for terminate)
#   "trim_cancel"  -> asyncio.Event flagged when user taps Cancel
#   "trim_total"   -> source duration in seconds (for % progress)
#   "trim_msg"     -> the bot message we're updating
TRIM_PRESETS: list[tuple[str, str]] = [
    # (button label, "start-end" in HH:MM:SS)
    ("🎯 First 30s",   "00:00:00-00:00:30"),
    ("🎯 First 1 min", "00:00:00-00:01:00"),
    ("🎯 First 5 min", "00:00:00-00:05:00"),
    ("⏯️ 0:00 → 10:00", "00:00:00-00:10:00"),
    ("⏯️ 1:00 → 5:00",  "00:01:00-00:05:00"),
]


def build_trim_keyboard(session_id: int) -> Any:
    """Sub-menu for the Trim action.

    Top body: quick presets (one row per preset).
    l_body: a "Custom range…" prompt (cancelled by Back).
    Footer: Back.
    """
    buttons = ButtonMaker()
    for label, ts in TRIM_PRESETS:
        # callback: ``vt <sid> trmpre:00:00:00-00:00:30``
        buttons.data_button(
            label,
            f"{CALLBACK_PREFIX} {session_id} {TRM_PRESET}:{ts}",
        )
    buttons.data_button(
        "✏️ Custom range (reply with HH:MM:SS-HH:MM:SS)",
        f"{CALLBACK_PREFIX} {session_id} {TRM_INPUT}",
        "l_body",
    )
    buttons.data_button("⬅️ Back", f"{CALLBACK_PREFIX} {session_id} back", "footer")
    return buttons.build_menu(b_cols=1, lb_cols=1, f_cols=1)


def build_progress_keyboard(session_id: int) -> Any:
    """A single 'Terminate' button shown next to a progress bar."""
    buttons = ButtonMaker()
    buttons.data_button(
        "🛑 Terminate",
        f"{CALLBACK_PREFIX} {session_id} {TRM_CANCEL}",
        "footer",
    )
    return buttons.build_menu(f_cols=1)


def _hms_to_seconds(ts: str) -> float | None:
    """Parse HH:MM:SS(.ms) or MM:SS or seconds → float seconds (or None)."""
    ts = ts.strip()
    if not ts:
        return None
    try:
        if ":" not in ts:
            return float(ts)
        parts = ts.split(":")
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        return None
    return None


def _seconds_to_hms(secs: float) -> str:
    secs = max(0.0, float(secs))
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _progress_bar(pct: float, width: int = 18) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct / 100.0 * width))
    return "█" * filled + "░" * (width - filled)


_FFMPEG_TIME_RE = re.compile(r"out_time_ms=(\d+)")


async def run_ffmpeg_with_progress(
    cmd: str,
    duration_sec: float,
    on_progress: Callable[[float, float], Awaitable[None]],
    cancel_event: asyncio.Event,
) -> tuple[int, str]:
    """Run ``cmd`` (an ffmpeg command string), parse ``-progress pipe:1`` output
    on stdout, call ``on_progress(elapsed_sec, pct)`` for live updates, and
    terminate the process if ``cancel_event`` is set.

    The caller must include ``-progress pipe:1 -nostats`` in ``cmd`` so ffmpeg
    emits machine-parsable lines like ``out_time_ms=1234567`` every ~500 ms.

    Returns ``(returncode, last_stderr_tail)``.
    """
    LOGGER.info(f"[VT] FFmpeg(progress) → {cmd}")
    proc = await create_subprocess_shell(cmd, stdout=PIPE, stderr=PIPE)

    last_pct = -1.0

    async def _watch_cancel():
        await cancel_event.wait()
        if proc.returncode is None:
            try:
                proc.terminate()
                # If terminate doesn't take effect quickly, escalate to kill.
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    proc.kill()
            except ProcessLookupError:
                pass

    cancel_task = asyncio.create_task(_watch_cancel())

    try:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode(errors="ignore").strip()
            if not text:
                continue
            m = _FFMPEG_TIME_RE.search(text)
            if not m:
                continue
            elapsed = int(m.group(1)) / 1_000_000.0
            pct = (elapsed / duration_sec * 100.0) if duration_sec > 0 else 0.0
            # Throttle by integer-percent change to avoid edit_message spam.
            if int(pct) != int(last_pct):
                last_pct = pct
                try:
                    await on_progress(elapsed, pct)
                except Exception as e:
                    LOGGER.debug(f"[VT] progress callback dropped update: {e}")
    finally:
        cancel_task.cancel()
        try:
            await cancel_task
        except (asyncio.CancelledError, BaseException):
            pass

    stderr_b = await proc.stderr.read() if proc.stderr else b""
    rc = await proc.wait()
    stderr_tail = stderr_b.decode(errors="ignore")[-800:] if stderr_b else ""
    if rc != 0 and not cancel_event.is_set():
        LOGGER.error(f"[VT] FFmpeg(progress) failed (rc={rc}): {stderr_tail}")
    return rc, stderr_tail


async def _await_user_text_reply(
    client,
    session: dict,
    chat_id: int,
    user_id: int,
    timeout: float = 120.0,
) -> str | None:
    """Wait for a single text message from ``user_id`` in ``chat_id``.

    Used by the Trim "custom range" flow. Returns the trimmed text, or None
    on timeout / handler removal. The handler is removed exactly once,
    whether the message arrives or not.
    """
    from pyrogram.handlers import MessageHandler as _MH
    from pyrogram.filters import (
        chat as _f_chat,
        text as _f_text,
        user as _f_user,
    )

    fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    async def _on_msg(_, message):
        if not fut.done():
            fut.set_result((message.text or "").strip())

    handler = _MH(_on_msg, _f_chat(chat_id) & _f_user(user_id) & _f_text)
    handler_id = client.add_handler(handler, group=-1)

    try:
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return None
    finally:
        try:
            client.remove_handler(*handler_id)
        except Exception:
            pass


async def op_trim_with_progress(
    video: str,
    start: str,
    end: str,
    *,
    on_progress: Callable[[float, float], Awaitable[None]],
    cancel_event: asyncio.Event,
    register_proc: Callable[[Any], None] | None = None,
) -> tuple[str | None, bool]:
    """Cancellable trim with live progress.

    Returns ``(out_path_or_None, was_cancelled)``.  When the user terminates,
    the partial output file is removed before returning.
    """
    out_path = _output_path(video, "trimmed")
    duration_sec = max(0.0, (_hms_to_seconds(end) or 0.0) - (_hms_to_seconds(start) or 0.0))

    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y "
        f"-i {shlex.quote(video)} -ss {shlex.quote(start)} -to {shlex.quote(end)} "
        f"-c copy -progress pipe:1 -nostats "
        f"{shlex.quote(out_path)}"
    )

    rc, _stderr = await run_ffmpeg_with_progress(
        cmd, duration_sec, on_progress, cancel_event
    )

    cancelled = cancel_event.is_set()
    if cancelled or rc != 0:
        # Tidy up the partial file.
        try:
            if await aiopath.exists(out_path):
                await remove(out_path)
        except Exception as e:
            LOGGER.debug(f"[VT] couldn't remove partial trim output: {e}")
        return None, cancelled

    return out_path, False


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


async def op_merge_video_audio(
    video: str,
    audio: str,
    lang: str = "und",
    title: str | None = None,
) -> str | None:
    """2) Merge an external audio file into the video container.

    Behaviour:
      * Output is forced to ``.mkv`` (Matroska is the only widely-deployed
        container that copes with arbitrary codecs *and* attachments).
      * **All** input-0 streams are kept (video, all existing audios, all
        existing subs, all attachments / fonts) via ``-map 0``.
      * The new audio is appended via ``-map 1:a:0`` and tagged with the
        provided ISO-639-2 ``lang`` and a human-readable ``title`` (defaults
        to the audio file's stem so the player picker stays useful).
      * Codecs are copied — no re-encode.
    """
    out_path = _output_path(video, "merged_audio", ".mkv")
    if not title:
        title = ospath.splitext(ospath.basename(audio))[0]

    # The new audio's index inside the *output* file is the count of
    # input-0 streams that came before it. We don't know that without
    # ffprobe, but ffmpeg's relative metadata syntax (``-metadata:s:a:N``
    # where N is the audio-stream index in the output) handles the
    # ordering for us. Tagging by output stream-type index is the most
    # reliable choice.
    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y "
        f"-i {shlex.quote(video)} -i {shlex.quote(audio)} "
        f"-map 0 -map 1:a:0 -c copy "
        f"-metadata:s:a:{{NEW_AUDIO}} language={shlex.quote(lang)} "
        f"-metadata:s:a:{{NEW_AUDIO}} title={shlex.quote(title)} "
        f"-disposition:a:{{NEW_AUDIO}} 0 "
        f"{shlex.quote(out_path)}"
    )
    # Resolve the {NEW_AUDIO} placeholder to the index this new audio will
    # occupy in the output file (= number of audios already in input 0).
    meta = await probe_video(video)
    existing_audios = sum(
        1 for s in meta.get("streams", []) if s.get("codec_type") == "audio"
    )
    cmd = cmd.replace("{NEW_AUDIO}", str(existing_audios))

    rc, _, _ = await run_ffmpeg(cmd)
    return out_path if rc == 0 else None


async def op_merge_video_subtitle(
    video: str,
    sub: str,
    lang: str = "und",
    title: str | None = None,
) -> str | None:
    """3) Soft-mux an external subtitle file into the video container.

    Behaviour:
      * Output is forced to ``.mkv`` so PGS / ASS / SRT / VTT all fit and
        attachments survive.
      * **All** input-0 streams are kept (video, all audios, all existing
        subs, all attachments / fonts) via ``-map 0``.
      * The new subtitle is appended via ``-map 1`` and tagged with the
        provided ``lang`` and ``title``.
      * The new subtitle's codec is copied; existing streams are left
        untouched. SRT/ASS tag the new track for friendly player picking.
    """
    out_path = _output_path(video, "softsub", ".mkv")
    if not title:
        title = ospath.splitext(ospath.basename(sub))[0]

    sub_ext = ospath.splitext(sub)[1].lower()
    # We let ffmpeg auto-pick the subtitle codec on copy. It only needs an
    # explicit ``-c:s`` when *trans-coding* between text formats; with -c
    # copy the source is preserved verbatim.
    sub_codec_flag = ""
    if sub_ext in {".srt", ".vtt"}:
        sub_codec_flag = " -c:s:0 srt"  # only the *new* sub stream
    elif sub_ext in {".ass", ".ssa"}:
        sub_codec_flag = " -c:s:0 ass"

    meta = await probe_video(video)
    existing_subs = sum(
        1 for s in meta.get("streams", []) if s.get("codec_type") == "subtitle"
    )

    cmd = (
        f"{_ffmpeg()} -hide_banner -loglevel error -y "
        f"-i {shlex.quote(video)} -i {shlex.quote(sub)} "
        f"-map 0 -map 1:0 -c copy{sub_codec_flag} "
        f"-metadata:s:s:{existing_subs} language={shlex.quote(lang)} "
        f"-metadata:s:s:{existing_subs} title={shlex.quote(title)} "
        f"-disposition:s:{existing_subs} 0 "
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
# 12) Custom multi-stream extraction
# ---------------------------------------------------------------------------


def _stream_label(stream: dict) -> str:
    """Render a one-liner for an ffprobe stream dict.

    Format::  [V|A|S] #idx · codec · lang · title
    """
    kind_map = {"video": "V", "audio": "A", "subtitle": "S"}
    kind = kind_map.get(stream.get("codec_type"), "?")
    idx = stream.get("index", "?")
    codec = stream.get("codec_name") or "?"
    tags = stream.get("tags") or {}
    lang = tags.get("language") or tags.get("LANGUAGE") or "und"
    title = tags.get("title") or tags.get("TITLE") or ""

    extra: list[str] = []
    if kind == "V":
        w, h = stream.get("width"), stream.get("height")
        if w and h:
            extra.append(f"{w}x{h}")
    elif kind == "A":
        ch = stream.get("channels")
        if ch:
            extra.append(f"{ch}ch")
    extras = " · " + " ".join(extra) if extra else ""

    title_part = f" · {title[:40]}" if title else ""
    return f"[{kind}] #{idx} · {codec} · {lang}{extras}{title_part}"


async def list_streams(video: str) -> list[dict]:
    """Return all streams of the given video file via ffprobe."""
    meta = await probe_video(video)
    streams: list[dict] = []
    for s in meta.get("streams", []):
        if s.get("codec_type") in ("video", "audio", "subtitle"):
            streams.append(s)
    return streams


def _ext_for_stream(stream: dict) -> str:
    """Best-effort container/extension for a single extracted stream."""
    kind = stream.get("codec_type")
    codec = (stream.get("codec_name") or "").lower()
    if kind == "video":
        return {
            "h264": ".h264", "hevc": ".hevc", "h265": ".h265",
            "av1": ".av1", "vp9": ".webm", "vp8": ".webm",
        }.get(codec, ".mkv")
    if kind == "audio":
        return {
            "aac": ".m4a", "mp3": ".mp3", "flac": ".flac",
            "opus": ".opus", "vorbis": ".ogg", "ac3": ".ac3",
            "eac3": ".eac3", "dts": ".dts", "alac": ".m4a",
            "pcm_s16le": ".wav", "pcm_s24le": ".wav",
        }.get(codec, ".mka")
    # subtitle
    return {
        "subrip": ".srt", "srt": ".srt", "ass": ".ass", "ssa": ".ssa",
        "webvtt": ".vtt", "mov_text": ".srt", "hdmv_pgs_subtitle": ".sup",
        "dvd_subtitle": ".sub",
    }.get(codec, ".mks")


async def op_custom_extract(
    video: str, stream_indexes: list[int], all_streams: list[dict]
) -> list[str]:
    """12) Extract one file per selected stream, preserving codec when possible.

    `stream_indexes` are absolute stream indexes from ffprobe (the values
    that sit in ``stream["index"]`` and that ffmpeg accepts after ``-map 0:``).
    """
    by_index = {int(s["index"]): s for s in all_streams}
    base = ospath.splitext(video)[0]
    outputs: list[str] = []

    for idx in stream_indexes:
        s = by_index.get(int(idx))
        if not s:
            continue
        kind = s.get("codec_type", "?")
        tags = s.get("tags") or {}
        lang = tags.get("language") or tags.get("LANGUAGE") or "und"
        ext = _ext_for_stream(s)
        # e.g. movie.s2.eng.srt or movie.a3.jpn.m4a
        suffix = f"{kind[:1]}{idx}.{lang}"
        out_path = f"{base}.{suffix}{ext}"

        # Subtitle text codecs need an explicit codec when going to .srt/.ass
        sub_codec = ""
        if kind == "subtitle":
            target = ext.lstrip(".")
            sub_codec = {
                "srt": " -c:s srt",
                "ass": " -c:s ass",
                "ssa": " -c:s ass",
                "vtt": " -c:s webvtt",
            }.get(target, " -c:s copy")

        cmd = (
            f"{_ffmpeg()} -hide_banner -loglevel error -y "
            f"-i {shlex.quote(video)} -map 0:{int(idx)} "
            f"-c copy{sub_codec} {shlex.quote(out_path)}"
        )
        rc, _, _ = await run_ffmpeg(cmd)
        if rc == 0:
            outputs.append(out_path)
        else:
            # Last-ditch retry without -c copy for tricky containers.
            retry = (
                f"{_ffmpeg()} -hide_banner -loglevel error -y "
                f"-i {shlex.quote(video)} -map 0:{int(idx)} "
                f"{shlex.quote(out_path)}"
            )
            rc2, _, _ = await run_ffmpeg(retry)
            if rc2 == 0:
                outputs.append(out_path)
    return outputs


def build_custom_extract_keyboard(
    session_id: int, streams: list[dict], selected: set[int]
) -> Any:
    """Build the toggleable stream-picker keyboard.

    Each stream gets one button labelled with kind/codec/lang and prefixed by
    a check-mark when selected. Footer rows offer All-Audio / All-Subs /
    Clear and the Run / Back actions.
    """
    buttons = ButtonMaker()

    for s in streams:
        idx = int(s["index"])
        prefix = "✅ " if idx in selected else "▫️ "
        # callback: vt <sid> cextog <stream_idx>
        buttons.data_button(
            prefix + _stream_label(s),
            f"{CALLBACK_PREFIX} {session_id} {CEX_TOGGLE} {idx}",
        )

    # Bulk helpers (l_body row, 3 across)
    buttons.data_button("🔊 All Audio", f"{CALLBACK_PREFIX} {session_id} {CEX_ALL} a", "l_body")
    buttons.data_button("📝 All Subtitles", f"{CALLBACK_PREFIX} {session_id} {CEX_ALL} s", "l_body")
    buttons.data_button("🧹 Clear", f"{CALLBACK_PREFIX} {session_id} {CEX_NONE}", "l_body")

    # Action row
    buttons.data_button(
        f"⚡ Extract Selected ({len(selected)})",
        f"{CALLBACK_PREFIX} {session_id} {CEX_RUN}",
        "footer",
    )
    buttons.data_button("⬅️ Back", f"{CALLBACK_PREFIX} {session_id} back", "footer")

    return buttons.build_menu(b_cols=1, lb_cols=3, f_cols=2)


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
        # `extra` arrives as the ISO 639-2 code picked from build_merge_lang_keyboard,
        # or empty/None when invoked without a picker (defaults to "und").
        lang = (extra or "und").strip() or "und"
        # With -m, pair videos[i] with audios[i] (round-robin). Without -m, just first.
        for i, v in enumerate(targets):
            a = audios[i % len(audios)]
            out = await op_merge_video_audio(v, a, lang=lang)
            if out:
                outputs.append(out)

    # --- 3) Merge V+S ------------------------------------------------------
    elif op == OP_MERGE_VS:
        subs = await find_subs(work_dir)
        if not subs:
            return False, "Merge V+S requires at least one subtitle file.", []
        lang = (extra or "und").strip() or "und"
        for i, v in enumerate(targets):
            s = subs[i % len(subs)]
            out = await op_merge_video_subtitle(v, s, lang=lang)
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

        # Mark as handled so the listener doesn't re-open the menu on the
        # second pass and instead proceeds straight to upload.
        listener._vt_handled = True

        # Re-enter the standard download-complete flow with the produced
        # files in place. WZML-X's TaskListener.on_download_complete handles
        # the rest (metadata, ffmpeg_cmds, mirror/leech upload, etc.).
        if hasattr(listener, "on_download_complete"):
            await listener.on_download_complete()
        elif hasattr(listener, "proceed_upload"):
            await listener.proceed_upload()
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


async def _run_trim_with_progress(
    client,
    progress_msg,
    session: dict,
    session_id: int,
    video: str,
    start: str,
    end: str,
) -> None:
    """Drive a cancellable trim run end-to-end:

      * Renders an initial progress message with a Terminate button.
      * Streams ffmpeg ``-progress pipe:1`` updates and edits the message
        every full % change (Telegram-friendly throttle).
      * On cancel, the partial output is removed and the message is updated.
      * On success, hands off to the listener for upload.
    """
    cancel_event = asyncio.Event()
    session["trim_cancel"] = cancel_event

    target_dur = max(0.0, (_hms_to_seconds(end) or 0.0) - (_hms_to_seconds(start) or 0.0))
    started_at = time.time()
    base_text = (
        f"✂️ **Trim in progress**\n"
        f"📄 `{ospath.basename(video)}`\n"
        f"🎬 Range: `{start} → {end}`  (≈ {_seconds_to_hms(target_dur)})\n"
    )

    # First paint with 0%.
    await edit_message(
        progress_msg,
        base_text + f"\n`{_progress_bar(0.0)}` **0.0%**\n_starting…_",
        buttons=build_progress_keyboard(session_id),
    )

    # Throttle edits to roughly once a second too — even if % moves by 1.
    last_edit_at = 0.0

    async def _on_progress(elapsed: float, pct: float) -> None:
        nonlocal last_edit_at
        now = time.time()
        if now - last_edit_at < 1.0 and pct < 99.0:
            return
        last_edit_at = now
        eta_text = ""
        if pct > 1.0:
            wall = now - started_at
            eta = max(0.0, wall * (100.0 - pct) / pct)
            eta_text = f"  ·  ETA `{_seconds_to_hms(eta)}`"
        try:
            await edit_message(
                progress_msg,
                base_text
                + f"\n`{_progress_bar(pct)}` **{pct:5.1f}%**"
                + f"\nElapsed `{_seconds_to_hms(elapsed)}` / `{_seconds_to_hms(target_dur)}`"
                + eta_text,
                buttons=build_progress_keyboard(session_id),
            )
        except Exception as e:
            LOGGER.debug(f"[VT] progress edit dropped: {e}")

    try:
        out_path, was_cancelled = await op_trim_with_progress(
            video, start, end,
            on_progress=_on_progress,
            cancel_event=cancel_event,
        )
    finally:
        session.pop("trim_cancel", None)

    if was_cancelled:
        await edit_message(
            progress_msg,
            base_text + "\n🛑 **Cancelled by user.** Partial output removed.",
            buttons=None,
        )
        return
    if not out_path:
        await edit_message(
            progress_msg,
            base_text + "\n❌ FFmpeg failed. Check the bot log for details.",
            buttons=None,
        )
        return

    await edit_message(
        progress_msg,
        base_text
        + f"\n`{_progress_bar(100.0)}` **100.0%**\n"
        + f"✅ Trim complete → `{ospath.basename(out_path)}`",
        buttons=None,
    )
    await _handoff_to_listener(session, [out_path])
    VT_SESSIONS.pop(session_id, None)


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

    # ---- Custom Extract sub-flow ----------------------------------------
    # First click opens the picker; subsequent clicks toggle / run / clear.
    if op == OP_CUSTOM_EX:
        videos = await find_videos(session["work_dir"])
        if not videos:
            await query.answer("No videos found.", show_alert=True)
            return
        target = videos[0]  # C3: probe and apply against the first video
        streams = await list_streams(target)
        if not streams:
            await query.answer(
                "ffprobe found no V/A/S streams in the file.", show_alert=True
            )
            return
        session["cex_target"] = target
        session["cex_streams"] = streams
        session["cex_selected"] = set()
        await edit_message(
            query.message,
            (
                "🎯 **Custom Extract Streams**\n"
                f"📄 File: `{ospath.basename(target)}`\n"
                f"🎚 Streams found: **{len(streams)}**\n\n"
                "Tap a stream to toggle, then **⚡ Extract Selected**.\n"
                "Format: `[V|A|S] #idx · codec · lang · WxH/ch · title`"
            ),
            buttons=build_custom_extract_keyboard(
                session_id, streams, session["cex_selected"]
            ),
        )
        await query.answer()
        return

    if op == CEX_TOGGLE:
        if "cex_streams" not in session:
            await query.answer("Re-open Custom Extract.", show_alert=True)
            return
        try:
            idx = int(data[3])
        except (IndexError, ValueError):
            await query.answer("Bad stream index.", show_alert=True)
            return
        sel: set[int] = session["cex_selected"]
        sel.discard(idx) if idx in sel else sel.add(idx)
        await edit_message(
            query.message,
            query.message.text.markdown
            if hasattr(query.message.text, "markdown")
            else (query.message.text or ""),
            buttons=build_custom_extract_keyboard(
                session_id, session["cex_streams"], sel
            ),
        )
        await query.answer(f"{len(sel)} selected")
        return

    if op == CEX_ALL:
        kind = data[3] if len(data) > 3 else ""
        kind_map = {"v": "video", "a": "audio", "s": "subtitle"}
        if kind not in kind_map or "cex_streams" not in session:
            await query.answer()
            return
        target_kind = kind_map[kind]
        for s in session["cex_streams"]:
            if s.get("codec_type") == target_kind:
                session["cex_selected"].add(int(s["index"]))
        await edit_message(
            query.message,
            query.message.text.markdown
            if hasattr(query.message.text, "markdown")
            else (query.message.text or ""),
            buttons=build_custom_extract_keyboard(
                session_id, session["cex_streams"], session["cex_selected"]
            ),
        )
        await query.answer(f"{len(session['cex_selected'])} selected")
        return

    if op == CEX_NONE:
        if "cex_streams" not in session:
            await query.answer()
            return
        session["cex_selected"] = set()
        await edit_message(
            query.message,
            query.message.text.markdown
            if hasattr(query.message.text, "markdown")
            else (query.message.text or ""),
            buttons=build_custom_extract_keyboard(
                session_id, session["cex_streams"], session["cex_selected"]
            ),
        )
        await query.answer("Cleared")
        return

    if op == CEX_RUN:
        sel = session.get("cex_selected") or set()
        if not sel:
            await query.answer("Pick at least one stream first.", show_alert=True)
            return
        target = session["cex_target"]
        streams = session["cex_streams"]
        await edit_message(
            query.message,
            f"⏳ Extracting {len(sel)} stream(s) from `{ospath.basename(target)}` …",
            buttons=None,
        )
        await query.answer("Started")
        outputs = await op_custom_extract(target, sorted(sel), streams)
        if not outputs:
            await edit_message(
                query.message,
                "❌ No streams were extracted. Check the bot log for ffmpeg errors.",
            )
            return
        listing = "\n".join(f"• `{ospath.basename(o)}`" for o in outputs)
        await edit_message(
            query.message,
            f"✅ Extracted {len(outputs)} stream(s):\n{listing}",
        )
        # Clear the per-flow state and hand off for upload.
        for k in ("cex_target", "cex_streams", "cex_selected"):
            session.pop(k, None)
        await _handoff_to_listener(session, outputs)
        VT_SESSIONS.pop(session_id, None)
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

    # ---- Trim sub-flow --------------------------------------------------
    # Tap "✂️ Trim" → open preset picker + custom-range prompt.
    if op == OP_TRIM:
        videos = await find_videos(session["work_dir"])
        if not videos:
            await query.answer("No videos found.", show_alert=True)
            return
        # ffprobe the first video so we can show the source duration.
        first = videos[0]
        meta = await probe_video(first)
        try:
            dur = float(meta.get("format", {}).get("duration", 0.0) or 0.0)
        except (TypeError, ValueError):
            dur = 0.0
        session["trim_target"] = first
        session["trim_duration"] = dur
        await edit_message(
            query.message,
            (
                "✂️ **Trim Video**\n"
                f"📄 File: `{ospath.basename(first)}`\n"
                f"⏱ Duration: **{_seconds_to_hms(dur) if dur else 'unknown'}**\n\n"
                "Pick a preset, or tap **Custom range** and reply to the prompt "
                "with `HH:MM:SS-HH:MM:SS` (e.g. `00:01:30-00:04:15`)."
            ),
            buttons=build_trim_keyboard(session_id),
        )
        await query.answer()
        return

    # Cancel an in-flight trim run (or any future cancellable op).
    if op == TRM_CANCEL:
        ev: asyncio.Event | None = session.get("trim_cancel")
        if ev is not None and not ev.is_set():
            ev.set()
            await query.answer("Terminating…", show_alert=False)
        else:
            await query.answer("Nothing to cancel.")
        return

    # Custom range prompt → wait for user's text reply, then run with progress.
    if op == TRM_INPUT:
        target = session.get("trim_target") or (
            (await find_videos(session["work_dir"])) or [None]
        )[0]
        if not target:
            await query.answer("No video.", show_alert=True)
            return

        prompt_msg = await edit_message(
            query.message,
            (
                "✏️ **Custom Trim Range**\n"
                f"📄 File: `{ospath.basename(target)}`\n\n"
                "Reply to *this message* with a range in the format\n"
                "  `HH:MM:SS-HH:MM:SS`   or   `MM:SS-MM:SS`\n"
                "_Waiting up to 2 minutes…_"
            ),
            buttons=None,
        )
        await query.answer()

        chat_id = (
            query.message.chat.id if query.message and query.message.chat else session["user_id"]
        )
        text = await _await_user_text_reply(client, session, chat_id, session["user_id"])
        if not text:
            await edit_message(
                prompt_msg or query.message,
                "⏳ Timed out waiting for a custom range. Tap Trim again to retry.",
            )
            return
        # Validate format: must be START-END.
        if "-" not in text:
            await edit_message(
                prompt_msg or query.message,
                "❌ Bad format. Expected `HH:MM:SS-HH:MM:SS`.",
            )
            return
        start, _, end = text.partition("-")
        s_sec = _hms_to_seconds(start)
        e_sec = _hms_to_seconds(end)
        if s_sec is None or e_sec is None or e_sec <= s_sec:
            await edit_message(
                prompt_msg or query.message,
                f"❌ Invalid range `{text}`. End must be after start.",
            )
            return

        await _run_trim_with_progress(
            client, query.message, session, session_id, target,
            _seconds_to_hms(s_sec), _seconds_to_hms(e_sec),
        )
        return

    # Preset trim: callback like "trmpre:00:00:00-00:00:30"
    if op.startswith(TRM_PRESET + ":"):
        target = session.get("trim_target") or (
            (await find_videos(session["work_dir"])) or [None]
        )[0]
        if not target:
            await query.answer("No video.", show_alert=True)
            return
        # Strip the leading "trmpre:" — the rest is the H:M:S-H:M:S range.
        ts_range = op.split(":", 1)[1]
        if "-" not in ts_range:
            await query.answer("Bad preset range.", show_alert=True)
            return
        start, _, end = ts_range.partition("-")
        s_sec = _hms_to_seconds(start)
        e_sec = _hms_to_seconds(end)
        if s_sec is None or e_sec is None or e_sec <= s_sec:
            await query.answer("Invalid range.", show_alert=True)
            return
        await query.answer("Started")
        await _run_trim_with_progress(
            client, query.message, session, session_id, target,
            _seconds_to_hms(s_sec), _seconds_to_hms(e_sec),
        )
        return

    # Merge V+A / V+S open a language picker first; the picked ISO 639-2 code
    # comes back as ``mlang:mva:eng`` (or :mvs: …). The constraints (C1 + C2)
    # are still validated up-front — before opening the picker.
    if op in (OP_MERGE_VA, OP_MERGE_VS):
        err = _validate_pre_click(session, op)
        if err:
            await query.answer(err, show_alert=True)
            return
        kind = "audio" if op == OP_MERGE_VA else "subtitle"
        await edit_message(
            query.message,
            f"🌐 **Pick a language for the new {kind} track**\n"
            f"This tag is written into the MKV's stream metadata so players "
            f"show it in the audio / subtitle picker.",
            buttons=build_merge_lang_keyboard(session_id, op),
        )
        await query.answer()
        return

    # Language picker callback: rewrite the op + extra so the rest of the
    # router treats it as a regular op-with-extra dispatch.
    if op.startswith(MRG_LANG + ":"):
        # data[2] == "mlang:mva:eng"  →  split twice
        try:
            _, real_op, lang_code = op.split(":", 2)
        except ValueError:
            await query.answer("Bad language payload.", show_alert=True)
            return
        op = real_op  # mva or mvs

        # Run the merge with the language as extra, then return.
        await edit_message(
            query.message,
            f"⏳ Running operation `{op}` :: lang=`{lang_code}` …",
            buttons=None,
        )
        await query.answer("Started")
        ok, msg, outputs = await _process_op(session, op, extra=lang_code)
        if not ok:
            await edit_message(query.message, f"❌ {msg}")
            return
        listing = "\n".join(f"• `{ospath.basename(o)}`" for o in outputs)
        await edit_message(
            query.message,
            f"✅ {msg}\nProduced {len(outputs)} file(s):\n{listing}",
        )
        await _handoff_to_listener(session, outputs)
        VT_SESSIONS.pop(session_id, None)
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
