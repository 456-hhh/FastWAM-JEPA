from __future__ import annotations

import logging
import os
import platform
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


SANDBOX_FAILURE_MARKERS = (
    "1223",
    "launch canceled",
    "orchestrator_helper_launch_canceled",
    "ShellExecuteExW failed",
)
CACHE_ENV_KEYS = (
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
)


def _as_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_wsl_path(value: str | os.PathLike[str] | None) -> bool:
    if value is None:
        return False
    text = str(value).replace("\\", "/").lower()
    return any(
        marker in text
        for marker in (
            "//wsl$",
            "//wsl.localhost/",
            "/mnt/wsl/",
            "ext4.vhdx",
            "\\wsl$",
        )
    )


def _running_inside_wsl() -> bool:
    if platform.system().lower() != "linux":
        return False
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        version = ""
    return "microsoft" in version or "wsl" in version


def _sandbox_failure_env_seen() -> bool:
    env_text = " ".join(
        str(os.environ.get(key, ""))
        for key in (
            "CODEX_SANDBOX_ERROR",
            "CODEX_SANDBOX_FAILURE",
            "CODEX_LAST_ERROR",
            "FASTWAM_JEPA_SANDBOX_ERROR",
        )
    ).lower()
    return any(marker.lower() in env_text for marker in SANDBOX_FAILURE_MARKERS)


def configure_runtime_stability(
    *,
    disable_wsl_fallback: bool = True,
    log_level: str = "INFO",
    log_path: str | None = None,
    max_log_mb: int = 100,
) -> dict[str, Any]:
    """Configure low-noise runtime defaults for FastWAM-JEPA scripts.

    This does not change Codex's internal sandbox implementation. It only keeps
    project scripts from enabling verbose Python logging or writing cache/logs to
    obvious WSL/ext4 fallback paths.
    """

    normalized_level = str(log_level or "INFO").upper()
    if normalized_level == "TRACE":
        normalized_level = "INFO"
    level = getattr(logging, normalized_level, logging.INFO)
    if max_log_mb <= 0:
        raise ValueError(f"`max_log_mb` must be positive, got {max_log_mb}.")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root_logger.addHandler(stream_handler)

    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=int(max_log_mb) * 1024 * 1024,
            backupCount=2,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        root_logger.addHandler(file_handler)

    os.environ.setdefault("FASTWAM_DEBUG", "0")
    os.environ.setdefault("FASTWAM_TRACE", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HYDRA_FULL_ERROR", "0")

    sandbox_failure_seen = _sandbox_failure_env_seen()
    safe_mode = bool(disable_wsl_fallback or sandbox_failure_seen)
    if safe_mode:
        os.environ["FASTWAM_JEPA_SAFE_MODE"] = "1"
        os.environ["FASTWAM_JEPA_DISABLE_WSL_FALLBACK"] = "1"
        os.environ["DISABLE_WSL_FALLBACK"] = "1"

    if safe_mode and _running_inside_wsl():
        raise RuntimeError(
            "FastWAM-JEPA safe mode is active and this process appears to be running inside WSL. "
            "Use native Windows Python for local Codex work, or run on the Linux training server."
        )

    checked_paths: dict[str, str] = {}
    if safe_mode:
        for key in CACHE_ENV_KEYS:
            value = os.environ.get(key)
            if not value:
                continue
            checked_paths[key] = value
            if _looks_like_wsl_path(value):
                raise RuntimeError(
                    f"FastWAM-JEPA safe mode blocks {key}={value!r} because it looks like a WSL/ext4 cache path."
                )
        cwd = Path.cwd()
        if _looks_like_wsl_path(cwd):
            raise RuntimeError(
                f"FastWAM-JEPA safe mode blocks running from WSL/ext4-looking cwd: {cwd}"
            )

    return {
        "safe_mode": safe_mode,
        "disable_wsl_fallback": bool(disable_wsl_fallback),
        "sandbox_failure_seen": sandbox_failure_seen,
        "log_level": logging.getLevelName(level),
        "log_rotation_max_mb": int(max_log_mb) if log_path else 0,
        "disk_log_enabled": bool(log_path),
        "log_path": str(Path(log_path).resolve()) if log_path else None,
        "checked_cache_env": checked_paths,
    }
