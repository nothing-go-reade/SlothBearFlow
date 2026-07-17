from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Iterable, List


class UnsafeCommand(ValueError):
    pass


_SHELL_METACHARACTERS = {";", "&&", "||", "|", ">", ">>", "<", "`", "$("}
_DANGEROUS_EXECUTABLES = {"sudo", "su", "shutdown", "reboot", "mkfs", "mount", "umount"}


def validate_command(command: str, *, allowed_executables: Iterable[str]) -> List[str]:
    if any(ord(character) < 32 for character in command):
        raise UnsafeCommand("command contains control characters")
    if any(value in command for value in _SHELL_METACHARACTERS):
        raise UnsafeCommand("shell operators are not allowed")
    argv = shlex.split(command)
    if not argv:
        raise UnsafeCommand("command is empty")
    executable_path = _trusted_executable(argv[0], label="command executable")
    trusted_allowlist = {
        _trusted_executable(value, label="allowed executable")
        for value in allowed_executables
    }
    executable = executable_path.name
    if executable in _DANGEROUS_EXECUTABLES or executable_path not in trusted_allowlist:
        raise UnsafeCommand("executable is not allowed")
    if executable == "rm" and _rm_is_recursive(argv[1:]):
        raise UnsafeCommand("dangerous recursive deletion is not allowed")
    argv[0] = str(executable_path)
    return argv


def _trusted_executable(value: object, *, label: str) -> Path:
    raw_path = Path(str(value or ""))
    if not raw_path.is_absolute():
        raise UnsafeCommand(f"{label} must be an absolute path")
    try:
        resolved = raw_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UnsafeCommand(f"{label} does not exist") from exc
    if not resolved.is_file() or not os.access(str(resolved), os.X_OK):
        raise UnsafeCommand(f"{label} is not an executable file")
    return resolved


def _rm_is_recursive(arguments: Iterable[str]) -> bool:
    for argument in arguments:
        if argument == "--":
            break
        if argument == "--recursive" or argument == "--no-preserve-root":
            return True
        if argument.startswith("--") or not argument.startswith("-"):
            continue
        if any(flag in {"r", "R"} for flag in argument[1:]):
            return True
    return False


def validate_workspace_path(path: str, *, workspace_root: str) -> Path:
    root = Path(workspace_root).resolve()
    candidate = Path(path).resolve()
    if candidate != root and root not in candidate.parents:
        raise UnsafeCommand("path escapes the workspace")
    return candidate
