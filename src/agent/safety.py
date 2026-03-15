from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyConfig:
    max_steps: int = 50
    default_timeout_s: int = 600
    max_log_chars: int = 4000


def check_shell_safe(cmd: str) -> tuple[bool, str]:
    text = cmd.strip().lower()
    if not text:
        return False, "empty command"

    banned_substrings = [
        "sudo ",
        " rm ",
        "\nrm ",
        ";rm ",
        " rm\n",
        "git reset --hard",
        "mkfs",
        "shutdown",
        "reboot",
    ]
    for needle in banned_substrings:
        if needle in f" {text} ":
            return False, f"blocked by safety rule: '{needle.strip()}'"

    # Check for dd as a standalone command (not inside words like "add", "odd")
    import re as _re
    if _re.search(r"(?:^|[;\|\n&])\s*dd\s", text):
        return False, "blocked by safety rule: 'dd'"

    return True, ""

