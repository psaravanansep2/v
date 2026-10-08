"""Linux system pieces v can't install by itself, and the exact command for
this computer's distribution (Ubuntu and Debian use apt; Fedora and Nobara
dnf; Arch and Manjaro pacman; openSUSE zypper; package names differ)."""

from __future__ import annotations

import ctypes.util
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

MANAGERS = {  # how to install, and how each system names the packages
    "apt": ("sudo apt update && sudo apt install -y", {"portaudio": "libportaudio2", "espeak": "espeak-ng",
                                                       "wl-clipboard": "wl-clipboard", "xclip": "xclip"}),
    "dnf": ("sudo dnf install -y", {"portaudio": "portaudio", "espeak": "espeak-ng", "wl-clipboard": "wl-clipboard",
                                    "xclip": "xclip"}),
    "pacman": ("sudo pacman -S --needed", {"portaudio": "portaudio", "espeak": "espeak-ng",
                                           "wl-clipboard": "wl-clipboard", "xclip": "xclip"}),
    "zypper": ("sudo zypper install", {"portaudio": "libportaudio2", "espeak": "espeak-ng",
                                       "wl-clipboard": "wl-clipboard", "xclip": "xclip"}),
}
FAMILIES = {"debian": "apt", "ubuntu": "apt", "fedora": "dnf", "rhel": "dnf", "centos": "dnf", "nobara": "dnf",
            "arch": "pacman", "manjaro": "pacman", "endeavouros": "pacman", "suse": "zypper", "opensuse": "zypper"}


def os_release(path: Path = Path("/etc/os-release")) -> dict:
    info = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            info[key.strip()] = value.strip().strip('"')
    except OSError:
        pass
    return info


def package_manager(release: Optional[dict] = None, which=shutil.which) -> Optional[str]:
    release = os_release() if release is None else release
    ids = [release.get("ID", "")] + release.get("ID_LIKE", "").split()
    for name in ids:
        for family, manager in FAMILIES.items():
            if name.lower().startswith(family) and which(manager if manager != "apt" else "apt-get"):
                return manager
    return next((m for m in MANAGERS if which(m if m != "apt" else "apt-get")), None)


def missing(env=os.environ, find_library=ctypes.util.find_library, which=shutil.which) -> list:
    """[(what it's for, package key)] for system pieces this computer lacks."""
    wayland = env.get("XDG_SESSION_TYPE") == "wayland" or bool(env.get("WAYLAND_DISPLAY"))
    out = []
    if not find_library("portaudio"):
        out.append(("talking to v in the terminal", "portaudio"))
    if not (which("espeak-ng") or which("espeak") or find_library("espeak-ng")):
        out.append(("v's voice in the terminal", "espeak"))
    if wayland and not which("wl-copy"):
        out.append(("copy and paste", "wl-clipboard"))
    elif not wayland and not (which("xclip") or which("xsel")):
        out.append(("copy and paste", "xclip"))
    return out


def install_hint(needs: list, manager: Optional[str]) -> Optional[str]:
    if not needs:
        return None
    if manager is None:
        return "Install these with your system's package manager: " + ", ".join(key for _, key in needs)
    command, names = MANAGERS[manager]
    return f"{command} {' '.join(dict.fromkeys(names[key] for _, key in needs))}"


def linux_report(platform: str = sys.platform) -> list:
    """Lines to show the user, or [] when nothing's missing (or not on Linux)."""
    if not platform.startswith("linux"):
        return []
    needs = missing()
    if not needs:
        return []
    lines = [f"Optional, for {', '.join(dict.fromkeys(what for what, _ in needs))} (the v window and phones work without):"]
    lines.append(f"  {install_hint(needs, package_manager())}")
    return lines
