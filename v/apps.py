"""Finding what people mean by "open X": an installed app, a well-known
website, a folder, or a file. Only exact, checkable matches count; anything
else is left to the model, so "open" never opens the wrong thing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

SITES = {
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "google": "https://www.google.com",
    "google maps": "https://maps.google.com",
    "maps": "https://maps.google.com",
    "google drive": "https://drive.google.com",
    "drive": "https://drive.google.com",
    "google docs": "https://docs.google.com",
    "google sheets": "https://sheets.google.com",
    "google calendar": "https://calendar.google.com",
    "calendar": "https://calendar.google.com",
    "google translate": "https://translate.google.com",
    "translate": "https://translate.google.com",
    "google news": "https://news.google.com",
    "news": "https://news.google.com",
    "netflix": "https://www.netflix.com",
    "amazon": "https://www.amazon.com",
    "ebay": "https://www.ebay.com",
    "wikipedia": "https://www.wikipedia.org",
    "reddit": "https://www.reddit.com",
    "github": "https://github.com",
    "linkedin": "https://www.linkedin.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "whatsapp": "https://web.whatsapp.com",
    "outlook": "https://outlook.live.com",
    "spotify": "https://open.spotify.com",
    "discord": "https://discord.com/app",
    "slack": "https://app.slack.com",
    "zoom": "https://zoom.us",
    "dropbox": "https://www.dropbox.com",
    "pinterest": "https://www.pinterest.com",
    "twitch": "https://www.twitch.tv",
}

SITE_NAMES = {"youtube": "YouTube", "github": "GitHub", "linkedin": "LinkedIn", "whatsapp": "WhatsApp", "ebay": "eBay",
              "x": "X", "gmail": "Gmail", "google maps": "Google Maps", "maps": "Google Maps", "google drive": "Google Drive",
              "drive": "Google Drive", "calendar": "Google Calendar", "translate": "Google Translate", "news": "Google News"}

FOLDERS = {
    "downloads": "Downloads",
    "documents": "Documents",
    "desktop": "Desktop",
    "pictures": "Pictures",
    "photos": "Pictures",
    "music": "Music",
    "videos": "Movies" if sys.platform == "darwin" else "Videos",
    "movies": "Movies" if sys.platform == "darwin" else "Videos",
    "home": "",
    "home folder": "",
}

# What people say -> what the app is called
ALIASES = {
    "vs code": "visual studio code",
    "vscode": "visual studio code",
    "code": "visual studio code",
    "chrome": "google chrome",
    "word": "microsoft word",
    "excel": "microsoft excel",
    "powerpoint": "microsoft powerpoint",
    "teams": "microsoft teams",
    "onenote": "microsoft onenote",
    "edge": "microsoft edge",
    "system preferences": "system settings",
}

# Windows apps that don't have a Start-menu shortcut file
WINDOWS_BUILTINS = {
    "calculator": "calc",
    "notepad": "notepad",
    "paint": "mspaint",
    "settings": "ms-settings:",
    "file explorer": "explorer",
    "explorer": "explorer",
    "files": "explorer",
    "task manager": "taskmgr",
    "command prompt": "cmd",
    "terminal": "wt",
    "control panel": "control",
    "snipping tool": "snippingtool",
}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9+]+", " ", name.lower()).strip()


def _default_roots(platform: str, home: Path) -> list[Path]:
    if platform == "darwin":
        return [Path("/Applications"), Path("/Applications/Utilities"), Path("/System/Applications"),
                Path("/System/Applications/Utilities"), home / "Applications"]
    if platform.startswith("win"):
        return [Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Microsoft/Windows/Start Menu/Programs",
                Path(os.environ.get("APPDATA", str(home / "AppData/Roaming"))) / "Microsoft/Windows/Start Menu/Programs"]
    return [Path("/usr/share/applications"), Path("/usr/local/share/applications"), home / ".local/share/applications",
            Path("/var/lib/flatpak/exports/share/applications"), home / ".local/share/flatpak/exports/share/applications",
            Path("/var/lib/snapd/desktop/applications")]


class AppFinder:
    def __init__(self, platform: str = sys.platform, home: Optional[Path] = None, roots: Optional[list[Path]] = None):
        self.platform = platform
        self.home = home or Path.home()
        self.roots = roots if roots is not None else _default_roots(platform, self.home)
        self._index: Optional[dict[str, str]] = None

    def _scan(self) -> dict[str, str]:
        if self._index is not None:
            return self._index
        index: dict[str, str] = {}
        for root in self.roots:
            if not root.is_dir():
                continue
            try:
                if self.platform == "darwin":
                    for p in root.glob("*.app"):
                        index.setdefault(norm(p.stem), str(p))
                elif self.platform.startswith("win"):
                    for p in root.rglob("*.lnk"):
                        if "uninstall" not in p.stem.lower():
                            index.setdefault(norm(p.stem), str(p))
                else:
                    for p in root.glob("*.desktop"):
                        name = _desktop_name(p)
                        if name:
                            index.setdefault(norm(name), str(p))
            except OSError:
                continue
        self._index = index
        return index

    def find(self, spoken: str) -> Optional[str]:
        """A launch target for the app, or None if it isn't installed."""
        name = norm(spoken)
        if not name:
            return None
        index = self._scan()
        # the name as said, then its full name ("word" is "Word" in new Office, "Microsoft Word" in old)
        for candidate in dict.fromkeys([name, ALIASES.get(name, name)]):
            if candidate in index:
                return index[candidate]
        for candidate in dict.fromkeys([ALIASES.get(name, name), name]):
            # "chrome" -> "google chrome": a whole-word match, shortest name wins
            matches = [k for k in index if re.search(rf"(^| ){re.escape(candidate)}( |$)", k)]
            if matches:
                return index[min(matches, key=len)]
        if self.platform.startswith("win") and name in WINDOWS_BUILTINS:
            return "cmd:" + WINDOWS_BUILTINS[name]
        return None  # only real apps: never start a windowless command-line program


def _desktop_name(path: Path) -> Optional[str]:
    """The Name= of a Linux .desktop file, unless it's hidden."""
    name = None
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("[") and line.strip() != "[Desktop Entry]":
                break
            if line.startswith("Name=") and name is None:
                name = line[5:].strip()
            if line.strip() in ("NoDisplay=true", "Hidden=true"):
                return None
    except OSError:
        return None
    return name


def launch(target: str) -> None:
    """Open an app target from AppFinder, a URL, or a path."""
    if target.startswith("cmd:"):
        subprocess.Popen(["cmd", "/c", "start", "", target[4:]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    if target.startswith("exec:"):
        subprocess.Popen([target[5:]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return
    if target.endswith(".desktop"):
        if shutil.which("gtk-launch"):
            subprocess.Popen(["gtk-launch", Path(target).stem], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            return
        for line in Path(target).read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Exec="):
                argv = [a for a in line[5:].split() if not a.startswith("%")]
                subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                return
    from .tools import open_with_system

    open_with_system(target)


_DOMAIN = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+(/\S*)?$")
_FILLER = re.compile(r"^(the|my|a|an)\s+|\s+(app|application|program|website|web site|site|page|folder|directory)$")


def resolve(spoken: str, finder: AppFinder, home: Optional[Path] = None) -> Optional[tuple[str, str, str]]:
    """What to open for "open <spoken>": (kind, target, label), or None if
    there's no exact match (the model then works out what was meant)."""
    home = home or Path.home()
    raw = spoken.strip().lower().replace(" dot ", ".")
    name = raw
    for _ in range(3):
        name = _FILLER.sub("", name).strip()
    if not name or len(name.split()) > 5:
        return None
    if name in FOLDERS:
        path = home / FOLDERS[name]
        if path.is_dir():
            return ("folder", str(path), f"your {name.removesuffix(' folder')} folder" if name != "home" else "your home folder")
    if _DOMAIN.match(name) and " " not in name:
        return ("site", "https://" + name, name)
    app = finder.find(name)
    if app:
        label = Path(app).stem if not app.startswith(("cmd:", "exec:")) else name.title()
        if app.endswith(".desktop"):
            label = _desktop_name(Path(app)) or name.title()
        return ("app", app, label)
    if name in SITES:
        return ("site", SITES[name], SITE_NAMES.get(name, name.title()))
    path = Path(os.path.expanduser(raw))
    if raw.startswith(("~", "/")) and path.exists():
        return ("file", str(path), path.name)
    return None
