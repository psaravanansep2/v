"""v as a desktop app: an app window, and an icon to start it.

The window is v's own web page served from this computer (the same page
phones use), opened as an app window in Chrome or Edge when one is
installed, else in the default browser. `v install-shortcut` puts a v icon
in Applications (macOS), the Start menu and Desktop (Windows), or the app
menu (Linux), which runs `v app --background`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional

STATE_DIR = Path.home() / ".v"


def find_app_browser(platform: str = sys.platform) -> Optional[str]:
    """A Chromium-based browser that can open a page as an app window."""
    if platform == "darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ]
    elif platform.startswith("win"):
        env = os.environ
        candidates = [
            os.path.join(env.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), r"Microsoft\Edge\Application\msedge.exe"),
            os.path.join(env.get("ProgramFiles", r"C:\Program Files"), r"Microsoft\Edge\Application\msedge.exe"),
            os.path.join(env.get("ProgramFiles", r"C:\Program Files"), r"Google\Chrome\Application\chrome.exe"),
            os.path.join(env.get("LOCALAPPDATA", ""), r"Google\Chrome\Application\chrome.exe"),
        ]
    else:
        candidates = [shutil.which(n) or "" for n in
                      ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "brave-browser")]
    return next((c for c in candidates if c and os.path.exists(c)), None)


def open_window(url: str) -> None:
    browser = find_app_browser()
    if browser is None:
        webbrowser.open(url)
        return
    profile = STATE_DIR / "window"  # its own profile: remembers the microphone permission, no browser clutter
    subprocess.Popen(
        [browser, f"--app={url}", "--window-size=460,860", f"--user-data-dir={profile}",
         "--no-first-run", "--no-default-browser-check"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )


def draw_icon(size: int = 512):
    """v's icon: an indigo V on a dark rounded square."""
    from PIL import Image, ImageDraw

    scale = 4  # draw big, then shrink, for smooth edges
    big = size * scale
    image = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, big - 1, big - 1), radius=int(big * 0.22), fill=(17, 24, 39, 255))
    width = int(big * 0.11)
    points = [(big * 0.285, big * 0.33), (big * 0.5, big * 0.72), (big * 0.715, big * 0.33)]
    draw.line(points, fill=(165, 180, 252, 255), width=width, joint="curve")
    for x, y in (points[0], points[2]):
        r = width / 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(165, 180, 252, 255))
    return image.resize((size, size), Image.LANCZOS)


def launch_command(python: Optional[str] = None) -> list[str]:
    python = python or sys.executable
    if sys.platform.startswith("win"):
        windowless = Path(python).with_name("pythonw.exe")
        if windowless.exists():
            python = str(windowless)  # no black console window
    return [python, "-m", "v", "app", "--background"]


def install_shortcut(platform: str = sys.platform, home: Optional[Path] = None, python: Optional[str] = None) -> list[Path]:
    """Create the v icon. Returns the files created."""
    home = home or Path.home()
    state = home / ".v"
    state.mkdir(parents=True, exist_ok=True)
    command = launch_command(python)
    created: list[Path] = []

    if platform == "darwin":
        app = home / "Applications" / "v.app" / "Contents"
        (app / "MacOS").mkdir(parents=True, exist_ok=True)
        (app / "Resources").mkdir(parents=True, exist_ok=True)
        (app / "Info.plist").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>v</string>
  <key>CFBundleDisplayName</key><string>v</string>
  <key>CFBundleIdentifier</key><string>io.github.psaravanansep2.v</string>
  <key>CFBundleExecutable</key><string>v</string>
  <key>CFBundleIconFile</key><string>v.icns</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>LSUIElement</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>v listens when you talk to it.</string>
</dict></plist>
""")
        launcher = app / "MacOS" / "v"
        launcher.write_text("#!/bin/bash\nexec " + " ".join(f'"{c}"' for c in command) + ' "$@"\n')
        launcher.chmod(0o755)
        draw_icon(512).save(app / "Resources" / "v.icns")
        created.append(app.parent)
        return created

    png = state / "v.png"
    draw_icon(256).save(png)

    if platform.startswith("win"):
        ico = state / "v.ico"
        draw_icon(256).save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
        targets = [
            Path(os.environ.get("APPDATA", home / "AppData/Roaming")) / "Microsoft/Windows/Start Menu/Programs/v.lnk",
            home / "Desktop" / "v.lnk",
        ]
        for link in targets:
            if not link.parent.is_dir():
                continue
            ps = (
                "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{link}'); "
                "$s.TargetPath = '{target}'; $s.Arguments = '{args}'; $s.IconLocation = '{ico}'; "
                "$s.WorkingDirectory = '{home}'; $s.Description = 'v, your free voice assistant'; $s.Save()"
            ).format(link=_ps(link), target=_ps(command[0]), args=_ps(" ".join(command[1:])), ico=_ps(ico), home=_ps(home))
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps], check=True,
                           capture_output=True)
            created.append(link)
        return created

    entry = (
        "[Desktop Entry]\nType=Application\nName=v\nComment=Your free voice assistant\n"
        f"Exec={' '.join(_desktop_quote(c) for c in command)}\nIcon={png}\nTerminal=false\nCategories=Utility;\n"
    )
    menu = home / ".local/share/applications/v.desktop"
    menu.parent.mkdir(parents=True, exist_ok=True)
    menu.write_text(entry)
    created.append(menu)
    desktop = home / "Desktop"
    if desktop.is_dir():
        on_desktop = desktop / "v.desktop"
        on_desktop.write_text(entry)
        on_desktop.chmod(0o755)
        if shutil.which("gio"):
            subprocess.run(["gio", "set", str(on_desktop), "metadata::trusted", "true"], capture_output=True)
        created.append(on_desktop)
    return created


def _ps(value) -> str:
    return str(value).replace("'", "''")


def _desktop_quote(arg: str) -> str:
    return f'"{arg}"' if " " in arg else arg
