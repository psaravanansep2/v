import plistlib
import sys

from PIL import Image

from v.desktop import draw_icon, find_app_browser, install_shortcut, launch_command


def test_macos_app_bundle(tmp_path):
    created = install_shortcut("darwin", tmp_path, "/opt/py/bin/python3")
    app = tmp_path / "Applications" / "v.app"
    assert created == [app]
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "v" and info["LSUIElement"] is True
    launcher = app / "Contents" / "MacOS" / "v"
    if sys.platform != "win32":
        assert launcher.stat().st_mode & 0o111  # executable
    assert '"/opt/py/bin/python3" "-m" "v" "app" "--background"' in launcher.read_text()
    assert Image.open(app / "Contents" / "Resources" / "v.icns").size[0] >= 256


def test_linux_menu_and_desktop_entries(tmp_path):
    (tmp_path / "Desktop").mkdir()
    created = install_shortcut("linux", tmp_path, "/usr/bin/python3")
    names = sorted(p.name for p in created)
    assert names == ["v.desktop", "v.desktop"]
    entry = (tmp_path / ".local/share/applications/v.desktop").read_text()
    assert "Exec=/usr/bin/python3 -m v app --background" in entry and "Terminal=false" in entry
    assert (tmp_path / ".v" / "v.png").exists()


def test_icon_is_square_and_transparent_outside():
    icon = draw_icon(128)
    assert icon.size == (128, 128) and icon.getpixel((0, 0))[3] == 0 and icon.getpixel((64, 64))[3] == 255


def test_launch_command():
    assert launch_command("/x/python")[1:] == ["-m", "v", "app", "--background"]


def test_no_browser_found_on_a_bare_system(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert find_app_browser("linux") is None
