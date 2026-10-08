"""The exact command for a Linux computer's missing pieces, for its distribution."""

import pytest

from v import system

RELEASES = {
    "nobara": {"ID": "nobara", "ID_LIKE": "rhel centos fedora"},
    "ubuntu": {"ID": "ubuntu", "ID_LIKE": "debian"},
    "mint": {"ID": "linuxmint", "ID_LIKE": "ubuntu debian"},
    "arch": {"ID": "arch"},
    "opensuse": {"ID": "opensuse-tumbleweed", "ID_LIKE": "opensuse suse"},
}


@pytest.mark.parametrize("distro,tool,command", [
    ("nobara", "dnf", "sudo dnf install -y portaudio espeak-ng wl-clipboard"),
    ("ubuntu", "apt-get", "sudo apt update && sudo apt install -y libportaudio2 espeak-ng wl-clipboard"),
    ("mint", "apt-get", "sudo apt update && sudo apt install -y libportaudio2 espeak-ng wl-clipboard"),
    ("arch", "pacman", "sudo pacman -S --needed portaudio espeak-ng wl-clipboard"),
    ("opensuse", "zypper", "sudo zypper install libportaudio2 espeak-ng wl-clipboard"),
])
def test_right_command_for_each_distribution(distro, tool, command):
    manager = system.package_manager(RELEASES[distro], which=lambda name: f"/usr/bin/{name}" if name == tool else None)
    needs = system.missing({"XDG_SESSION_TYPE": "wayland"}, find_library=lambda name: None, which=lambda name: None)
    assert system.install_hint(needs, manager) == command


def test_x11_needs_xclip_and_nothing_when_all_there():
    needs = system.missing({"XDG_SESSION_TYPE": "x11"}, find_library=lambda n: None, which=lambda n: None)
    assert [key for _, key in needs] == ["portaudio", "espeak", "xclip"]
    have = system.missing({"XDG_SESSION_TYPE": "wayland"}, find_library=lambda n: "lib", which=lambda n: "/usr/bin/x")
    assert have == [] and system.install_hint(have, "dnf") is None
    assert system.install_hint(needs, None).startswith("Install these with your system's package manager")


def test_reads_os_release(tmp_path):
    f = tmp_path / "os-release"
    f.write_text('NAME="Nobara Linux"\nID=nobara\nID_LIKE="rhel centos fedora"\n')
    assert system.os_release(f)["ID_LIKE"] == "rhel centos fedora"
    assert system.os_release(tmp_path / "missing") == {}
    assert system.linux_report(platform="darwin") == []
