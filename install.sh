#!/bin/sh
# Installs v, the free voice assistant, on macOS or Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/psaravanansep2/v/main/install.sh | sh
#
# What it does, all inside your home folder (no admin password):
#   1. installs uv, a small tool that brings its own Python
#   2. installs v with it
#   3. downloads the free AI model that fits this computer (one time, a few GB)
#   4. adds a v icon to your apps, and opens v
#
# Options (environment variables): V_KOKORO=1 adds the most natural voice
# (a larger download); V_SKIP_MODEL=1 skips step 3 (v does it on first launch).
set -eu

REPO="${V_REPO:-https://github.com/psaravanansep2/v}"
REF="${V_REF:-main}"
SOURCE="${V_SOURCE:-$REPO/archive/refs/heads/$REF.zip}"
EXTRAS="${V_EXTRAS:-all}"
PYTHON_VERSION="${V_PYTHON:-3.12}"
[ "${V_KOKORO:-0}" = "1" ] && EXTRAS="$EXTRAS,kokoro"

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

case "$(uname -s)" in
  Darwin|Linux) ;;
  *) fail "This installer is for macOS and Linux. On Windows, use install.ps1 (see the README)." ;;
esac

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

bold "1/4  Getting uv (brings its own Python)…"
if ! command -v uv >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || true
  fi
  if ! command -v uv >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
    python3 -m pip install --user --quiet uv || true
  fi
  command -v uv >/dev/null 2>&1 || fail "Couldn't install uv. Check the internet connection and try again."
fi

bold "2/4  Installing v…"
uv tool install --force --python "$PYTHON_VERSION" "v[$EXTRAS] @ $SOURCE" \
  || fail "Installing v failed (see above). Check the internet connection and try again."
command -v v >/dev/null 2>&1 || fail "v was installed but isn't on your PATH. Open a new terminal and run: v app"

if [ "${V_SKIP_MODEL:-0}" != "1" ]; then
  bold "3/4  Getting the free AI model ready (one time; this can take a while)…"
  v setup || printf '\nThe model download didn'"'"'t finish; v will try again when it starts.\n'
fi

bold "4/4  Adding the v icon…"
v install-shortcut || true

bold "Done! v is ready."
echo "Open it any time from its icon, or run: v app"
echo "Use it from your phone: click the phone button in v's window and scan the code."
if [ "${V_NO_LAUNCH:-0}" != "1" ]; then
  (nohup v app --background >/dev/null 2>&1 &) || true
fi
