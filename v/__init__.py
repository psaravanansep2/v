__version__ = "0.1.0"

_build = None


def build_id() -> str:
    """A fingerprint of the code that's installed, so a running v can tell it's been updated
    (the version number alone doesn't change between updates)."""
    global _build
    if _build is None:
        import hashlib
        from pathlib import Path

        here = Path(__file__).resolve().parent
        digest = hashlib.sha1()
        for f in sorted(here.rglob("*")):
            if f.suffix in (".py", ".html") and "__pycache__" not in f.parts:
                digest.update(f.relative_to(here).as_posix().encode())
                digest.update(f.read_bytes())
        _build = digest.hexdigest()[:12]
    return _build
