import zipfile
from types import SimpleNamespace

import pytest
from PIL import Image

from v.config import Config, load_goal
from v.confirm import Confirmer
from v.tools import Screen, Toolbox, html_to_text, parse_arguments, parse_search_results, read_image


class RecordingUI:
    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        return lambda *args: self.events.append((name, *args))


def toolbox(tmp_path, answers=(), confirm="risky", screen=None, fetch=None):
    answers = list(answers)
    questions = []

    def ask(q):
        questions.append(q)
        return answers.pop(0)

    cfg = Config(project_dir=tmp_path / "project", confirm=confirm)
    cfg.project_dir.mkdir(exist_ok=True)
    opened = []
    tb = Toolbox(cfg, Confirmer(confirm, ask), RecordingUI(), screen, opener=opened.append, fetch=fetch)
    tb.questions, tb.opened = questions, opened
    return tb


def test_files_inside_the_project_need_no_confirmation(tmp_path):
    tb = toolbox(tmp_path)
    assert "Wrote" in tb.run("write_file", {"path": "notes/todo.txt", "content": "milk\neggs"})
    assert tb.run("read_file", {"path": "notes/todo.txt"}) == "milk\neggs"
    assert "Edited" in tb.run("edit_file", {"path": "notes/todo.txt", "old_text": "eggs", "new_text": "bread"})
    assert (tmp_path / "project/notes/todo.txt").read_text() == "milk\nbread"
    assert tb.questions == []


def test_writing_outside_the_project_asks_first(tmp_path):
    tb = toolbox(tmp_path, answers=["no", "yes"])
    target = tmp_path / "elsewhere.txt"
    assert tb.run("write_file", {"path": str(target), "content": "x"}) == "The user declined this change."
    assert not target.exists()
    assert "Wrote" in tb.run("write_file", {"path": str(target), "content": "x"})
    assert target.read_text() == "x" and len(tb.questions) == 2


def test_edit_needs_a_unique_match(tmp_path):
    tb = toolbox(tmp_path)
    tb.run("write_file", {"path": "a.txt", "content": "x x"})
    assert "appears 2 times" in tb.run("edit_file", {"path": "a.txt", "old_text": "x", "new_text": "y"})
    assert "wasn't found" in tb.run("edit_file", {"path": "a.txt", "old_text": "z", "new_text": "y"})


def test_list_and_find_files(tmp_path):
    tb = toolbox(tmp_path)
    for name in ("report.pdf", "sub/old-report.pdf", "photo.jpg", ".hidden/secret.pdf"):
        p = tmp_path / "project" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"%PDF")
    listing = tb.run("list_files", {})
    assert "sub/" in listing and "photo.jpg" in listing and ".hidden" not in listing
    found = tb.run("list_files", {"pattern": "*.pdf"})
    assert "report.pdf" in found and "sub/old-report.pdf" in found and "secret" not in found


def test_read_word_document(tmp_path):
    tb = toolbox(tmp_path)
    doc = tmp_path / "project" / "letter.docx"
    with zipfile.ZipFile(doc, "w") as z:
        z.writestr(
            "word/document.xml",
            '<w:document><w:body><w:p><w:r><w:t>Dear Sam,</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>Thanks &amp; see you soon.</w:t></w:r></w:p></w:body></w:document>',
        )
    assert tb.run("read_file", {"path": "letter.docx"}) == "Dear Sam,\nThanks & see you soon."


def test_binary_files_are_not_dumped(tmp_path):
    tb = toolbox(tmp_path)
    (tmp_path / "project" / "x.bin").write_bytes(b"\x00\x01" * 100)
    assert "isn't a text file" in tb.run("read_file", {"path": "x.bin"})


def test_commands_ask_and_run(tmp_path):
    tb = toolbox(tmp_path, answers=["yes"])
    out = tb.run("run_command", {"command": "echo from-v"})
    assert "from-v" in out and tb.questions == ["Run this command: echo from-v?"]


def test_open_and_goal(tmp_path):
    tb = toolbox(tmp_path)
    assert tb.run("open", {"target": "https://example.com"}) == "Opened https://example.com."
    assert tb.opened == ["https://example.com"]
    (tmp_path / "project" / "docs").mkdir()
    (tmp_path / "project" / "docs" / "report.pdf").write_bytes(b"%PDF")
    tb.run("open", {"target": "docs/report.pdf"})
    assert tb.opened[-1] == str((tmp_path / "project" / "docs" / "report.pdf").resolve())  # not relative to where v started
    tb.run("open", {"target": "Spotify"})
    assert tb.opened[-1] == "Spotify"  # an app name stays a name
    tb.run("set_project_goal", {"goal": "Tidy my downloads"})
    assert load_goal(tb.cfg.project_dir) == "Tidy my downloads"


DDG_PAGE = """
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title"><a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2Fdownloads%2F&amp;rut=abc">Download <b>Python</b></a></h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">The official home of the <b>Python</b> language.</a>
</div>
<div class="result">
  <h2 class="result__title"><a class="result__a" href="https://docs.python.org/3/">Python docs</a></h2>
  <div class="result__snippet">Documentation for Python 3.</div>
</div>
"""


def test_web_search_results(tmp_path):
    seen = []

    def fetch(url):
        seen.append(url)
        return DDG_PAGE, "text/html"

    tb = toolbox(tmp_path, fetch=fetch)
    out = tb.run("web_search", {"query": "python download"})
    assert seen == ["https://html.duckduckgo.com/html/?q=python+download"]
    assert out.startswith("1. Download Python\nhttps://www.python.org/downloads/\nThe official home of the Python language.")
    assert "2. Python docs\nhttps://docs.python.org/3/" in out


def test_search_parser_ignores_junk():
    assert parse_search_results("<html><body>nothing</body></html>") == []


def test_read_webpage_strips_markup(tmp_path):
    page = "<html><head><title>Recipe</title><script>var x=1</script></head><body><nav>Menu</nav><h1>Pancakes</h1><p>Mix flour and milk.</p></body></html>"
    tb = toolbox(tmp_path, fetch=lambda url: (page, "text/html; charset=utf-8"))
    assert tb.run("read_webpage", {"url": "example.com/pancakes"}) == "Recipe\n\nPancakes\nMix flour and milk."
    assert html_to_text("<p>a</p><p>b</p>") == ("", "a\nb")


def test_errors_come_back_as_text(tmp_path):
    tb = toolbox(tmp_path)
    assert tb.run("read_file", {"path": "missing.txt"}).startswith("Error: FileNotFoundError")
    assert tb.run("read_file", {"nope": 1}).startswith("Error: wrong arguments")
    assert tb.run("teleport", {}).startswith("Error: there is no tool named teleport. Available: clipboard, edit_file")
    assert tb.run("look_at_screen", {}).startswith("Error: there is no tool named look_at_screen.")  # screen is off


def test_long_results_are_clipped(tmp_path):
    tb = toolbox(tmp_path)
    (tmp_path / "project" / "big.txt").write_text("x" * 50_000)
    assert len(tb.run("read_file", {"path": "big.txt"})) < 9_000


def test_parse_arguments():
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments({"a": 1}) == {"a": 1}
    assert parse_arguments("") == {}
    with pytest.raises(ValueError):
        parse_arguments("[1, 2]")


# --- screen ---


class FakeOCR:
    def __call__(self, image_array):
        h = image_array.shape[0]
        scale = h / 1080  # boxes come back in captured (physical) pixels
        box = lambda x, y: [[(x - 20) * scale, (y - 10) * scale], [(x + 20) * scale, (y - 10) * scale], [(x + 20) * scale, (y + 10) * scale], [(x - 20) * scale, (y + 10) * scale]]  # noqa: E731
        return [
            (box(900, 600), "Submit", 0.95),
            (box(100, 50), "File", 0.9),
            (box(200, 52), "Edit", 0.9),
            (box(500, 300), "~~", 0.2),  # low confidence: dropped
        ], 0.1


class FakeGui:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a, k))


def screen_toolbox(tmp_path, physical=(1920, 1080), confirm="risky", answers=()):
    gui = FakeGui()
    computer = SimpleNamespace(gui=gui, width=1920, height=1080, platform="linux",
                               grab=lambda: Image.new("RGB", physical, "white"), _type=lambda text: gui.calls.append(("type", (text,), {})))
    tb = toolbox(tmp_path, answers=answers, confirm=confirm, screen=Screen(computer, ocr=FakeOCR()))
    return tb, gui


def test_look_lists_text_in_reading_order(tmp_path):
    tb, _ = screen_toolbox(tmp_path)
    out = tb.run("look_at_screen", {})
    assert out.splitlines()[1:] == ["1: File @ 100,50", "2: Edit @ 200,52", "3: Submit @ 900,600"]
    assert any(e[0] == "image" for e in tb.ui.events)  # preview goes to the phone


def test_click_by_number_on_a_retina_screen(tmp_path):
    tb, gui = screen_toolbox(tmp_path, physical=(3840, 2160))  # 2x capture, 1920x1080 points
    tb.run("look_at_screen", {})
    assert tb.run("click", {"element": 3}).startswith("Clicked at 900,600")
    assert gui.calls[-1] == ("click", (), {"x": 900, "y": 600, "clicks": 1, "interval": 0.05, "button": "left"})
    assert "no element 9" in tb.run("click", {"element": 9})


def test_screen_actions_ask_under_confirm_all(tmp_path):
    tb, gui = screen_toolbox(tmp_path, confirm="all", answers=["no", "yes"])
    tb.run("look_at_screen", {})
    assert tb.run("click", {"element": 3}) == "The user declined this action."
    assert tb.questions == ["I'm about to click “Submit”. Okay?"]
    assert tb.run("press_keys", {"keys": "ctrl+s"}) == "Pressed ctrl+s."
    assert gui.calls[-1] == ("hotkey", ("ctrl", "s"), {})


def test_forgives_common_tool_mistakes(tmp_path):
    tb = toolbox(tmp_path)
    (tmp_path / "project" / "a.txt").write_text("hello")
    assert tb.run("open_app", {"app": "Spotify"}) == "Opened Spotify."  # wrong tool and argument names
    assert tb.opened == ["Spotify"]
    assert tb.run("cat", {"file_path": "a.txt", "reason": "user asked"}) == "hello"  # extra argument dropped
    assert "Wrote" in tb.run("create_file", {"filename": "b.txt", "text": "x"})
    assert tb.run("edit_file", {"file": "b.txt", "old_string": "x", "new_string": "y"}).startswith("Edited")
    assert (tmp_path / "project" / "b.txt").read_text() == "y"


def test_click_with_numbers_as_text_and_implied_arguments(tmp_path):
    tb, gui = screen_toolbox(tmp_path)
    tb.run("look_at_screen", {})
    assert tb.run("double_click", {"element_id": "3"}).startswith("Clicked at 900,600")
    assert gui.calls[-1][2]["clicks"] == 2


def test_files_keep_their_line_endings_and_unicode(tmp_path):
    tb = toolbox(tmp_path)
    crlf = tmp_path / "project" / "windows.txt"
    crlf.write_bytes("first line\r\nsecond line\r\ncafé\r\n".encode("utf-8"))
    assert tb.run("edit_file", {"path": "windows.txt", "old_text": "first line\nsecond", "new_text": "1st line\n2nd"}).startswith("Edited")
    assert crlf.read_bytes() == "1st line\r\n2nd line\r\ncafé\r\n".encode("utf-8")  # still CRLF, nothing else touched
    tb.run("write_file", {"path": "notes.txt", "content": "Zürich\nnaïve"})
    assert (tmp_path / "project" / "notes.txt").read_bytes() == "Zürich\nnaïve".encode("utf-8")


class WordsOCR:
    """Fake OCR: words at given boxes, out of order like real OCR output can be."""

    def __call__(self, image):
        def box(x, y, w=80, h=20):
            return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

        return [
            (box(200, 52), "$4.50", 0.95),
            (box(10, 10), "Corner Cafe", 0.99),
            (box(10, 50), "Latte", 0.9),
            (box(10, 90, h=24), "Total", 0.9),
            (box(200, 93), "$4.50", 0.9),
            (box(300, 300), "smudge", 0.2),  # low confidence: dropped
        ], 0.1


def test_reads_the_words_in_a_picture(tmp_path):
    Image.new("RGB", (400, 200), "white").save(tmp_path / "receipt.jpg")
    out = read_image(tmp_path / "receipt.jpg", WordsOCR())
    assert out.splitlines()[1:] == ["Corner Cafe", "Latte  $4.50", "Total  $4.50"]
    assert out.startswith("receipt.jpg is a 400x200 picture.")
    tb = toolbox(tmp_path, screen=SimpleNamespace(_ocr=WordsOCR()))
    assert "Latte  $4.50" in tb.run("read_file", {"path": str(tmp_path / "receipt.jpg")})


def test_reads_iphone_photos(tmp_path):
    heif = pytest.importorskip("pillow_heif")
    heif.register_heif_opener()
    Image.new("RGB", (400, 200), "white").save(tmp_path / "IMG_0042.HEIC")
    out = read_image(tmp_path / "IMG_0042.HEIC", WordsOCR())
    assert out.startswith("IMG_0042.HEIC is a 400x200 picture.") and "Latte  $4.50" in out


def test_reads_a_real_photo_of_text(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    from PIL import ImageDraw

    from v.check import sample_font

    image = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 40), "Meeting moved", fill="black", font=sample_font(56))
    draw.text((30, 160), "Thursday 3pm", fill="black", font=sample_font(56))
    image.rotate(90, expand=True).save(tmp_path / "note.jpg", exif=_rotated_exif())  # stored sideways, like phone photos
    lines = read_image(tmp_path / "note.jpg").lower().splitlines()[1:]
    assert len(lines) == 2 and "meeting" in lines[0] and "thursday" in lines[1]


def _rotated_exif():
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90° clockwise to display
    return exif


def test_web_pages_are_checked_when_written(tmp_path):
    tb = toolbox(tmp_path)
    out = tb.run("write_file", {"path": "site/style.css",
                                "content": "@keyframes slideDown {from{top:-9px}to{top:0}}\nh1 { animation: 2s colorShift infinite; }\n"})
    assert "an animation uses colorShift, but there's no @keyframes colorShift" in out
    assert "@keyframes slideDown isn't used by any animation" in out
    out = tb.run("write_file", {"path": "site/index.html", "content":
                                '<link href="styles.css" rel="stylesheet"><img src="images/a.jpg"><a href="#top">t</a>'
                                '<script src="https://cdn.example/x.js"></script><a href="mailto:x@y.z">m</a>'})
    assert "it links to styles.css, which doesn't exist (there is style.css)" in out
    assert "images/a.jpg, which doesn't exist" in out and "cdn.example" not in out and "#top" not in out
    out = tb.run("edit_file", {"path": "site/index.html", "old_text": "styles.css", "new_text": "style.css"})
    assert "styles.css" not in out and "images/a.jpg" in out
    fine = tb.run("write_file", {"path": "site/ok.css", "content": "p { animation: fade 1s; } @keyframes fade {to{opacity:1}}"})
    assert "Problems" not in fine
    assert "Problems" not in tb.run("write_file", {"path": "notes.txt", "content": "styles.css"})  # not a web page


def _png(w=2400, h=1600):
    import io

    buf = io.BytesIO()
    Image.new("RGB", (w, h), "red").save(buf, format="PNG")
    return buf.getvalue()


def test_free_pictures_for_a_page(tmp_path):
    import json as _json

    asked = []

    def fetch(url):
        asked.append(url)
        if "openverse" in url:
            return _json.dumps({"results": [
                {"url": "https://img.example/1.png", "title": "Margherita", "creator": "Ana", "license": "by",
                 "license_version": "2.0", "foreign_landing_url": "https://flickr.example/1"},
                {"url": "https://img.example/broken", "title": "x", "creator": "y", "license": "by"},
                {"url": "https://img.example/2.png", "title": "Oven", "creator": "Ben", "license": "cc0",
                 "license_version": "1.0", "foreign_landing_url": "https://flickr.example/2"},
            ]}), "application/json"
        raise OSError("unexpected")

    def download(url):
        if "broken" in url:
            raise OSError("404")
        return _png(), "image/png"

    tb = toolbox(tmp_path, fetch=fetch)
    tb.download = download
    out = tb.run("find_images", {"q": "pizza", "path": "site/images", "number": 2})  # forgiving names
    assert out.startswith("Saved 2 pictures in site/images: pizza-1.jpg, pizza-2.jpg.")
    assert "“Margherita” by Ana, CC BY 2.0 (https://flickr.example/1)" in out
    saved = tmp_path / "project" / "site" / "images"
    with Image.open(saved / "pizza-1.jpg") as image:
        assert max(image.size) == 1600  # resized for the web
    assert "Oven" in (saved / "CREDITS.txt").read_text()
    assert "q=pizza" in asked[0]


def test_pictures_fall_back_to_wikimedia(tmp_path):
    import json as _json

    def fetch(url):
        if "openverse" in url:
            raise OSError("down")
        return _json.dumps({"query": {"pages": {"1": {"title": "File:Pizza.jpg", "imageinfo": [{
            "thumburl": "https://upload.example/pizza.jpg", "descriptionurl": "https://commons.example/Pizza",
            "extmetadata": {"Artist": {"value": "<a href='x'>Cara</a>"}, "LicenseShortName": {"value": "CC BY-SA 4.0"}}}]}}}}), ""

    tb = toolbox(tmp_path, fetch=fetch)
    tb.download = lambda url: (_png(800, 600), "image/jpeg")
    out = tb.run("get_images", {"query": "pizza", "count": 1})
    assert "Saved 1 pictures in images: pizza-1.jpg" in out and "“Pizza.jpg” by Cara, CC BY-SA 4.0" in out
    tb.download = lambda url: (_ for _ in ()).throw(OSError("offline"))
    assert "Couldn't find or download pictures" in tb.run("get_images", {"query": "pasta"})
