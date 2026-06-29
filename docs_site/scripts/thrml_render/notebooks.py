"""Notebook export: hide-input tagging, image externalization, titles."""

import base64
import hashlib
import re

from nbconvert import HTMLExporter

from .config import FIG_DIR


def tag_hidden_inputs(nb):
    """Mirror each cell's JupyterLab source_hidden state onto a hide-input tag.

    Mutates ``nb`` in place; the source ``.ipynb`` on disk is never touched.
    """
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        hidden = cell.get("metadata", {}).get("jupyter", {}).get("source_hidden", False)
        if not hidden:
            continue
        tags = list(cell.get("metadata", {}).get("tags", []))
        if "hide-input" not in tags:
            tags.append("hide-input")
        cell.metadata["tags"] = tags


def build_exporter():
    return HTMLExporter(template_name="lab", embed_images=True)


_IMG_TAG_RE = re.compile(r"<img\b[^>]*?>", re.IGNORECASE)
_SRC_DATA_RE = re.compile(r'src="data:image/png;base64,([^"]+)"')


def _png_size(raw):
    """Return (width, height) from a PNG's IHDR, or None if not a PNG."""
    if len(raw) >= 24 and raw[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    return None


def externalize_images(body, stem):
    """Write inline base64 figure PNGs out to rendered/figures/ and rewrite the
    <img> tags to reference them, with lazy/async loading and intrinsic height.

    nbconvert's lab template inlines every output figure as a base64 data URI,
    which makes a notebook page 600KB-1.9MB (~79% image payload) that the browser
    must download and parse before first paint. Externalizing the figures shrinks
    the HTML ~5x, lets the browser cache/parallel-load images, and makes the
    figures behave like PennyLane's separately hosted plots. The lab-template
    favicon lives in CSS as url(data:...) (not an <img>), so it is left untouched.
    """
    FIG_DIR.mkdir(exist_ok=True)
    counter = {"i": 0}

    def repl(match):
        tag = match.group(0)
        data = _SRC_DATA_RE.search(tag)
        if not data:
            return tag
        raw = base64.b64decode(data.group(1))
        digest = hashlib.sha1(raw).hexdigest()[:12]
        idx = counter["i"]
        counter["i"] += 1
        fname = f"{stem}_{idx}_{digest}.png"
        (FIG_DIR / fname).write_bytes(raw)
        tag = _SRC_DATA_RE.sub(f'src="assets/{fname}"', tag, count=1)

        extra = ""
        if "loading=" not in tag:
            extra += ' loading="lazy" decoding="async"'
        size = _png_size(raw)
        width_attr = re.search(r'width="(\d+)"', tag)
        if size and width_attr and "height=" not in tag:
            disp_w = int(width_attr.group(1))
            png_w, png_h = size
            if png_w:
                extra += f' height="{round(disp_w * png_h / png_w)}"'
        if extra:
            if tag.endswith("/>"):
                tag = tag[:-2] + extra + "/>"
            elif tag.endswith(">"):
                tag = tag[:-1] + extra + ">"
        return tag

    return _IMG_TAG_RE.sub(repl, body)


def notebook_title(nb, fallback):
    for cell in nb.cells:
        if cell.get("cell_type") == "markdown":
            match = re.search(r"^#\s+(.+)", "".join(cell.get("source", "")), re.MULTILINE)
            if match:
                return match.group(1).strip()
    return fallback
