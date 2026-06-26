"""Static CSS/JS assets read from scripts/assets/ at build time.

The files under ``scripts/assets/`` hold raw CSS/JS only, with no embedded
``<style>``/``<script>`` tags. ``css(...)`` and ``js(...)`` wrap a blob in the
matching tag at the emit point, so tag ownership lives in one place instead of
being split between the files and the call sites.
"""

from pathlib import Path

from thrml_render.config import ASSET_BASE

_ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _read(name):
    return (_ASSETS / name).read_text(encoding="utf-8")


def css(blob):
    """Wrap raw CSS in a <style> block, resolving the ASSET_BASE/ placeholder in
    @font-face urls to the configured CDN host.

    Asserts the placeholder is fully resolved, so a malformed ASSET_BASE/ token
    fails loudly at build time instead of shipping a broken font url."""
    resolved = blob.replace("ASSET_BASE/", ASSET_BASE.rstrip("/") + "/")
    assert "ASSET_BASE/" not in resolved, "unresolved ASSET_BASE/ placeholder after substitution"
    return f"<style>\n{resolved}</style>\n"


def js(blob):
    """Wrap raw JS in a <script> block for emission into a page."""
    return f"<script>\n{blob}</script>\n"


# Shared notebook + docs theme (also reused by the hand-authored docs pages).
COLLAPSE_STYLE = _read("notebook.css")
# Docs-page layout on top of the shared theme.
DOC_CSS = _read("docs.css")
# Landing-page styles.
INDEX_CSS = _read("index.css")
# Notebook/docs page chrome behaviour (collapsible inputs, external-link target).
COLLAPSE_SCRIPT = _read("chrome.js")
# Landing-page behaviour (copy buttons + animated LED panel).
INDEX_SCRIPT = _read("index.js")
