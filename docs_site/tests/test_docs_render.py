"""End-to-end checks for the docs site renderer (``docs_site/scripts``).

Runs the real renderer once per session and asserts on the emitted HTML, so
template drift, API-surface drift, or a regression in the per-class member
rendering fails here instead of on deploy. Lives under ``docs_site/tests`` (not
the top-level ``tests/``) and ``importorskip``s the docs extras, so it runs only
in the docs-build job where ``nbconvert``/``nbformat`` are installed.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("nbconvert")
pytest.importorskip("nbformat")

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "docs_site" / "scripts"
RENDERED = REPO / "docs_site" / "rendered"
NOTEBOOKS = REPO / "examples"


def _load(module_name):
    """Import a renderer module out of ``docs_site/scripts`` at runtime."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module(module_name)


@pytest.fixture(scope="session")
def site():
    """Run the real site build once and return the rendered output dir."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "render_html.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"render_html.py exited {result.returncode}\n" f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return RENDERED


def test_core_pages_exist_and_nonempty(site):
    expected = ["index.html", "getting-started.html", "concepts.html", "examples.html", "llms.txt"]
    expected += [p.name for p in sorted(site.glob("api-*.html"))]
    expected += [f"{p.stem}.html" for p in sorted(NOTEBOOKS.glob("[0-9]*.ipynb"))]
    assert len(list(site.glob("api-*.html"))) >= 1
    assert len(list(NOTEBOOKS.glob("[0-9]*.ipynb"))) >= 1
    for name in expected:
        page = site / name
        assert page.exists(), f"missing rendered page: {name}"
        assert page.stat().st_size > 0, f"empty rendered page: {name}"


def test_notebook_title_marker_replaced(site):
    for page in site.glob("*.html"):
        assert "<title>Notebook</title>" not in page.read_text(encoding="utf-8")


def test_rewrite_nb_links_maps_relative_and_keeps_absolute():
    chrome = _load("thrml_render.chrome")
    rewrite = chrome.rewrite_nb_links
    rel = rewrite('<a href="02_spin_models.ipynb">x</a>')
    assert 'href="02_spin_models.html"' in rel
    anchor = rewrite('<a href="01_all_of_thrml.ipynb#section">x</a>')
    assert 'href="01_all_of_thrml.html#section"' in anchor
    absolute = '<a href="https://example.com/nb/a.ipynb">x</a>'
    assert rewrite(absolute) == absolute


def test_linkify_api_wraps_code_spans():
    api = _load("thrml_render.api_reference")
    out = api.linkify_api("<p>Use <code>Block</code> first.</p>")
    assert '<a class="api-xref"' in out
    assert 'href="api-blocks.html#Block"' in out
    assert "<code>Block</code>" in out


def test_class_members_render(site):
    # The renderer documents each class's members (methods/properties/attributes)
    # with linkable per-member anchors, not just the constructor signature.
    html = (site / "api-blocks.html").read_text(encoding="utf-8")
    assert 'class="api-members"' in html
    assert 'id="Block.nodes"' in html, "Block.nodes attribute member missing"
    assert 'id="Block.node_type"' in html, "Block.node_type property member missing"


def test_no_abstractvar_leaks_into_pages(site):
    # _public_fields skips equinox AbstractVar / typing.ClassVar annotations, so an
    # abstract interface field never renders as a concrete attribute.
    for page in site.glob("api-*.html"):
        assert "AbstractVar" not in page.read_text(encoding="utf-8"), f"AbstractVar leaked into {page.name}"


def test_api_annotations_keep_generic_subscripts(site):
    # Regression guard: attribute annotations must preserve generic subscripts
    # (tuple[~_Node, ...]) rather than collapsing to a bare name.
    text = "".join(page.read_text(encoding="utf-8") for page in site.glob("api-*.html"))
    assert any(token in text for token in ("tuple[", "list[", "dict[")), "generic subscripts stripped"


def test_no_asset_placeholder_remains(site):
    for page in site.glob("*.html"):
        assert "ASSET_BASE/" not in page.read_text(encoding="utf-8"), f"unresolved ASSET_BASE/ in {page.name}"


def test_notebook_catalog_is_consistent():
    config = _load("thrml_render.config")
    assert config.validate_notebook_catalog() == []


def test_replace_once_raises_on_missing_or_duplicate_marker():
    text = _load("thrml_render.text")
    with pytest.raises(RuntimeError):
        text.replace_once("no marker here", "</main>", "x</main>", "</main> marker")
    with pytest.raises(RuntimeError):
        text.replace_once("<a></a><a></a>", "<a>", "<b>", "<a> marker")


def test_annotation_text_strips_module_noise_without_mangling():
    # Regression: a naive ``typing.`` strip used to fuse ``jaxtyping.PyTree`` into
    # ``jaxPyTree``. The denylist must strip whole module prefixes only.
    api = _load("thrml_render.api_reference")
    assert api._annotation_text("dict[str, jaxtyping.PyTree]") == "dict[str, PyTree]"
    assert api._annotation_text("typing.Optional[int]") == "Optional[int]"
    assert "jaxPyTree" not in api._annotation_text("list[jaxtyping.PyTree]")


def test_no_mangled_jax_prefix_in_pages(site):
    for page in site.glob("api-*.html"):
        assert "jaxPyTree" not in page.read_text(encoding="utf-8"), f"mangled jaxtyping prefix in {page.name}"


def test_member_docstring_latex_not_mangled(site):
    # Raw-string LaTeX in member docstrings must survive: a non-raw "$\theta$" used to
    # be parsed as "$<tab>heta$" and render as broken MathJax ("$ heta$").
    html = (site / "api-samplers.html").read_text(encoding="utf-8")
    assert "\\theta" in html, "\\theta missing/mangled in rendered member docs"
    assert " heta$" not in html, "tab-mangled \\theta ($ heta$) shipped to the page"


def test_callable_protocol_documents_dunder_call(site):
    # __call__ is the contract method of callable protocols (AbstractObserver); member
    # rendering must surface it rather than filtering it out as a dunder.
    html = (site / "api-observers.html").read_text(encoding="utf-8")
    assert 'id="AbstractObserver.__call__"' in html, "__call__ omitted from AbstractObserver members"


def test_validate_catalog_tolerates_skip_render(monkeypatch):
    # Skipping a notebook (present on disk, kept off the site) must not false-fail
    # the catalog validator just because the notebook is still listed in the catalog.
    config = _load("thrml_render.config")
    stems = [p.stem for p in sorted(NOTEBOOKS.glob("[0-9]*.ipynb"))]
    assert stems, "no example notebooks found"
    monkeypatch.setattr(config, "SKIP_RENDER", {stems[-1]})
    assert config.validate_notebook_catalog() == []
