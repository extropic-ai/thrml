"""Render the THRML docs site: a landing page, hand-authored prose docs, the API
reference, and every example notebook, all sharing one warm-dark Extropic theme.

The site has three kinds of page:

* ``index.html`` -- a full-bleed landing (hero, core data structures, quickstart,
  applications, and a card grid of every notebook).
* ``getting-started.html`` / ``concepts.html`` -- hand-authored prose docs that
  sit inside a left sidebar, alongside the notebooks and the API reference.
* ``NN_name.html`` -- each example notebook, exported from its ``.ipynb`` with the
  same sidebar chrome.

GitHub ignores ``metadata.jupyter.source_hidden``; these renders honor it, so a
cell collapsed in JupyterLab folds behind a gutter chevron here and the page
opens results-forward. Figures are externalized, so each file stays light.

Usage:
    uv run python docs_site/scripts/render_html.py

The source notebooks are never modified: the hide-input tagging happens on an
in-memory copy used only for export.
"""

import html as html_lib
import shutil

import nbformat
from thrml_render.api_reference import api_inner, linkify_api, validate_api_reference
from thrml_render.chrome import (
    fix_known_links,
    inject_chrome,
    rewrite_local_images,
    rewrite_nb_links,
)
from thrml_render.config import (
    API_CATEGORIES,
    BRAND_DIR,
    DOCS_ASSETS_DIR,
    FIG_DIR,
    NB_DIR,
    OUT_DIR,
    ROOT,
    SKIP_RENDER,
    STATIC_DIR,
    validate_notebook_catalog,
)
from thrml_render.notebooks import (
    build_exporter,
    externalize_images,
    notebook_title,
    tag_hidden_inputs,
)
from thrml_render.pages import (
    concepts_inner,
    examples_inner,
    getting_started_inner,
    write_doc_page,
    write_index,
    write_llms_txt,
)
from thrml_render.text import replace_once


def _fail_if_errors(errors, label):
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(f"{label} failed:\n{joined}")


def copy_static():
    """Copy the file-backed assets the pages reference into rendered/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # The social-card image ships at the site root so og:image resolves to /og.png.
    shutil.copy2(BRAND_DIR / "og.png", OUT_DIR / "og.png")
    # Brand figures committed beside the notebooks (examples/) or in brand/.
    for src, dst in [
        (NB_DIR / "fps.png", FIG_DIR / "fps.png"),
        (NB_DIR / "codon_pipeline.png", FIG_DIR / "codon_pipeline.png"),
        (BRAND_DIR / "flow.png", FIG_DIR / "flow.png"),
        (BRAND_DIR / "extropic_wordmark.png", FIG_DIR / "extropic_wordmark.png"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)
    copy_licensed_assets()
    # Committed static pages (e.g. the codon-optimization paper) copied verbatim,
    # so rendered/ holds them without a separate render step.
    if STATIC_DIR.is_dir():
        for src in STATIC_DIR.rglob("*"):
            if src.is_file():
                dst = OUT_DIR / src.relative_to(STATIC_DIR)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)


def copy_licensed_assets():
    """Copy licensed fonts/videos from the docs-assets checkout into
    rendered/. Absent checkout (local/CI) -> skip; pages fall back to system fonts."""
    for sub in ("fonts", "videos"):
        src_dir = DOCS_ASSETS_DIR / sub
        if not src_dir.is_dir():
            print(f"licensed assets: no {src_dir}; skipping {sub}/")
            continue
        dst_dir = OUT_DIR / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_dir.iterdir()):
            if src.is_file():
                shutil.copy2(src, dst_dir / src.name)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    # Fail loudly on a notebook/catalog mismatch; warn (don't fail) on API drift.
    _fail_if_errors(validate_notebook_catalog(), "notebook catalog validation")
    validate_api_reference()
    copy_static()
    exporter = build_exporter()

    # First pass: collect the notebook entries so the sidebar is complete.
    notebooks = []
    for path in sorted(NB_DIR.glob("[0-9]*.ipynb")):
        if path.stem in SKIP_RENDER:
            continue
        nb = nbformat.read(path, as_version=4)
        number = path.stem.split("_", 1)[0]
        title = notebook_title(nb, path.stem)
        notebooks.append((path, nb, number, title))
    entries = [(number, title, f"{path.stem}.html") for path, _nb, number, title in notebooks]

    # Render each notebook with the shared chrome.
    for path, nb, _number, title in notebooks:
        tag_hidden_inputs(nb)
        body, _ = exporter.from_notebook_node(nb)
        body = replace_once(
            body,
            "<title>Notebook</title>",
            f"<title>{html_lib.escape(title)} &middot; THRML</title>",
            "notebook title marker",
        )
        body = externalize_images(body, path.stem)
        body = rewrite_local_images(body)
        body = rewrite_nb_links(body)
        body = fix_known_links(body)
        body = linkify_api(body)
        body = inject_chrome(body, entries, active_stem=path.stem, title=title)
        out = OUT_DIR / f"{path.stem}.html"
        out.write_text(body, encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}  ({len(body) // 1024} KB)")

    # Docs pages and landing.
    write_doc_page("getting-started", "Getting started", getting_started_inner(entries), entries, "getting-started")
    write_doc_page("concepts", "Concepts", concepts_inner(entries), entries, "concepts", mathjax=True)
    write_doc_page("examples", "Examples", examples_inner(entries), entries, "examples")
    for cat in API_CATEGORIES:
        write_doc_page(cat["slug"], cat["label"], api_inner(cat), entries, None, mathjax=True, active_api=cat["slug"])
    write_index(entries)
    write_llms_txt(entries)
    print(
        f"\n{len(entries)} notebooks + getting-started + concepts + {len(API_CATEGORIES)} API pages "
        f"+ index + llms.txt rendered to {OUT_DIR.relative_to(ROOT)}/"
    )


if __name__ == "__main__":
    main()
