"""Shared page chrome (top bar, sidebar, nav) and notebook link rewriting."""

import html as html_lib
import re
from pathlib import Path

from .assets import COLLAPSE_SCRIPT, COLLAPSE_STYLE, css, js
from .config import (
    _LINK_FIXES,
    API_CATEGORIES,
    INDEX_GITHUB,
    INDEX_SECTIONS,
    LOGO_SVG,
    SPECULATION_RULES,
    og_meta,
)
from .text import replace_once


def build_topbar():
    return (
        '<header class="thrml-topbar">'
        '<div class="thrml-bar-left">'
        '<button class="thrml-burger" type="button" aria-label="Toggle navigation">'
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>'
        "</button>"
        '<a class="thrml-brand" href="index.html">' + LOGO_SVG + '<span class="thrml-title">THRML</span></a>'
        "</div>"
        '<nav class="thrml-pills">'
        '<a class="thrml-pill" href="getting-started.html">Get started</a>'
        '<a class="thrml-pill" href="examples.html">Examples</a>'
        '<a class="thrml-pill" href="papers/codon-optimization/">Paper</a>'
        '<a class="thrml-pill" href="' + INDEX_GITHUB + '">GitHub</a>'
        "</nav></header>"
    )


def build_sidebar(entries, active_stem=None, active_page=None, active_api=None):
    by_num = {number: (title, href) for number, title, href in entries}
    parts = ['<aside class="thrml-sidebar"><nav class="thrml-sidenav">']

    def doc_link(slug, label):
        cls = "sb-link active" if active_page == slug else "sb-link"
        return f'<a class="{cls}" href="{slug}.html">{label}</a>'

    parts.append(doc_link("getting-started", "Getting started"))
    parts.append(doc_link("concepts", "Concepts"))
    parts.append('<div class="sb-section">Examples</div>')
    for name, _sub, nums in INDEX_SECTIONS:
        parts.append(f'<div class="sb-group"><div class="sb-grouphead">{html_lib.escape(name, quote=False)}</div>')
        for number in nums:
            if number not in by_num:
                continue
            title, href = by_num[number]
            stem = href[:-5]
            cls = "sb-nb active" if active_stem == stem else "sb-nb"
            parts.append(
                f'<a class="{cls}" href="{html_lib.escape(href, quote=True)}"><span class="sb-n">{number}</span><span class="sb-t">{html_lib.escape(title, quote=False)}</span></a>'
            )
        parts.append("</div>")
    parts.append('<div class="sb-section">API reference</div>')
    current_group = None
    for cat in API_CATEGORIES:
        if cat["group"] != current_group:
            if current_group is not None:
                parts.append("</div>")
            parts.append(f'<div class="sb-group"><div class="sb-grouphead">{cat["group"]}</div>')
            current_group = cat["group"]
        cls = "sb-link active" if active_api == cat["slug"] else "sb-link"
        parts.append(f'<a class="{cls}" href="{cat["slug"]}.html">{cat["label"]}</a>')
    if current_group is not None:
        parts.append("</div>")
    parts.append("</nav></aside>")
    return "".join(parts)


def _inject_body(html, body_class, chrome):
    """Add a body class and inject the chrome markup in one pass over <body>.

    Raises if there is not exactly one ``<body>`` so a notebook cell that emits a
    stray ``<body`` (or a template change) fails loudly instead of injecting the
    sidebar at the wrong spot."""
    if html.count("<body") != 1:
        raise RuntimeError(f"expected exactly one <body> marker, found {html.count('<body')}")
    match = re.search(r"<body([^>]*)>", html)
    if not match:
        raise RuntimeError("missing <body> marker")
    attrs = match.group(1)
    if 'class="' in attrs:
        attrs = re.sub(r'class="', f'class="{body_class} ', attrs, count=1)
    else:
        attrs = attrs + f' class="{body_class}"'
    return html[: match.start()] + f"<body{attrs}>{chrome}" + html[match.end() :]


def reading_order(entries):
    """Notebook entries flattened into the sidebar reading order."""
    by_num = {number: (title, href) for number, title, href in entries}
    ordered = []
    for _name, _sub, nums in INDEX_SECTIONS:
        for number in nums:
            if number in by_num:
                title, href = by_num[number]
                ordered.append((number, title, href))
    return ordered


def prev_next_nav(entries, active_stem):
    """Previous / next notebook links for the bottom of a notebook page."""
    ordered = reading_order(entries)
    idx = next((i for i, (_n, _t, href) in enumerate(ordered) if href[:-5] == active_stem), None)
    if idx is None:
        return ""

    def card(entry, direction):
        number, title, href = entry
        arrow = "&larr; Previous" if direction == "prev" else "Next &rarr;"
        return (
            f'<a class="thrml-pn thrml-pn-{direction}" href="{href}">'
            f'<span class="thrml-pn-dir">{arrow}</span>'
            f'<span class="thrml-pn-title"><span class="thrml-pn-n">{number}</span>'
            f"{html_lib.escape(title)}</span></a>"
        )

    prev_html = card(ordered[idx - 1], "prev") if idx > 0 else '<span class="thrml-pn thrml-pn-empty"></span>'
    next_html = (
        card(ordered[idx + 1], "next") if idx < len(ordered) - 1 else '<span class="thrml-pn thrml-pn-empty"></span>'
    )
    return '<nav class="thrml-pagenav" aria-label="Notebook navigation">' + prev_html + next_html + "</nav>"


def inject_chrome(html, entries, active_stem, title):
    """Add the theme, top bar, sidebar, and social-card metadata to a notebook page."""
    head_extra = og_meta(f"{title} · THRML", f"{active_stem}.html") + css(COLLAPSE_STYLE)
    html = replace_once(html, "</head>", "\n" + head_extra + "</head>", "</head> marker")
    chrome = build_topbar() + build_sidebar(entries, active_stem=active_stem)
    html = _inject_body(html, "thrml-has-sidebar", chrome)
    nav = prev_next_nav(entries, active_stem)
    if nav:
        # insert before the template's closing </main> (the last one), so a stray
        # </main> in a notebook cell's raw-HTML output can't misplace the nav
        idx = html.rfind("</main>")
        if idx == -1:
            raise RuntimeError("expected a </main> marker, found none")
        html = html[:idx] + nav + html[idx:]
    return replace_once(html, "</body>", "\n" + js(COLLAPSE_SCRIPT) + SPECULATION_RULES + "</body>", "</body> marker")


def rewrite_nb_links(html):
    """Rewrite relative cross-notebook links from .ipynb to .html. Absolute URLs
    (which contain ``://``) are left untouched."""
    return re.sub(r'href="([^":#]*?)\.ipynb(#[^"]*)?"', _nb_to_html, html)


def rewrite_local_images(html):
    """Point notebook <img> tags at file-backed figures (e.g. fps.png, saved by a
    notebook and committed beside it) to the externalized copy under assets/."""
    return re.sub(r'src="\.{0,2}/?([\w./-]+\.png)"', lambda m: f'src="assets/{Path(m.group(1)).name}"', html)


def fix_known_links(html):
    for bad, good in _LINK_FIXES.items():
        html = html.replace(bad, good)
    return html


def _nb_to_html(m):
    stem, anchor = m.group(1), m.group(2) or ""
    return f'href="{stem}.html{anchor}"'
