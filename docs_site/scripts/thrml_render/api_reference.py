"""API reference: introspect the live thrml package and render docstrings."""

import html as html_lib
import importlib
import inspect
import re

from .config import (
    _API_TOKEN_RE,
    _API_XREF_RE,
    API_SYMBOL_URL,
)


def linkify_api(html, skip=None):
    """Link bare API-name code spans (markdown inline code) to the API reference.

    Only matches whole `<code>Name</code>` spans, so code cell source and outputs,
    which use different markup, are left untouched. ``skip`` suppresses self-links
    (e.g. a symbol linking to its own page from its own section).
    """
    skip = skip or set()

    def repl(m):
        name = m.group(1)
        if name in skip:
            return m.group(0)
        return f'<a class="api-xref" href="{API_SYMBOL_URL[name]}"><code>{name}</code></a>'

    return _API_XREF_RE.sub(repl, html)


def linkify_types(text, self_name=None):
    """Link API type names that appear as plain (escaped) text, e.g. inside a
    signature or a docstring sentence, to their reference. Skips the symbol's own
    name so a definition does not link to itself."""

    def repl(m):
        name = m.group(1)
        if name == self_name:
            return name
        return f'<a class="api-xref-plain" href="{API_SYMBOL_URL[name]}">{name}</a>'

    return _API_TOKEN_RE.sub(repl, text)


def _api_clean_sig(obj):
    # jitted functions (equinox.filter_jit, jax.jit) hide the real signature behind
    # a wrapper; unwrap to the underlying callable so the signature is meaningful.
    target = getattr(obj, "__wrapped__", obj)
    try:
        sig = inspect.signature(target.__init__ if inspect.isclass(target) else target)
    except (ValueError, TypeError):
        return "(...)"
    s = str(sig).replace("self, ", "").replace("self", "")
    for pat in (
        r"jax\.jaxlib\._jax\.",
        r"jaxtyping\.",
        r"jaxlib\._jax\.",
        r"collections\.abc\.",
        r"equinox\._[\w.]+\.",
        r"thrml\.[\w.]+\.",
    ):
        s = re.sub(pat, "", s)
    return re.sub(r" -> None$", "", s)


def _api_inline(s):
    s = html_lib.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+?)\*\*", r"<strong>\1</strong>", s)
    return s


def _api_body(text):
    out, para, in_list = [], [], False

    def flush_para():
        if para:
            out.append("<p>" + _api_inline(" ".join(para)) + "</p>")
            para.clear()

    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("- "):
            flush_para()
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + _api_inline(s[2:]) + "</li>")
        elif not s:
            flush_para()
            if in_list:
                out.append("</ul>")
                in_list = False
        else:
            para.append(s)
    flush_para()
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def render_docstring(doc):
    """Render a THRML docstring (light markdown + LaTeX) to themed HTML for MathJax."""
    math = []

    def stash(m):
        math.append(m.group(0))
        return f"\x00M{len(math) - 1}\x00"

    doc = re.sub(r"\$\$.*?\$\$", stash, doc, flags=re.S)
    doc = re.sub(r"\$[^$\n]+?\$", stash, doc)
    lines = doc.split("\n")
    out, para = [], []
    in_list = [False]

    def flush():
        if para:
            out.append("<p>" + _api_inline(" ".join(para)) + "</p>")
            para.clear()

    def close_list():
        if in_list[0]:
            out.append("</ul>")
            in_list[0] = False

    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        admon = re.match(r'\?{3}\+?\s*\w+\s*"?([^"]*)"?\s*$', ln)
        if admon:
            flush()
            close_list()
            title = admon.group(1) or "Note"
            i += 1
            body = []
            while i < len(lines) and (lines[i].strip() == "" or lines[i].startswith("    ")):
                body.append(lines[i][4:] if lines[i].startswith("    ") else "")
                i += 1
            inner = _api_body("\n".join(body))
            for k, mx in enumerate(math):
                inner = inner.replace(f"\x00M{k}\x00", mx)
            out.append(f'<div class="api-note"><span class="api-note-t">{_api_inline(title)}</span>{inner}</div>')
            continue
        if re.match(r"^\x00M\d+\x00$", ln):
            flush()
            close_list()
            out.append(f'<div class="api-disp">{ln}</div>')
            i += 1
            continue
        if ln.startswith("- "):
            flush()
            if not in_list[0]:
                out.append("<ul>")
                in_list[0] = True
            item = [ln[2:]]
            i += 1
            while i < len(lines) and lines[i].startswith("    ") and not lines[i].strip().startswith("- "):
                item.append(lines[i].strip())
                i += 1
            out.append("<li>" + _api_inline(" ".join(item)) + "</li>")
            continue
        if ln == "":
            flush()
            close_list()
            i += 1
            continue
        para.append(ln)
        i += 1
    flush()
    close_list()
    res = "\n".join(out)
    for k, mx in enumerate(math):
        res = res.replace(f"\x00M{k}\x00", mx)
    return res


def api_inner(cat):
    mod = importlib.import_module(cat["module"])
    parts = [f"<h1>{cat['label']}</h1>\n", f'<p class="lede">{linkify_types(cat["blurb"])}</p>\n']
    for name in cat["symbols"]:
        obj = getattr(mod, name, None)
        if obj is None:
            continue
        target = getattr(obj, "__wrapped__", obj)
        sig = linkify_types(html_lib.escape(name + _api_clean_sig(obj)), self_name=name)
        doc = inspect.getdoc(target) or ""
        kind = "class" if inspect.isclass(target) else "function"
        parts.append(
            f'<section class="api-sym" id="{name}">'
            f'<div class="api-head"><code class="api-name">{name}</code><span class="api-kind">{kind}</span></div>'
            f'<pre class="api-sig"><code>{sig}</code></pre>'
            f'<div class="api-doc">{linkify_api(render_docstring(doc), skip={name})}</div>'
            "</section>"
        )
    return "".join(parts)
