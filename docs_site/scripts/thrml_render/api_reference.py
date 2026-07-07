"""API reference: introspect the live thrml package and render docstrings.

Each documented class renders its members (methods, properties, and attributes
with their signatures and docstrings), reachable through the MRO and ordered by
source line, so abstract bases and concrete classes alike show their interface
rather than just a constructor signature. A symbol listed in a category but
absent from its module is skipped with a warning, so a thrml API change never
silently drops a symbol from the docs.
"""

import functools
import html as html_lib
import importlib
import inspect
import math
import re
import sys
import typing

import equinox

from .config import (
    _API_TOKEN_RE,
    _API_XREF_RE,
    API_CATEGORIES,
    API_SYMBOL_URL,
)

# Distinguishes "the module has no such attribute" from "it exports it as None";
# a bare ``getattr(..., None) is None`` conflates the two.
_MISSING = object()


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


# Module-path noise stripped from rendered types; shared by the signature path
# and the attribute-annotation path so the two can never diverge.
_TYPE_NOISE_PATTERNS = (
    r"jax\.jaxlib\._jax\.",
    r"jaxtyping\.",
    r"jaxlib\._jax\.",
    r"collections\.abc\.",
    r"equinox\._[\w.]+\.",
    r"thrml\.[\w.]+\.",
    # Negative lookbehind (not ``\b``) so a leading ``jaxtyping.`` (handled above)
    # or ``jax.typing.`` is never partially matched and fused (jaxPyTree / jax.).
    r"(?<![\w.])typing\.",
)


def _strip_type_noise(s):
    # ``<class 'pkg.Name'>`` -> ``pkg.Name`` (the module-path patterns below then
    # trim the prefix). In _strip_type_noise so the signature and attribute paths
    # share it and cannot diverge.
    s = re.sub(r"<class '([^']+)'>", r"\1", s)
    # a function default reprs as ``<function f at 0x...>`` (non-deterministic
    # address -> non-reproducible build); reduce to the bare name, and strip any
    # remaining ``at 0x...`` address from other object reprs.
    s = re.sub(r"<function (\w+) at 0x[0-9a-fA-F]+>", r"\1", s)
    s = re.sub(r" at 0x[0-9a-fA-F]+", "", s)
    for pat in _TYPE_NOISE_PATTERNS:
        s = re.sub(pat, "", s)
    return s


def _api_clean_sig(obj):
    # jitted functions (equinox.filter_jit, jax.jit) hide the real signature behind
    # a wrapper; unwrap to the underlying callable so the signature is meaningful.
    target = getattr(obj, "__wrapped__", obj)
    try:
        sig = inspect.signature(target.__init__ if inspect.isclass(target) else target)
    except (ValueError, TypeError):
        return "(...)"
    s = str(sig).replace("self, ", "").replace("self", "")
    return re.sub(r" -> None$", "", _strip_type_noise(s))


def _annotation_text(annotation):
    """Render a field annotation to a compact string, preserving subscripts.

    ``str(annotation)`` keeps subscripts (``dict[str, Array]``, ``tuple[int, ...]``)
    that ``__name__`` would have dropped; the module-noise denylist then strips
    the long dotted paths.
    """
    s = annotation if isinstance(annotation, str) else str(annotation)
    return _strip_type_noise(s).strip()


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
    math_spans = []

    def stash(m):
        math_spans.append(m.group(0))
        return f"\x00M{len(math_spans) - 1}\x00"

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
            for k, mx in enumerate(math_spans):
                inner = inner.replace(f"\x00M{k}\x00", html_lib.escape(mx, quote=False))
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
    for k, mx in enumerate(math_spans):
        res = res.replace(f"\x00M{k}\x00", html_lib.escape(mx, quote=False))
    return res


def _member_target(static) -> typing.Any:
    """Unwrap a descriptor to the callable that carries the signature/docstring.

    Returns ``Any`` deliberately: the result is an arbitrary unwrapped object fed
    to ``inspect.getsourcelines``/``signature`` (whose failures the callers catch),
    so this is the one justified type boundary in the renderer (cf. torx's typed
    ``_find_repo_root``)."""
    if isinstance(static, property):
        return static.fget
    if isinstance(static, functools.cached_property):
        return static.func
    if isinstance(static, (staticmethod, classmethod)):
        return static.__func__
    return static


# Sort key for members whose source line cannot be located (builtins, C-level);
# they sort after every locatable member rather than at a magic line number.
_UNLOCATABLE_LINENO = math.inf


def _member_lineno(cls, name):
    try:
        static = inspect.getattr_static(cls, name)
    except AttributeError:
        static = getattr(cls, name, None)
    target = _member_target(static)
    try:
        return inspect.getsourcelines(target)[1]
    except (OSError, TypeError):
        return _UNLOCATABLE_LINENO


def _method_names(cls):
    """Public methods and properties on cls (plus ``__call__``), inherited, source-ordered."""
    names = []
    for name in dir(cls):
        # Skip private/dunder names except __call__, the contract method of the
        # callable protocols (AbstractObserver, the conditional samplers).
        if name.startswith("_") and name != "__call__":
            continue
        try:
            static = inspect.getattr_static(cls, name)
        except AttributeError:
            continue
        is_member = (
            isinstance(static, (property, functools.cached_property, staticmethod, classmethod))
            or inspect.isfunction(static)
            or inspect.ismethod(static)
        )
        if is_member:
            names.append(name)
    return sorted(names, key=lambda n: _member_lineno(cls, n))


def _is_abstract_annotation(ftype):
    """True if the annotation marks an abstract or class-level field that is not a
    concrete instance attribute (``equinox.AbstractVar`` / ``typing.ClassVar``)."""
    if ftype is equinox.AbstractVar or ftype is typing.ClassVar:
        return True
    if typing.get_origin(ftype) in (equinox.AbstractVar, typing.ClassVar):
        return True
    if isinstance(ftype, str) and re.match(r"(\w+\.)?(AbstractVar|ClassVar)\b", ftype):
        return True
    return False


def _public_fields(cls):
    """Public concrete annotated fields across the MRO (equinox/dataclass attrs).

    Abstract interface annotations (``AbstractVar``) and class-level ``ClassVar``
    annotations are skipped, as are names a subclass overrides with a
    property/descriptor, so an abstract ``dims: AbstractVar`` never leaks in and
    shadows the concrete ``dims`` property.
    """
    seen = set()
    fields = []
    for klass in inspect.getmro(cls):
        if klass is object:
            continue
        for fname, ftype in getattr(klass, "__annotations__", {}).items():
            if fname.startswith("_") or fname in seen:
                continue
            seen.add(fname)
            if _is_abstract_annotation(ftype):
                continue
            try:
                static = inspect.getattr_static(cls, fname)
            except AttributeError:
                static = None
            if isinstance(static, (property, functools.cached_property)) or inspect.isdatadescriptor(static):
                continue
            fields.append((fname, ftype))
    return fields


def _render_field(cls_name, fname, ftype):
    name = html_lib.escape(fname)
    sig = linkify_types(html_lib.escape(f"{fname}: {_annotation_text(ftype)}"), self_name=cls_name)
    return (
        f'<div class="api-member" id="{cls_name}.{fname}">'
        f'<div class="api-head"><code class="api-name">{name}</code>'
        f'<span class="api-kind">attribute</span></div>'
        f'<pre class="api-sig"><code>{sig}</code></pre>'
        "</div>"
    )


def _render_member(cls, name, cls_name):
    try:
        static = inspect.getattr_static(cls, name)
    except AttributeError:
        static = getattr(cls, name, None)
    target = _member_target(static)
    if isinstance(static, (property, functools.cached_property)):
        kind, sig = "property", ""
    elif isinstance(static, staticmethod):
        kind, sig = "staticmethod", _api_clean_sig(target)
    elif isinstance(static, classmethod):
        # drop the bound ``cls`` the way the receiver is dropped for methods
        kind, sig = "classmethod", re.sub(r"^\(cls(?:, )?", "(", _api_clean_sig(target))
    else:
        kind, sig = "method", _api_clean_sig(target)
    doc = inspect.getdoc(target) or ""
    sig_html = linkify_types(html_lib.escape(name + sig), self_name=cls_name)
    body = linkify_api(render_docstring(doc), skip={cls_name}) if doc else ""
    return (
        f'<div class="api-member" id="{cls_name}.{name}">'
        f'<div class="api-head"><code class="api-name">{html_lib.escape(name)}</code>'
        f'<span class="api-kind">{kind}</span></div>'
        f'<pre class="api-sig"><code>{sig_html}</code></pre>'
        f'<div class="api-doc">{body}</div>'
        "</div>"
    )


def _render_members(cls, cls_name):
    """Field and method member cards for a class, fields first, then methods in
    source order. Names carried by both (a field shadowed by a property) render
    once, as a field."""
    fields = _public_fields(cls)
    field_names = {fname for fname, _ in fields}
    out = [_render_field(cls_name, fname, ftype) for fname, ftype in fields]
    for member in _method_names(cls):
        if member in field_names:
            continue
        out.append(_render_member(cls, member, cls_name))
    return "".join(out)


def validate_api_reference():
    """Warn (without raising) about symbols listed in a category but absent from
    their module, so a thrml API rename surfaces in the build log instead of
    silently dropping the symbol from the docs."""
    missing = []
    for cat in API_CATEGORIES:
        mod = importlib.import_module(cat["module"])
        for name in cat["symbols"]:
            if not hasattr(mod, name):
                missing.append(f"{cat['module']}.{name}")
    if missing:
        print(
            "warning: API_CATEGORIES lists symbols missing from their module "
            f"(they will be skipped): {sorted(missing)}",
            file=sys.stderr,
        )


def api_inner(cat):
    mod = importlib.import_module(cat["module"])
    parts = [f"<h1>{cat['label']}</h1>\n", f'<p class="lede">{linkify_types(cat["blurb"])}</p>\n']
    for name in cat["symbols"]:
        obj = getattr(mod, name, _MISSING)
        if obj is _MISSING:
            print(
                f"warning: API symbol {name!r} is not exported by {cat['module']}; skipping",
                file=sys.stderr,
            )
            continue
        target = getattr(obj, "__wrapped__", obj)
        sig = linkify_types(html_lib.escape(name + _api_clean_sig(obj)), self_name=name)
        doc = inspect.getdoc(target) or ""
        kind = "class" if inspect.isclass(target) else "function"
        section = [
            f'<section class="api-sym" id="{name}">'
            f'<div class="api-head"><code class="api-name">{name}</code><span class="api-kind">{kind}</span></div>'
            f'<pre class="api-sig"><code>{sig}</code></pre>'
            f'<div class="api-doc">{linkify_api(render_docstring(doc), skip={name})}</div>'
        ]
        if inspect.isclass(target):
            members = _render_members(target, name)
            if members:
                section.append(f'<div class="api-members">{members}</div>')
        section.append("</section>")
        parts.append("".join(section))
    return "".join(parts)
