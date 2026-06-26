"""Small text-substitution helpers shared across the renderer."""

from __future__ import annotations


def replace_once(text: str, old: str, new: str, label: str) -> str:
    """Substitute the single expected occurrence of ``old`` with ``new``.

    Raises if ``old`` does not appear exactly once, so a template change or a
    notebook cell that emits a stray marker (e.g. raw HTML containing a literal
    ``</main>``) fails loudly at build time instead of silently injecting chrome
    at the wrong spot. ``label`` names the marker in the error.
    """
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label}, found {count}")
    return text.replace(old, new, 1)
