"""THRML docs site renderer, split into cohesive modules.

``scripts/render_html.py`` (one directory up) is the entry point; this package
holds the pieces:

* ``config`` -- paths, URLs, and pure-data configuration.
* ``assets`` -- the raw CSS/JS blobs read from ``scripts/assets/``, wrapped in
  ``<style>``/``<script>`` at the emit point.
* ``notebooks`` -- notebook export and image externalization.
* ``api_reference`` -- live-introspected API reference + docstring rendering.
* ``chrome`` -- shared page chrome and notebook link rewriting.
* ``pages`` -- hand-authored page builders (including the docs code card) and
  the HTML emitters.
"""
