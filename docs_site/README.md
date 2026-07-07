# THRML docs site

The THRML documentation site: a landing page, hand-authored prose docs, an
auto-generated API reference, and every example notebook, all sharing one
Extropic theme.

## Build

The API reference is introspected from the live `thrml` package, so the build
needs an editable install first:

```sh
pip install -e .[docs]
uv run python docs_site/scripts/render_html.py
```

This regenerates everything under `docs_site/rendered/` (HTML pages, the API
pages, externalized notebook figures, and `llms.txt`). That directory is build
output and is gitignored.

## Inputs

- `scripts/render_html.py` builds the site.
- `brand/` holds the logo and brand images.
- The example notebooks are single-sourced from the repo's `examples/`
  directory (`NB_DIR = ROOT.parent / "examples"`), so adding a notebook there
  publishes it on the site.
- The brand fonts and hero/footer videos are licensed; the build pulls them from
  the private `docs-assets` repo (cloned into `docs_site/_assets`) and serves
  them from the docs host, so no external CDN is referenced.

## Adding a public API symbol

The API reference symbol lists are hand-maintained in `API_CATEGORIES` in
`thrml_render/config.py`. A new public export only appears in the reference
once its name is added to that list.
