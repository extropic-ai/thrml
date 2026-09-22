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
- `brand/wordmark.svg` supplies the THRML lettering on the landing page, docs,
  notebooks, and paper. It follows Torx's vector wordmark treatment, reusing its
  T/R paths and matching the remaining letters to the same stroke geometry.
- The brand fonts and hero/footer videos are licensed. For a complete local
  preview, run `gh repo clone extropic-ai/docs-assets docs_site/_assets` before
  building. Local builds without that private checkout use system fonts.
- Read the Docs requires `DOCS_ASSETS_TOKEN` with read access to
  `extropic-ai/docs-assets`. A missing or invalid credential fails the hosted
  build instead of publishing missing fonts/videos. Assets are copied into the
  output and served from the docs host, with no external CDN.

## Adding a public API symbol

The API reference symbol lists are hand-maintained in `API_CATEGORIES` in
`thrml_render/config.py`. A new public export only appears in the reference
once its name is added to that list.
