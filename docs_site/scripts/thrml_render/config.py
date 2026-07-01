"""Paths, URLs, and pure-data configuration for the THRML docs build."""

import html as html_lib
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
NB_DIR = ROOT.parent / "examples"
OUT_DIR = ROOT / "rendered"
BRAND_DIR = ROOT / "brand"
# Committed static files (e.g. the codon-optimization paper page and its figures)
# copied verbatim into rendered/ at build time so rendered/ stays fully derived.
STATIC_DIR = ROOT / "static"

INDEX_GITHUB = "https://github.com/extropic-ai/thrml"

# Licensed fonts/videos aren't committed (commercial license); the build copies
# them from the private docs-assets checkout into rendered/ and serves them
# page-relative (./fonts, ./videos) from the docs host -- no CDN. All pages live
# at the output root, so a relative ref resolves on prod and RTD preview alike.
ASSET_BASE = "."

# docs-assets checkout; Read the Docs clones it here at build time.
DOCS_ASSETS_DIR = Path(os.environ.get("THRML_DOCS_ASSETS") or (ROOT / "_assets"))


# The THRML mark (Extropic brand). The source is monochrome; recoloring its fill
# to currentColor lets it inherit text color and shimmer on hover like the rest
# of the chrome, exactly as the dot-triangle mark did.
def _load_logo():
    raw = (BRAND_DIR / "logo.svg").read_text(encoding="utf-8")
    # Drop inline style fills (they pin a fixed grey and win over the presentation
    # attribute), then recolor to currentColor so the mark inherits text color and
    # shimmers on hover like the rest of the chrome.
    raw = re.sub(r'\s*style="[^"]*"', "", raw)
    raw = re.sub(r'fill="#[0-9A-Fa-f]{6}"', 'fill="currentColor"', raw)
    raw = re.sub(r"<svg\b", '<svg class="thrml-logo" aria-hidden="true"', raw, count=1)
    raw = re.sub(r'\s(width|height)="\d+"', "", raw)
    return raw.strip()


LOGO_SVG = _load_logo()

# "assets" (not "figures") because a global .gitignore rule ignores figures/ dirs
# used for the notebook-local paper outputs; these externalized PNGs must be tracked.
FIG_DIR = OUT_DIR / "assets"

INDEX_BLURBS = {
    "00": "What a probabilistic computer is, how Extropic's sampling hardware works, and a first model sampled with THRML.",
    "01": "The whole library end to end: nodes and blocks, factors and interaction groups, programs, and block Gibbs sampling.",
    "02": "Ising and spin energy-based models built from scratch, then scaled to measure block-Gibbs throughput on 8 B200s.",
    "03": "A real design problem end to end: optimize a gene's codons by writing the objective as an energy function, building it as a Potts model and an equivalent Ising model, and sampling with simulated annealing.",
}

INDEX_SECTIONS = [
    (
        "Tutorials",
        "From a first Ising chain to the full sampling stack and hardware-scale spin models.",
        ["00", "01", "02"],
    ),
    ("Applications", "Real problems compiled to graphical models and sampled on thermodynamic hardware.", ["03"]),
]

MATHJAX = (
    "<script>window.MathJax = { tex: { "
    "inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], "
    "displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']], "
    "processEscapes: true }, "
    "svg: { fontCache: 'global' } };</script>\n"
    '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" id="MathJax-script" async></script>'
)

# Prerender hint shared by the landing page and the notebook/docs chrome. The
# rule is defined in one place here; SPECULATION_RULES is the block the chrome
# injects, and SPECULATION_RULES_INLINE is the single-line form the landing
# page's <head> carries alongside its other inline head fragments.
_SPECULATION_JSON = '{ "prerender": [{ "where": { "href_matches": "/*" }, "eagerness": "moderate" }] }'
SPECULATION_RULES = '<script type="speculationrules">\n' + _SPECULATION_JSON + "\n</script>\n"
SPECULATION_RULES_INLINE = '<script type="speculationrules">' + _SPECULATION_JSON + "</script>\n"

# API reference, generated from the installed thrml package and grouped the way
# the library's own mkdocs nav groups it. Each category names the module its
# symbols live on ("thrml" or "thrml.models") and the sidebar group it sits under.
API_CATEGORIES = [
    {
        "label": "Graphical model components",
        "slug": "api-pgm",
        "module": "thrml",
        "group": "Core",
        "blurb": "Nodes are the variables of a graphical model. A node carries the type and shape of one site's state.",
        "symbols": ["AbstractNode", "SpinNode", "CategoricalNode"],
    },
    {
        "label": "Block management",
        "slug": "api-blocks",
        "module": "thrml",
        "group": "Core",
        "blurb": "A block is an ordered collection of nodes of the same type, the unit that block Gibbs updates in parallel. These tools build blocks and map between block state and the packed global state.",
        "symbols": [
            "Block",
            "BlockSpec",
            "block_state_to_global",
            "from_global_state",
            "get_node_locations",
            "make_empty_block_state",
            "verify_block_state",
        ],
    },
    {
        "label": "Factors",
        "slug": "api-factors",
        "module": "thrml",
        "group": "Core",
        "blurb": "Factors organize the interactions between variables and synthesize them into interaction groups. A FactorSamplingProgram wraps a set of factors into a runnable sampler.",
        "symbols": ["AbstractFactor", "WeightedFactor", "FactorSamplingProgram"],
    },
    {
        "label": "Interaction groups",
        "slug": "api-interaction",
        "module": "thrml",
        "group": "Core",
        "blurb": "An interaction group is the compiled, array-friendly form of a factor's interactions, ready for block Gibbs.",
        "symbols": ["InteractionGroup"],
    },
    {
        "label": "Conditional samplers",
        "slug": "api-samplers",
        "module": "thrml",
        "group": "Core",
        "blurb": "Conditional samplers draw a block's new state given its neighbors. They are the per-block kernels that block Gibbs strings together.",
        "symbols": [
            "AbstractConditionalSampler",
            "AbstractParametricConditionalSampler",
            "BernoulliConditional",
            "SoftmaxConditional",
        ],
    },
    {
        "label": "Block sampling",
        "slug": "api-block-sampling",
        "module": "thrml",
        "group": "Core",
        "blurb": "The sampling engine: schedules, programs, and the entry points that run block Gibbs and read states back.",
        "symbols": [
            "SamplingSchedule",
            "BlockGibbsSpec",
            "BlockSamplingProgram",
            "sample_states",
            "sample_blocks",
            "sample_single_block",
            "sample_with_observation",
        ],
    },
    {
        "label": "Sampling observers",
        "slug": "api-observers",
        "module": "thrml",
        "group": "Core",
        "blurb": "Observers accumulate statistics over a chain as it runs, so you read off moments or stored states without materializing every sample.",
        "symbols": ["AbstractObserver", "StateObserver", "MomentAccumulatorObserver"],
    },
    {
        "label": "Energy-based models",
        "slug": "api-ebm",
        "module": "thrml.models",
        "group": "Models",
        "blurb": "Energy-based models define a distribution through an energy function. THRML factorizes that energy so block Gibbs can sample it.",
        "symbols": ["AbstractEBM", "AbstractFactorizedEBM", "FactorizedEBM", "EBMFactor"],
    },
    {
        "label": "Discrete energy-based models",
        "slug": "api-discrete-ebm",
        "module": "thrml.models",
        "group": "Models",
        "blurb": "Discrete EBM building blocks for spin and categorical variables, with square-tensor specializations and their matching Gibbs conditionals.",
        "symbols": [
            "DiscreteEBMFactor",
            "DiscreteEBMInteraction",
            "SquareDiscreteEBMFactor",
            "SpinEBMFactor",
            "CategoricalEBMFactor",
            "SquareCategoricalEBMFactor",
            "SpinGibbsConditional",
            "CategoricalGibbsConditional",
        ],
    },
    {
        "label": "Ising models",
        "slug": "api-ising",
        "module": "thrml.models",
        "group": "Models",
        "blurb": "A ready-made Ising energy-based model, its sampling program, and the utilities to initialize, train, and estimate its moments.",
        "symbols": [
            "IsingEBM",
            "IsingSamplingProgram",
            "IsingTrainingSpec",
            "hinton_init",
            "estimate_moments",
            "estimate_kl_grad",
        ],
    },
]

# Map each public API symbol to its reference page anchor, so notebook prose can
# link class and function names straight to the API reference.
API_SYMBOL_URL = {name: f"{cat['slug']}.html#{name}" for cat in API_CATEGORIES for name in cat["symbols"]}
_API_XREF_RE = re.compile(
    r"<code>(" + "|".join(re.escape(n) for n in sorted(API_SYMBOL_URL, key=len, reverse=True)) + r")</code>"
)
# Whole-token matcher for linking type names that appear as plain text (API
# signatures, docstring prose). Longest-first so BlockSpec wins over Block, and
# \b keeps Block from matching inside BlockSpec.
_API_TOKEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in sorted(API_SYMBOL_URL, key=len, reverse=True)) + r")\b"
)

# Normalize stray http links in the source notebooks at render time so the .ipynb
# files stay pristine.
_LINK_FIXES = {
    'href="http://arxiv.org/': 'href="https://arxiv.org/',
}


# Notebooks present in the repo but kept off the published site for now.
SKIP_RENDER = set()


SITE_URL = "https://docs.thrml.ai"


def notebook_numbers(paths=None):
    """Published notebook numbers (NN prefixes), in filesystem discovery order."""
    if paths is None:
        paths = sorted(NB_DIR.glob("[0-9]*.ipynb"))
    return [p.stem.split("_", 1)[0] for p in paths if p.stem not in SKIP_RENDER]


def validate_notebook_catalog(paths=None):
    """Cross-check the notebooks on disk against INDEX_SECTIONS and INDEX_BLURBS.

    Returns a list of human-readable error strings (empty when consistent) so the
    build can fail loudly on a duplicated number, a notebook missing from the
    sidebar catalog, or a blurb with no notebook, instead of silently dropping it
    from the nav and the examples gallery.
    """
    # A SKIP_RENDER notebook is present on disk but kept off the site, so exclude
    # its number from the catalog side too; otherwise the (still-listed) skipped
    # notebook looks like a mismatch and would false-fail the build.
    skip_numbers = {stem.split("_", 1)[0] for stem in SKIP_RENDER}
    numbers = notebook_numbers(paths)  # already excludes SKIP_RENDER stems
    section_numbers = [num for _name, _blurb, nums in INDEX_SECTIONS for num in nums if num not in skip_numbers]
    blurb_numbers = set(INDEX_BLURBS) - skip_numbers
    errors = []
    if len(set(numbers)) != len(numbers):
        errors.append(f"duplicate notebook numbers: {numbers}")
    if len(set(section_numbers)) != len(section_numbers):
        errors.append(f"duplicate INDEX_SECTIONS numbers: {section_numbers}")
    if set(numbers) != set(section_numbers):
        errors.append("notebook/catalog mismatch: " f"notebooks={sorted(numbers)}, catalog={sorted(section_numbers)}")
    missing_blurbs = sorted(set(numbers) - blurb_numbers)
    extra_blurbs = sorted(blurb_numbers - set(numbers))
    if missing_blurbs:
        errors.append(f"missing INDEX_BLURBS entries: {missing_blurbs}")
    if extra_blurbs:
        errors.append(f"INDEX_BLURBS entries without notebooks: {extra_blurbs}")
    return errors


DEFAULT_DESCRIPTION = (
    "THRML is a JAX library for block Gibbs sampling of probabilistic hypergraphical "
    "and energy-based models, built to prototype on Extropic's thermodynamic sampling hardware."
)


def og_meta(title, page, description=DEFAULT_DESCRIPTION):
    """Open Graph + Twitter-card <head> tags for social link unfurls.

    ``page`` is the rendered filename (e.g. ``index.html``); the canonical URL and
    the shared 1200x630 card image are built against the live deploy host.
    """
    # Canonicalize to the published docs host; og.png ships into the site root.
    base = SITE_URL.rstrip("/")
    # the homepage's canonical URL is the bare host, not /index.html
    url = f"{base}/" if page == "index.html" else f"{base}/{page}"
    image = f"{base}/og.png"
    title = html_lib.escape(title)
    description = html_lib.escape(description)
    return (
        '<meta property="og:type" content="website">\n'
        '<meta property="og:site_name" content="THRML">\n'
        f'<meta property="og:title" content="{title}">\n'
        f'<meta property="og:description" content="{description}">\n'
        f'<meta property="og:url" content="{url}">\n'
        f'<meta property="og:image" content="{image}">\n'
        '<meta property="og:image:width" content="1200">\n'
        '<meta property="og:image:height" content="630">\n'
        # X/Twitter, Slack, and iMessage fall back to the og:* tags, so only
        # twitter:card (which requests the large-image layout) is non-redundant.
        '<meta name="twitter:card" content="summary_large_image">\n'
    )
