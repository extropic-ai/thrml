"""Hand-authored page builders and the HTML emitters that write them out."""

import html as html_lib
import importlib

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

from .api_reference import linkify_api
from .assets import (
    COLLAPSE_SCRIPT,
    COLLAPSE_STYLE,
    DOC_CSS,
    INDEX_CSS,
    INDEX_SCRIPT,
    css,
    js,
)
from .chrome import build_sidebar, build_topbar
from .config import (
    API_CATEGORIES,
    API_SYMBOL_URL,
    ASSET_BASE,
    INDEX_BLURBS,
    INDEX_GITHUB,
    INDEX_SECTIONS,
    LOGO_SVG,
    MATHJAX,
    OUT_DIR,
    SITE_URL,
    SPECULATION_RULES,
    SPECULATION_RULES_INLINE,
    og_meta,
)


def code_card(code, lang="python"):
    """A self-contained docs code card matching the notebook code-cell look."""
    lexer = get_lexer_by_name(lang)
    body = highlight(code.strip("\n"), lexer, HtmlFormatter(nowrap=True)).rstrip("\n")
    return (
        '<div class="thrml-codecard">'
        '<div class="thrml-code-head"><span class="thrml-lang">' + lang + "</span>"
        '<span class="thrml-tools"><button class="thrml-copy" type="button" title="Copy code" aria-label="Copy code"></button></span></div>'
        "<pre><code>" + body + "</code></pre>"
        "</div>"
    )


def write_doc_page(slug, title, inner, entries, active_page, *, mathjax=False, active_api=None):
    head = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title} &middot; THRML</title>\n"
        + og_meta(f"{title} · THRML", f"{slug}.html")
        + "\n"
        + css(COLLAPSE_STYLE)
        + "\n"
        + css(DOC_CSS)
        + (MATHJAX + "\n" if mathjax else "")
        + "</head>\n"
    )
    body = (
        '<body class="thrml-has-sidebar">'
        + build_topbar()
        + build_sidebar(entries, active_page=active_page, active_api=active_api)
        + f'<main class="thrml-doc">{inner}</main>'
        + "\n"
        + js(COLLAPSE_SCRIPT)
        + SPECULATION_RULES
        + "</body>\n</html>\n"
    )
    (OUT_DIR / f"{slug}.html").write_text(head + body, encoding="utf-8")


def examples_inner(entries):
    """A Browse Examples gallery: every notebook as a card, grouped by section."""
    by_num = {number: (title, href) for number, title, href in entries}
    n = len(entries)
    parts = [
        "<h1>Examples</h1>",
        f'<p class="lede">{n} runnable notebooks, from a first Ising chain to the full sampling '
        "stack and hardware-scale spin models. Each one runs end to end and is rendered here with "
        "its outputs.</p>",
    ]
    for name, blurb, nums in INDEX_SECTIONS:
        parts.append(f"<h2>{html_lib.escape(name, quote=False)}</h2>")
        if blurb:
            parts.append(f"<p>{html_lib.escape(blurb, quote=False)}</p>")
        parts.append('<div class="thrml-cards">')
        for number in nums:
            if number not in by_num:
                continue
            title, href = by_num[number]
            parts.append(
                f'<a class="thrml-card2" href="{html_lib.escape(href, quote=True)}">'
                f'<span class="c2t">{number} &middot; {html_lib.escape(title, quote=False)}</span>'
                f'<span class="c2b">{html_lib.escape(INDEX_BLURBS.get(number, ""), quote=False)}</span></a>'
            )
        parts.append("</div>")
    return "\n".join(parts)


def getting_started_inner(entries):
    by_num = {number: (title, href) for number, title, href in entries}
    intro_href = by_num.get("00", ("Getting Started", "#"))[1]
    all_href = by_num.get("01", ("All of THRML", "#"))[1]
    quick = (
        "import jax\n"
        "import jax.numpy as jnp\n"
        "from thrml import SpinNode, Block, SamplingSchedule, sample_states\n"
        "from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init\n"
        "\n"
        "# A 5-spin Ising chain: five nodes, four nearest-neighbour edges.\n"
        "nodes = [SpinNode() for _ in range(5)]\n"
        "edges = [(nodes[i], nodes[i + 1]) for i in range(4)]\n"
        "biases = jnp.zeros((5,))\n"
        "weights = jnp.ones((4,)) * 0.5\n"
        "beta = jnp.array(1.0)\n"
        "model = IsingEBM(nodes, edges, biases, weights, beta)\n"
        "\n"
        "# Two-colour the chain so each block updates independently under Gibbs.\n"
        "free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]\n"
        "program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])\n"
        "\n"
        "key = jax.random.key(0)\n"
        "k_init, k_samp = jax.random.split(key, 2)\n"
        "init_state = hinton_init(k_init, model, free_blocks, ())\n"
        "schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)\n"
        "\n"
        "samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])\n"
    )
    html = (
        "<h1>Getting started</h1>\n"
        '<p class="lede">THRML is a JAX library for building and sampling probabilistic graphical '
        "models, with a focus on efficient block Gibbs sampling and energy-based models. This page "
        "installs THRML and samples a first Ising chain.</p>\n"
        "<h2>Installation</h2>\n"
        "<p>THRML requires Python 3.10 or newer.</p>\n"
        + code_card("pip install thrml", "bash")
        + '<p>Or with <a href="https://docs.astral.sh/uv/">uv</a>:</p>\n'
        + code_card("uv pip install thrml", "bash")
        + "<p>For a local checkout:</p>\n"
        + code_card(
            "git clone https://github.com/extropic-ai/thrml\n" "cd thrml\n" "pip install -e .",
            "bash",
        )
        + "<h2>A first model</h2>\n"
        "<p>THRML samples graphical models by block Gibbs. Define the variables as nodes, wire their "
        "interactions into a model, divide the graph into blocks via graph-colouring so each block can "
        "update in parallel, then hand it all to a sampling program. Here is a small Ising chain end to "
        "end:</p>\n"
        + code_card(quick)
        + "<p>Each call to <code>sample_states</code> warms up the chain, then draws samples by "
        "alternating Gibbs updates over the two blocks. Because the even and odd spins never share an "
        "edge, every spin in a block resamples at once, which is exactly the structure Extropic's "
        "hardware is built to accelerate.</p>\n"
        '<div class="note"><span class="note-t">Note</span><p>A <code>Block</code> is an ordered '
        "collection of nodes of the same type, and it is the unit block Gibbs updates together. "
        "Graph-colouring so that no two neighbours land in the same block is what lets a whole "
        "block resample in parallel.</p></div>\n"
        "<h2>Where to go next</h2>\n"
        '<div class="thrml-cards">'
        f'<a class="thrml-card2" href="{intro_href}"><span class="c2t">Getting Started notebook &rarr;</span>'
        '<p class="c2b">What a probabilistic computer is, how the hardware samples, and a first model in THRML.</p></a>'
        f'<a class="thrml-card2" href="{all_href}"><span class="c2t">All of THRML &rarr;</span>'
        '<p class="c2b">The whole library end to end: nodes, blocks, factors, interaction groups, programs, and sampling.</p></a>'
        "</div>\n"
    )
    return linkify_api(html)


def _api_link(name, *, code=True):
    """Link a public symbol name to its API reference page (for hand-authored docs)."""
    inner = f"<code>{name}</code>" if code else name
    url = API_SYMBOL_URL.get(name)
    return f'<a class="api-xref" href="{url}">{inner}</a>' if url else inner


def concepts_inner(entries):
    L = _api_link
    return (
        "<h1>Concepts</h1>\n"
        '<p class="lede">THRML does block Gibbs sampling of graphical models at scale. From a '
        "user's perspective there are three things to work with: blocks, factors, and programs. "
        "This page is the mental model behind them.</p>\n"
        "<h2>Blocks</h2>\n"
        f"<p>Blocks are fundamental to THRML, because it implements block sampling. A {L('Block')} is "
        "a collection of nodes of the same type with an implicit ordering. Graph-colouring so "
        "that no two neighbours share a block is what lets every node in a block resample at once, "
        "which is the parallelism the hardware exploits.</p>\n"
        "<h2>Factors and interaction groups</h2>\n"
        f"<p>Factors and their conditionals are the backbone of sampling. {L('AbstractFactor')}s take "
        "their name from factor graphs and organize interactions between variables into a bipartite "
        f"graph of factors and variables. A factor synthesizes its interactions into {L('InteractionGroup')}s "
        "through a <code>to_interaction_groups()</code> method, which is the array-friendly form the "
        "sampler consumes.</p>\n"
        "<h2>Programs</h2>\n"
        f"<p>Programs are the orchestrating data structures. {L('BlockSamplingProgram')} handles the "
        "mapping and bookkeeping for padded block Gibbs sampling, managing the global state "
        f"representation efficiently for JAX. {L('FactorSamplingProgram')} is a convenient wrapper that "
        "converts factors to interaction groups. A program coordinates free and clamped blocks, "
        "samplers, and interactions to actually run the algorithm.</p>\n"
        "<h2>The global state</h2>\n"
        "<p>From a developer's perspective, the core approach is to represent as much as possible as "
        "contiguous arrays and pytrees, operate on those structures, then map to and from them for the "
        "user. Internally this is called the global state, in opposition to the block state. It is the "
        "same data-oriented idea as a struct-of-arrays layout, and it is similar to other JAX graphical "
        'model packages such as <a href="https://github.com/google-deepmind/PGMax">PGMax</a>. The '
        "distinction is that THRML supports pytree and heterogeneous states: nodes are split by their "
        "pytree type, and the global state is a list of those pytrees, stacked where several blocks "
        "share a type.</p>\n"
        "<p>Since JAX does not support ragged arrays, every block must be the same size in its array "
        "leaves. THRML solves this by stacking blocks of the same pytree type and padding them out as "
        "needed. There is a tradeoff between padding, which adds some runtime overhead, and looping over "
        "blocks, which would pay a likely untenable compile-time cost instead. Everything else in THRML "
        "exists to make building and running a program convenient; the focused core is block index "
        "management and padding, which keeps the codebase lightweight and hackable at around 1,000 lines."
        "</p>\n"
        '<figure class="thrml-fig"><img src="assets/flow.png" alt="Flow of components into the FactorSamplingProgram"></figure>\n'
        "<h2>Limitations</h2>\n"
        "<p>THRML is fast and efficient, but sampling itself is a genuinely hard problem. Drawing "
        "samples from a distribution in high-dimensional space can take prohibitively many steps even "
        "when proposals are parallelized. THRML is also focused on Gibbs sampling, since that is what "
        "Extropic's hardware accelerates, and for general problems it is not known when Gibbs is "
        "substantially faster or slower than other MCMC methods, so some problems will want other "
        "tools. As a small example, a two-node Ising model with a single edge at $J = -\\infty$, "
        "$h = 0$ never mixes between its ground states $\\{-1,-1\\}$ and $\\{1,1\\}$ under Gibbs, "
        "because it never flips once it reaches one of them, while a uniform Metropolis-Hastings move "
        "would converge quickly.</p>\n"
        "<h2>Factor and sampler hierarchies</h2>\n"
        "<p>THRML ships two parallel hierarchies, one for factors that define energy and one for the "
        "conditionals that sample them:</p>\n"
        "<p><strong>Factors</strong></p>\n"
        "<ul>\n"
        f"<li>{L('AbstractFactor')}\n"
        "<ul>\n"
        f"<li>{L('WeightedFactor')}: parameterized by weights</li>\n"
        f"<li>{L('EBMFactor')}: defines energy functions for energy-based models\n"
        "<ul>\n"
        f"<li>{L('DiscreteEBMFactor')}: EBMs with discrete states (spin and categorical)\n"
        "<ul>\n"
        f"<li>{L('SquareDiscreteEBMFactor')}: optimized for square interaction tensors\n"
        "<ul>\n"
        f"<li>{L('SpinEBMFactor')}: spin-only interactions ($\\{{-1, 1\\}}$ variables)</li>\n"
        f"<li>{L('SquareCategoricalEBMFactor')}: square categorical interactions</li>\n"
        "</ul></li>\n"
        f"<li>{L('CategoricalEBMFactor')}: categorical-only interactions</li>\n"
        "</ul></li>\n"
        "</ul></li>\n"
        "</ul></li>\n"
        "</ul>\n"
        "<p><strong>Samplers</strong></p>\n"
        "<ul>\n"
        f"<li>{L('AbstractConditionalSampler')}\n"
        "<ul>\n"
        f"<li>{L('AbstractParametricConditionalSampler')}\n"
        "<ul>\n"
        f"<li>{L('BernoulliConditional')}: spin-valued Bernoulli sampling\n"
        "<ul>\n"
        f"<li>{L('SpinGibbsConditional')}: Gibbs updates for spin variables in EBMs</li>\n"
        "</ul></li>\n"
        f"<li>{L('SoftmaxConditional')}: categorical softmax sampling\n"
        "<ul>\n"
        f"<li>{L('CategoricalGibbsConditional')}: Gibbs updates for categorical variables in EBMs</li>\n"
        "</ul></li>\n"
        "</ul></li>\n"
        "</ul></li>\n"
        "</ul>\n"
    )


# Copy-button icon, baked into the landing page only. The landing page is
# self-contained: it loads index.js (which wires its `.codecard .copy` button)
# and does not pull in chrome.js, so it cannot rely on chrome.js's runtime icon
# injection the way the sidebar doc/notebook code cards (`.thrml-copy`) do.
COPY_SVG = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="9" y="9" width="13" height="13" rx="2"/>'
    '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>'
)


def write_index(entries):
    by_num = {number: (title, href) for number, title, href in entries}

    def card(num):
        title, href = by_num[num]
        blurb = INDEX_BLURBS.get(num, "")
        return (
            f'      <a class="card" href="{html_lib.escape(href, quote=True)}">'
            f'<span class="card-num">{num}</span>'
            f'<span class="card-title">{html_lib.escape(title, quote=False)}</span>'
            f'<span class="card-blurb">{html_lib.escape(blurb, quote=False)}</span></a>'
        )

    featured_cards = "\n".join(card(n) for n in ("00", "01", "02") if n in by_num)
    browse_href = "examples.html"
    n_examples = len(entries)

    copy_btn = '<button class="copy" type="button" aria-label="Copy code">' + COPY_SVG + "</button>"
    quick_code = (
        "import jax, jax.numpy as jnp\n"
        "from thrml import SpinNode, Block, SamplingSchedule, sample_states\n"
        "from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init\n"
        "\n"
        "# A 5-spin Ising chain, two-coloured into parallel blocks\n"
        "nodes = [SpinNode() for _ in range(5)]\n"
        "edges = [(nodes[i], nodes[i + 1]) for i in range(4)]\n"
        "model = IsingEBM(\n"
        "    nodes, edges, jnp.zeros((5,)), jnp.ones((4,)) * 0.5, jnp.array(1.0))\n"
        "\n"
        "free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]\n"
        "program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])\n"
        "\n"
        "k_init, k_samp = jax.random.split(jax.random.key(0), 2)\n"
        "state = hinton_init(k_init, model, free_blocks, ())\n"
        "schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)\n"
        "samples = sample_states(\n"
        "    k_samp, program, schedule, state, [], [Block(nodes)])"
    )
    quick_highlighted = highlight(quick_code, get_lexer_by_name("python"), HtmlFormatter(nowrap=True)).rstrip("\n")
    first_model_card = (
        '<div class="codecard"><div class="chead"><span>python</span>' + copy_btn + "</div>"
        "<pre>" + quick_highlighted + "\n"
        '<span class="out"># samples: 1000 draws from the chain, by block Gibbs</span></pre></div>'
    )

    index_script = "\n" + js(INDEX_SCRIPT)

    head = (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>THRML &middot; Thermodynamic Hypergraphical Models</title>\n"
        '<meta name="description" content="THRML is a JAX library for block Gibbs sampling of probabilistic hypergraphical models and energy-based models, built to prototype on Extropic\'s thermodynamic sampling hardware.">\n'
        + og_meta("THRML · Thermodynamic Hypergraphical Models", "index.html")
        + css(INDEX_CSS)
        + "<style>@view-transition { navigation: auto; }</style>\n"
        + SPECULATION_RULES_INLINE
        + "</head>\n"
    )
    body = f"""<body>
  <header class="nav">
    <a class="brand" href="index.html">{LOGO_SVG}<span class="brand-name">THRML</span></a>
    <nav class="pills">
      <a class="pill" href="getting-started.html">Docs</a>
      <a class="pill" href="examples.html">Examples</a>
      <a class="pill" href="papers/codon-optimization/">Paper</a>
      <a class="pill" href="{INDEX_GITHUB}">GitHub</a>
    </nav>
  </header>

  <section class="hero">
    <div class="hero-text">
      <div class="hero-eyebrow"><a href="https://extropic.ai"><img src="assets/extropic_wordmark.png" alt="Extropic"></a></div>
      <h1>Thermodynamic hypergraphical models</h1>
      <p class="tagline">THRML is a JAX library for block Gibbs sampling of hypergraphical and
      energy-based models. Build a model from nodes and many-body factors, divide it into blocks via
      graph-colouring, and sample, the same structure Extropic's hardware is built to accelerate.</p>
      <div class="cta">
        <a class="pill solid" href="getting-started.html">Get started &rarr;</a>
        <a class="pill" href="{INDEX_GITHUB}">View on GitHub</a>
      </div>
    </div>
    <div class="hero-visual">
      <div class="hero-media">
        <video class="hero-video" autoplay loop muted playsinline aria-hidden="true">
          <source src="{ASSET_BASE}/videos/extropic.webm" type="video/webm">
          <source src="{ASSET_BASE}/videos/extropic.mp4" type="video/mp4">
        </video>
      </div>
    </div>
  </section>

  <div class="prims reveal">
    <div class="prim"><span class="name">Block</span><span class="dom">nodes</span><div class="desc">An ordered set of same-type nodes, resampled together in one parallel Gibbs sweep.</div></div>
    <div class="prim"><span class="name">Factor</span><span class="dom">energy</span><div class="desc">Organizes the interactions between variables, and the energy they contribute, into a factor graph.</div></div>
    <div class="prim"><span class="name">Program</span><span class="dom">sample</span><div class="desc">Coordinates blocks, samplers, and interactions to run block Gibbs efficiently on GPU.</div></div>
  </div>

  <section class="quick reveal">
    <h2 class="sec-title">Quickstart</h2>
    <p class="install">Install with <code>pip install thrml</code> (Python 3.10+), then build a model and sample it:</p>
    {first_model_card}
  </section>

  <section id="notebooks" class="nb-wrap reveal">
    <h2 class="sec-title">Example notebooks</h2>
    <p class="sec-sub">{n_examples} runnable notebooks, from a first Ising chain to the full sampling stack and hardware-scale spin models.</p>
    <div class="grid">
{featured_cards}
    </div>
    <a class="browse" href="{browse_href}">Browse all {n_examples} examples &rarr;</a>
    <div class="apps-links" style="margin-top:1rem">
      <a class="browse" href="https://github.com/pschilliOrange/dtm-replication">A larger project built on THRML: Denoising Thermodynamic Models &rarr;</a>
      <a class="browse" href="https://arxiv.org/abs/2510.23972">Read the DTM paper (arXiv) &rarr;</a>
    </div>
  </section>

  <section class="nb-wrap reveal" style="padding-top:3rem">
    <h2 class="sec-title">Applications</h2>
    <p class="sec-sub">What the sampling stack is for: real problems compiled to graphical models and sampled on thermodynamic hardware.</p>
    <div class="grid apps-grid">
      <a class="card" href="03_codon_optimization.html">
        <span class="card-num">Walkthrough</span>
        <span class="card-title">Codon optimization with THRML</span>
        <span class="card-blurb">A real design problem end to end: optimize a gene's codons by writing the objective as an energy function, building it as a Potts model and an equivalent Ising model, and sampling with simulated annealing. Validated hardware models project a sampling unit could solve it with roughly 10<sup>6</sup>&times; less energy than a GPU.</span>
      </a>
    </div>
    <div class="apps-links">
      <a class="browse" href="papers/codon-optimization/">Read the paper &rarr;</a>
      <a class="browse" href="https://github.com/extropic-ai/codon_opt">Reproduction code on GitHub &rarr;</a>
    </div>
  </section>

  <footer class="foot">
    <video class="foot-video" autoplay loop muted playsinline>
      <source src="{ASSET_BASE}/videos/extropic-footer.webm" type="video/webm">
      <source src="{ASSET_BASE}/videos/extropic-footer.mp4" type="video/mp4">
    </video>
    <div class="foot-inner">
      <span>THRML &middot; <a href="https://extropic.ai">EXTROPIC</a></span>
      <span><a href="{INDEX_GITHUB}">GitHub</a></span>
    </div>
  </footer>
{index_script}
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(head + body, encoding="utf-8")


def write_llms_txt(entries):
    """Write an llms.txt index of the docs, examples, and API for AI consumption."""
    out = [
        "# THRML",
        "",
        "> THRML is a JAX library for building and sampling probabilistic graphical models, with a "
        "focus on efficient block Gibbs sampling and energy-based models. A model is built from nodes "
        "and factors, divided into blocks via graph-colouring so each block resamples in parallel, and "
        "run by a sampling program. THRML supports sparse, heterogeneous graphs and pytree node states, "
        "and is a natural place to prototype today and experiment with future Extropic hardware.",
        "",
        "## Docs",
        f"- [Getting started]({SITE_URL}/getting-started.html): install THRML and sample a first Ising chain.",
        f"- [Concepts]({SITE_URL}/concepts.html): blocks, factors, programs, the global state, and the factor/sampler hierarchies.",
        "",
        "## Examples",
    ]
    for number, title, href in entries:
        out.append(f"- [{number} {title}]({SITE_URL}/{href}): {INDEX_BLURBS.get(number, '')}")
    out += ["", "## API reference"]
    for cat in API_CATEGORIES:
        mod = importlib.import_module(cat["module"])
        present = ", ".join(s for s in cat["symbols"] if hasattr(mod, s))
        out.append(f"- [{cat['label']}]({SITE_URL}/{cat['slug']}.html): {cat['blurb']} ({present})")
    (OUT_DIR / "llms.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
