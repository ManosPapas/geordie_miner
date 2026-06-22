"""In-browser visualisation of an exported GEXF network.

Produces `network.html`: a self-contained page that embeds the GEXF inline and
renders it with sigma.js + graphology (loaded from a CDN). Per the spec, it
**fails gracefully** — if the third-party viewer libraries can't load (e.g.
offline) or the graph can't be rendered, the page still surfaces a download link
to the GEXF file so it is never inaccessible.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

# Inline only reasonably-sized graphs; for huge ones, link to the file instead.
_MAX_INLINE_BYTES = 5 * 1024 * 1024

_CDN = [
    "https://cdn.jsdelivr.net/npm/graphology@0.25.4/dist/graphology.umd.min.js",
    "https://cdn.jsdelivr.net/npm/graphology-gexf@0.13.2/dist/graphology-gexf.umd.min.js",
    "https://cdn.jsdelivr.net/npm/graphology-layout-forceatlas2@0.10.1/dist/graphology-layout-forceatlas2.umd.min.js",
    "https://cdn.jsdelivr.net/npm/sigma@2.4.0/dist/sigma.min.js",
]


def _scripts() -> str:
    return "\n".join(f'<script src="{u}"></script>' for u in _CDN)


def write_network_html(
    gexf_path: str,
    out_path: str,
    log: Callable[[str], None],
    title: str = "Co-occurrence network",
) -> Optional[str]:
    """Write `out_path` (network.html) for the GEXF at `gexf_path`. Returns the path or None."""
    if not os.path.exists(gexf_path):
        return None

    gexf_name = os.path.basename(gexf_path)
    size = os.path.getsize(gexf_path)
    download = (
        f'<p>Download the raw network: '
        f'<a href="{gexf_name}" download><code>{gexf_name}</code></a> '
        f'(open in <a href="https://gephi.org/">Gephi</a> or '
        f'<a href="https://www.vosviewer.com/">VOSviewer</a> for full exploration).</p>'
    )

    if size > _MAX_INLINE_BYTES:
        body = (
            f"<h1>{title}</h1>\n"
            f"<p class='muted'>The network is large ({size // 1024} KB) — the inline viewer is "
            f"skipped to keep this page light.</p>\n{download}"
        )
        _write(out_path, title, body, scripts="")
        log(f"  network viewer: {gexf_name} too large to embed; wrote download-only {os.path.basename(out_path)}.")
        return out_path

    with open(gexf_path, "r", encoding="utf-8", errors="replace") as f:
        gexf_text = f.read()

    viewer_js = """
(function () {
  function fallback(msg) {
    var v = document.getElementById('viewer');
    if (v) v.style.display = 'none';
    var fb = document.getElementById('fallback');
    if (fb) { fb.style.display = 'block'; var m = document.getElementById('fallback-msg'); if (m && msg) m.textContent = msg; }
  }
  try {
    if (typeof graphology === 'undefined' || typeof GraphologyGEXF === 'undefined' || typeof Sigma === 'undefined') {
      return fallback('Interactive viewer libraries could not be loaded (are you offline?). The GEXF file is still available to download below.');
    }
    var Graph = graphology.Graph || graphology;
    var gexf = document.getElementById('gexf-data').textContent;
    var graph = GraphologyGEXF.parse(Graph, gexf);
    graph.forEachNode(function (node, attr) {
      if (attr.x === undefined) graph.setNodeAttribute(node, 'x', Math.random());
      if (attr.y === undefined) graph.setNodeAttribute(node, 'y', Math.random());
      graph.setNodeAttribute(node, 'size', Math.max(2, Math.sqrt(graph.degree(node) || 1)));
      if (!attr.color) graph.setNodeAttribute(node, 'color', '#2563eb');
      if (!attr.label) graph.setNodeAttribute(node, 'label', String(node));
    });
    try {
      if (typeof graphologyLayoutForceAtlas2 !== 'undefined') {
        graphologyLayoutForceAtlas2.assign(graph, { iterations: 120 });
      }
    } catch (e) { /* random layout is fine as a fallback */ }
    new Sigma(graph, document.getElementById('viewer'));
  } catch (e) {
    fallback('Could not render the network (' + e + '). The GEXF file is still available to download below.');
  }
})();
"""

    body = (
        f"<h1>{title}</h1>\n"
        f"{download}\n"
        f'<div id="viewer" style="height:80vh;border:1px solid #e3e3e3;border-radius:4px;"></div>\n'
        f'<div id="fallback" style="display:none" class="callout">'
        f'<strong>Interactive viewer unavailable.</strong> '
        f'<span id="fallback-msg"></span></div>\n'
        f'<script type="application/xml" id="gexf-data">{gexf_text}</script>\n'
        f"<script>{viewer_js}</script>"
    )
    _write(out_path, title, body, scripts=_scripts())
    log(f"  network viewer: embedded {gexf_name} into {os.path.basename(out_path)} (with graceful fallback).")
    return out_path


def _write(out_path: str, title: str, body: str, scripts: str) -> None:
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
.muted {{ color: #666; }}
.callout {{ background: #f5f5f7; border-left: 3px solid #2563eb; padding: 0.6rem 1rem; border-radius: 0 4px 4px 0; }}
code {{ background: #f5f5f7; padding: 0.1rem 0.3rem; border-radius: 3px; }}
</style>
{scripts}
</head>
<body>
{body}
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
