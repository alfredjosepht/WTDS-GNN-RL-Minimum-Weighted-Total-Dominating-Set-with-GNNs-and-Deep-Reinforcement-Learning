"""Interactive graph views (vis-network, inlined from the pyvis package so the app works offline)."""
import html
import json
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np

from .theme import AMBER, GREY, INK, LINE, MUTED, TEAL, TEAL_SOFT

MAX_DRAW = 2000


@lru_cache(maxsize=1)
def _vis_assets():
    import pyvis
    lib = Path(pyvis.__file__).parent / "templates" / "lib" / "vis-9.1.2"
    return (lib / "vis-network.min.js").read_text(encoding="utf-8"), \
        (lib / "vis-network.css").read_text(encoding="utf-8")


def layout(lg):
    """Fixed coordinates for lattices (from their geometry) and small graphs; None = let physics decide."""
    g, sp = lg.graph, (lg.spec or {})
    fam = sp.get("family")
    try:
        if fam == "grid":
            r, c = sp["rows"], sp["cols"]
            return {i: (60 * (i % c), 60 * (i // c)) for i in range(g.n)}
        if fam in ("tri", "hex"):
            G = (nx.triangular_lattice_graph if fam == "tri" else nx.hexagonal_lattice_graph)(
                sp["rows"], sp["cols"], with_positions=True)
            nodes = sorted(G.nodes)
            pos = nx.get_node_attributes(G, "pos")
            return {i: (55 * pos[u][0], -55 * pos[u][1]) for i, u in enumerate(nodes)}
        if g.n <= 250:
            p = nx.kamada_kawai_layout(g.to_networkx())
            return {i: (520 * p[i][0], 520 * p[i][1]) for i in range(g.n)}
    except Exception:
        pass
    return None


def _shell(body, script, height):
    js, css = _vis_assets()
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,500&display=swap" rel="stylesheet">
<style>{css}
html,body{{margin:0;padding:0;background:#FFFFFF;font-family:Inter,system-ui,sans-serif;color:{INK};}}
#net{{width:100%;height:{height}px;border:1px solid {LINE};border-radius:10px;background:#FFFFFF;}}
div.vis-tooltip{{font-family:Inter,sans-serif;font-size:12.5px;background:#FFFFFF;border:1px solid {LINE};
 border-radius:8px;box-shadow:0 4px 14px rgba(31,35,40,.08);padding:8px 10px;color:{INK};white-space:normal;max-width:260px;}}
.mono{{font-family:'JetBrains Mono',monospace;}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:{MUTED};margin:8px 2px 0 2px;}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px;}}
</style></head><body>{body}<script>{js}</script><script>{script}</script></body></html>"""


def _w(x):
    x = float(x)
    return f"{x:.0f}" if x.is_integer() else f"{x:.2f}"


def solution_html(lg, S, height=560, physics=None):
    g = lg.graph
    Sset = set(int(v) for v in S)
    pos = layout(lg)
    physics = (pos is None) if physics is None else physics
    big = 3 if g.n <= 20 else (2 if g.n <= 60 else 1)        # small graphs are zoomed out to fit: scale up
    nodes = []
    for v in range(g.n):
        nb = g.adj[v]
        cov = [html.escape(str(lg.label(u))) for u in nb if u in Sset]
        tip = (f"<b>vertex {html.escape(str(lg.label(v)))}</b><br>weight <b>{_w(g.w[v])}</b><br>degree {len(nb)}<br>"
               f"{'<b style=color:' + TEAL + '>in S</b>' if v in Sset else 'not in S'}<br>"
               f"covered by: {', '.join(map(str, cov[:8]))}{' …' if len(cov) > 8 else ''}")
        inS = v in Sset
        node = {"id": v, "title": tip, "label": _w(g.w[v]) if g.n <= 150 else "",
                "color": {"background": TEAL if inS else GREY, "border": "#0B5F58" if inS else "#B8BDC4",
                          "highlight": {"background": AMBER, "border": INK}},
                "size": (15 if inS else 8) * big,
                "font": {"color": "#0B5F58" if inS else MUTED, "size": (11 * big) if g.n <= 150 else 0,
                         "face": "JetBrains Mono", "vadjust": -2}}
        if pos is not None:
            node["x"], node["y"] = pos[v]
        nodes.append(node)
    edges = [{"from": int(u), "to": int(v)} for u, v in g.edges().tolist()]
    opts = {"physics": {"enabled": bool(physics), "solver": "barnesHut",
                        "barnesHut": {"gravitationalConstant": -2600, "springLength": 70, "damping": 0.35},
                        "stabilization": {"iterations": 200 if g.n <= 500 else 80}},
            "interaction": {"hover": True, "tooltipDelay": 80, "navigationButtons": False, "zoomView": True,
                            "dragView": True},
            "edges": {"color": {"color": "#CFCAC0", "highlight": AMBER}, "width": 1, "smooth": False},
            "nodes": {"shape": "dot", "borderWidth": 1}}
    script = f"""
const N = {json.dumps(nodes)};
N.forEach(n => {{ const d = document.createElement('div'); d.innerHTML = n.title; n.title = d; }});  // escaped in Python
const nodes = new vis.DataSet(N);
const edges = new vis.DataSet({json.dumps(edges)});
const net = new vis.Network(document.getElementById('net'), {{nodes, edges}}, {json.dumps(opts)});
net.once('stabilizationIterationsDone', () => net.setOptions({{physics: {{enabled: {str(bool(physics) and g.n > 300).lower()}}}}}));
"""
    body = f"""<div id="net"></div><div class="legend"><span><i style="background:{TEAL}"></i>in the total dominating set S</span>
<span><i style="background:{GREY}"></i>other vertices</span><span>numbers = vertex weights</span><span>hover a vertex for details · scroll to zoom · drag to move</span></div>"""
    return _shell(body, script, height)


def replay_html(lg, trace, forced, rr_removed, final_S, height=520):
    """Step-by-step replay of the RL construction, then redundancy removal."""
    g = lg.graph
    pos = layout(lg)
    nodes = []
    for v in range(g.n):
        nd = {"id": v, "label": str(lg.label(v)) if g.n <= 120 else "", "size": 9,
              "font": {"size": 10 if g.n <= 120 else 0, "face": "JetBrains Mono", "color": MUTED}}
        if pos is not None:
            nd["x"], nd["y"] = pos[v]
        nodes.append(nd)
    edges = [{"from": int(u), "to": int(v)} for u, v in g.edges().tolist()]
    # coverage contributed by forced vertices
    fcov = sorted({int(u) for f in forced for u in g.adj[f]})
    labels = [html.escape(str(lg.label(v))) for v in range(g.n)]
    weights = [float(x) for x in g.w]
    data = {"n": g.n, "forced": [int(f) for f in forced], "fcov": fcov,
            "steps": [{"v": s["vertex"], "q": s["q"], "newly": s["newly_covered"], "top5": s.get("top5", []),
                       "cov": s.get("covered_count")} for s in trace],
            "removed": [int(v) for v in rr_removed], "final": [int(v) for v in final_S], "labels": labels,
            "w": weights, "big": 3 if g.n <= 20 else (2 if g.n <= 60 else 1)}
    opts = {"physics": {"enabled": pos is None, "stabilization": {"iterations": 150}},
            "interaction": {"hover": True, "dragView": True, "zoomView": True},
            "edges": {"color": {"color": "#D9D5CC"}, "width": 1, "smooth": False},
            "nodes": {"shape": "dot", "borderWidth": 1}}
    body = f"""
<div style="display:flex;gap:14px;align-items:stretch;">
 <div style="flex:1 1 auto;min-width:0;"><div id="net"></div></div>
 <div style="flex:0 0 230px;display:flex;flex-direction:column;gap:10px;">
  <div class="panel"><div class="ttl">Step</div><div id="stepno" class="big mono">0</div><div id="cap" class="cap"></div></div>
  <div class="panel"><div class="ttl">Cumulative weight W</div><div id="cumw" class="big mono" style="color:{TEAL}">0</div></div>
  <div class="panel"><div class="ttl">Vertices covered</div>
    <div class="bar"><div id="barfill"></div></div><div id="covtxt" class="cap mono"></div></div>
  <div class="panel"><div class="ttl">Top-5 Q-values</div><div id="top5" class="mono" style="font-size:12.5px;"></div>
    <div class="cap" style="margin-top:6px;">among legal actions (gain &gt; 0) at this step</div></div>
 </div>
</div>
<div class="ctrl">
 <button id="back">&#8592; Step</button><button id="play" class="primary">Play</button><button id="fwd">Step &#8594;</button>
 <input id="slider" type="range" min="0" value="0" style="flex:1;">
 <span id="phase" class="mono" style="font-size:12px;color:{MUTED};width:150px;text-align:right;"></span>
</div>
<div class="legend"><span><i style="background:{TEAL}"></i>picked</span><span><i style="background:#fff;border:3px solid {TEAL}"></i>just picked</span>
<span><i style="background:{AMBER}"></i>newly covered</span><span><i style="background:{TEAL_SOFT}"></i>covered earlier</span>
<span><i style="background:{GREY}"></i>not yet covered</span><span><i style="background:{TEAL};border:2px dashed #0B5F58"></i>forced (degree-1 rule)</span></div>
<style>
.panel{{border:1px solid {LINE};border-radius:10px;padding:10px 12px;background:#FFFFFF;}}
.ttl{{font:600 10.5px/1 'JetBrains Mono',monospace;letter-spacing:.1em;text-transform:uppercase;color:{MUTED};margin-bottom:6px;}}
.big{{font-size:26px;color:{INK};}} .cap{{font-size:12px;color:{MUTED};line-height:1.45;margin-top:4px;}}
.bar{{height:8px;background:#EFECE6;border-radius:99px;overflow:hidden;}} #barfill{{height:100%;width:0;background:{TEAL};transition:width .25s;}}
.ctrl{{display:flex;gap:8px;align-items:center;margin-top:10px;}}
.ctrl button{{font:500 13px Inter,sans-serif;border:1px solid {LINE};background:#fff;border-radius:8px;padding:7px 12px;cursor:pointer;color:{INK};}}
.ctrl button.primary{{background:{TEAL};border-color:{TEAL};color:#fff;min-width:70px;}}
input[type=range]{{accent-color:{TEAL};}}
.qrow{{display:flex;justify-content:space-between;padding:2px 0;}} .qrow.sel{{color:{TEAL};font-weight:500;}}
</style>"""
    script = f"""
const D = {json.dumps(data)};
const nodes = new vis.DataSet({json.dumps(nodes)});
const edges = new vis.DataSet({json.dumps(edges)});
const net = new vis.Network(document.getElementById('net'), {{nodes, edges}}, {json.dumps(opts)});
net.once('stabilizationIterationsDone', () => net.setOptions({{physics: {{enabled: false}}}}));
const T = D.steps.length, LAST = T + 1;          // states 0..T = construction, T+1 = after redundancy removal
const slider = document.getElementById('slider'); slider.max = LAST;
const forced = new Set(D.forced);
function render(t) {{
  const picked = new Set(D.forced), covered = new Set(D.fcov);
  for (let i = 0; i < Math.min(t, T); i++) {{ picked.add(D.steps[i].v); D.steps[i].newly.forEach(u => covered.add(u)); }}
  const cur = (t >= 1 && t <= T) ? D.steps[t-1] : null;
  const newly = new Set(cur ? cur.newly : []);
  const removed = new Set(t === LAST ? D.removed : []);
  const upd = [];
  for (let v = 0; v < D.n; v++) {{
    let bg = '{GREY}', bd = '#B8BDC4', bw = 1, size = 9, op = 1.0, dash = false, lab = D.n <= 120 ? D.labels[v] : '';
    if (covered.has(v)) {{ bg = '{TEAL_SOFT}'; bd = '#9CCFC9'; }}
    if (newly.has(v)) {{ bg = '{AMBER}'; bd = '#B45309'; size = 11; }}
    if (picked.has(v)) {{ bg = '{TEAL}'; bd = '#0B5F58'; size = 14; }}
    if (forced.has(v)) {{ dash = true; bw = 2; lab = 'forced'; }}
    if (cur && v === cur.v) {{ bg = '#FFFFFF'; bd = '{TEAL}'; bw = 5; size = 17; }}
    if (removed.has(v)) {{ op = 0.18; }}
    upd.push({{id: v, size: size * D.big, opacity: op, label: lab, borderWidth: bw, shapeProperties: {{borderDashes: dash ? [4,3] : false}},
      color: {{background: bg, border: bd}}, font: {{size: ((D.n <= 120 || lab === 'forced') ? 10 : 0) * D.big,
      color: picked.has(v) && !(cur && v===cur.v) ? '#FFFFFF' : '{MUTED}', face: 'JetBrains Mono'}} }});
  }}
  nodes.update(upd);
  const ncov = covered.size;
  let W = 0; (t === LAST ? D.final : [...picked]).forEach(v => W += D.w[v]);
  document.getElementById('cumw').textContent = (Math.round(W * 100) / 100).toString();
  document.getElementById('stepno').textContent = t === LAST ? 'RR' : String(t);
  document.getElementById('barfill').style.width = (100 * ncov / D.n).toFixed(1) + '%';
  document.getElementById('covtxt').textContent = ncov + ' / ' + D.n;
  let cap = '';
  if (t === 0) cap = D.forced.length ? D.forced.length + ' vertices forced by the degree-1 rule (dashed).' : 'No leaves: nothing is forced. Press Play.';
  else if (cur) cap = 'Picked vertex ' + D.labels[cur.v] + ' (weight ' + D.w[cur.v] + ', Q = ' + cur.q.toFixed(3) + '); ' + cur.newly.length + ' newly covered.';
  else cap = 'Redundancy removal dropped ' + D.removed.length + ' vertex(es) (most expensive first); final |S| = ' + D.final.length + '.';
  document.getElementById('cap').textContent = cap;
  document.getElementById('phase').textContent = t === LAST ? 'after redundancy removal' : (t === T ? 'all vertices covered' : 'construction');
  const tp = document.getElementById('top5');
  tp.innerHTML = cur ? cur.top5.map(([u, q]) => '<div class="qrow' + (u === cur.v ? ' sel' : '') + '"><span>v ' + D.labels[u] +
     ' · w ' + D.w[u] + '</span><span>' + q.toFixed(3) + '</span></div>').join('') : '<span style="color:{MUTED}">—</span>';
  slider.value = t;
}}
let t = 0, timer = null;
function stop() {{ if (timer) {{ clearInterval(timer); timer = null; }} document.getElementById('play').textContent = 'Play'; }}
document.getElementById('play').onclick = () => {{
  if (timer) {{ stop(); return; }}
  if (t >= LAST) t = 0;
  document.getElementById('play').textContent = 'Pause';
  timer = setInterval(() => {{ t += 1; render(t); if (t >= LAST) stop(); }}, Math.max(120, Math.min(700, 9000 / Math.max(T,1))));
}};
document.getElementById('fwd').onclick = () => {{ stop(); t = Math.min(LAST, t + 1); render(t); }};
document.getElementById('back').onclick = () => {{ stop(); t = Math.max(0, t - 1); render(t); }};
slider.oninput = () => {{ stop(); t = +slider.value; render(t); }};
render(0);
"""
    return _shell(body, script, height)
