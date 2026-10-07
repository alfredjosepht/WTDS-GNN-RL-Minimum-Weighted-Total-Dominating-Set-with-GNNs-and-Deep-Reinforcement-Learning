"""Visual system: palette, injected CSS, header bar, cards and stat tiles."""
import html

import streamlit as st

TEAL = "#0F766E"
TEAL_SOFT = "#CCE7E4"
AMBER = "#D97706"
AMBER_SOFT = "#FCE7C8"
GREY = "#D1D5DB"
INK = "#1F2328"
MUTED = "#6B6F76"
PAPER = "#F7F5F0"
LINE = "#E4E0D8"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, label, button, input, textarea {{
  font-family: 'Inter', system-ui, sans-serif;
}}
.stApp {{ background: {PAPER}; }}
/* hide Streamlit chrome */
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stToolbarActions"], [data-testid="stDecoration"],
.stDeployButton, [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: {PAPER}; border-bottom: 1px solid {LINE}; }}
.block-container {{ padding-top: 4.2rem; padding-bottom: 4rem; max-width: 1320px; }}

h1, h2, h3, h4 {{ font-family: 'Fraunces', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 600 !important; }}
h2, h3 {{ font-weight: 500 !important; }}
p, li {{ color: {INK}; line-height: 1.6; }}
code, .mono {{ font-family: 'JetBrains Mono', ui-monospace, monospace !important; }}

/* header bar */
.lab-header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:24px;
  padding: 6px 0 18px 0; margin-bottom: 18px; border-bottom: 1px solid {LINE}; }}
.lab-header .eyebrow {{ font: 500 11px/1 'JetBrains Mono', monospace; letter-spacing:.14em; text-transform:uppercase;
  color:{TEAL}; margin-bottom:10px; }}
.lab-header h1 {{ font-size: 2.05rem; margin:0; padding:0; line-height:1.15; }}
.badge {{ display:inline-block; vertical-align:middle; font: 600 11px/1 'JetBrains Mono', monospace; letter-spacing:.1em;
  text-transform:uppercase; color:{AMBER}; border:1px solid {AMBER_SOFT}; background:#FFF8EC; border-radius:999px;
  padding:6px 9px; margin-left:8px; position:relative; top:-4px; }}
.stat.hero {{ grid-column: span 2; }}
.stat.hero .v {{ font-size: 40px; }}
.lab-header .sub {{ color:{MUTED}; margin-top:6px; font-size:0.98rem; max-width: 760px; }}
.lab-header .meta {{ font: 400 12px/1.5 'JetBrains Mono', monospace; color:{MUTED}; text-align:right; white-space:nowrap; }}

/* cards: st.container(border=True) */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: #FFFFFF; border: 1px solid {LINE} !important; border-radius: 10px !important;
  box-shadow: 0 1px 2px rgba(31,35,40,.04), 0 4px 14px rgba(31,35,40,.035);
}}
.card-title {{ font: 600 12px/1 'JetBrains Mono', monospace; letter-spacing:.12em; text-transform:uppercase;
  color:{MUTED}; margin: 2px 0 12px 0; }}
.section-title {{ font-family:'Fraunces', serif; font-size:1.35rem; color:{INK}; margin: 8px 0 4px 0; }}
.caption {{ color:{MUTED}; font-size:0.9rem; line-height:1.5; }}

/* stat tiles */
.stats {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap:12px; margin: 4px 0 14px 0; }}
.stat {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; padding:14px 16px 12px 16px; }}
.stat .k {{ font: 500 11px/1 'JetBrains Mono', monospace; letter-spacing:.1em; text-transform:uppercase; color:{MUTED}; }}
.stat .v {{ font: 500 30px/1.15 'JetBrains Mono', monospace; color:{INK}; margin-top:10px; }}
.stat .v.teal {{ color:{TEAL}; }}
.stat .v.amber {{ color:{AMBER}; }}
.stat .d {{ font-size:12.5px; color:{MUTED}; margin-top:4px; }}

/* notices */
.notice {{ border:1px solid {LINE}; border-left:3px solid {TEAL}; background:#FFFFFF; border-radius:8px;
  padding:12px 16px; margin: 6px 0 12px 0; font-size:0.95rem; }}
.notice.warn {{ border-left-color:{AMBER}; background:#FFFBF4; }}
.notice.err {{ border-left-color:#B42318; background:#FFF8F7; }}
.notice b {{ font-weight:600; }}
.pill {{ display:inline-block; font: 500 11.5px/1 'JetBrains Mono', monospace; padding:5px 8px; border-radius:999px;
  border:1px solid {LINE}; color:{MUTED}; background:{PAPER}; margin-right:6px; }}
.pill.teal {{ color:{TEAL}; border-color:{TEAL_SOFT}; background:#F0F8F7; }}

/* buttons */
.stButton > button, .stDownloadButton > button {{ border-radius:8px; font-weight:500; border:1px solid {LINE}; }}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {{ background:{TEAL}; border-color:{TEAL}; }}
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primary"] * {{ color:#FFFFFF !important; }}
/* top navigation */
header [data-testid="stToolbar"] a {{ font-family:'Inter',sans-serif; font-weight:500; color:{MUTED}; }}
header [data-testid="stToolbar"] a[aria-current="page"], header [data-testid="stToolbar"] a:hover {{ color:{TEAL}; }}
.stButton > button[kind="primary"]:hover {{ background:#0B5F58; border-color:#0B5F58; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 18px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 6px 0; font-weight:500; }}

/* tables */
.lab-table {{ width:100%; border-collapse:collapse; font-size:0.92rem; }}
.lab-table th {{ text-align:left; font: 600 11px/1.2 'JetBrains Mono', monospace; letter-spacing:.08em;
  text-transform:uppercase; color:{MUTED}; border-bottom:1px solid {LINE}; padding:8px 10px; }}
.lab-table td {{ padding:8px 10px; border-bottom:1px solid #EFECE6; color:{INK}; }}
.lab-table td.num {{ font-family:'JetBrains Mono', monospace; text-align:right; white-space:nowrap; }}
.lab-table tr.hl td {{ background:#F0F8F7; }}

/* pipeline diagram */
.pipe {{ display:flex; flex-wrap:wrap; align-items:stretch; gap:6px; margin: 8px 0 6px 0; }}
.pipe .step {{ flex:1 1 105px; min-width:105px; background:#FFFFFF; border:1px solid {LINE}; border-radius:10px; padding:12px 14px; }}
.pipe .step .n {{ font: 500 11px/1 'JetBrains Mono', monospace; color:{TEAL}; letter-spacing:.1em; }}
.pipe .step .t {{ font-family:'Fraunces', serif; font-size:1.02rem; margin-top:6px; color:{INK}; }}
.pipe .step .s {{ font-size:0.83rem; color:{MUTED}; margin-top:4px; line-height:1.45; }}
.pipe .arrow {{ align-self:center; color:{MUTED}; font-family:'JetBrains Mono', monospace; }}
.pipe .step.loop {{ border-style:dashed; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def header():
    st.markdown(
        """<div class="lab-header"><div>
        <div class="eyebrow">Research notebook · minimum weighted total dominating set</div>
        <h1>Weighted Total Domination with GNN + Reinforcement Learning <span class="badge">Weighted</span></h1>
        <div class="sub">Every vertex carries a weight. The agent picks vertices until every vertex has a chosen
        neighbour, minimising the total weight W(S); every answer is checked before it is shown.</div></div>
        <div class="meta">MPNN · 3 layers · Double DQN<br>reward −w(v) / w̄ · verified TDS</div></div>""",
        unsafe_allow_html=True)


def card_title(text):
    st.markdown(f'<div class="card-title">{html.escape(text)}</div>', unsafe_allow_html=True)


def stat_tiles(items):
    """items: list of (key, value, detail, tone); tone in {'', 'teal', 'amber', 'hero'} ('hero' = large teal card)."""
    cells = "".join(
        f'<div class="stat{" hero" if tone == "hero" else ""}"><div class="k">{html.escape(k)}</div><div class="v {"teal" if tone == "hero" else tone}">{html.escape(str(v))}</div>'
        f'<div class="d">{html.escape(d or "")}</div></div>' for k, v, d, tone in items)
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


def notice(text_html, tone=""):
    st.markdown(f'<div class="notice {tone}">{text_html}</div>', unsafe_allow_html=True)


def table(rows, columns, numeric=(), highlight=None):
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = []
    for r in rows:
        cls = ' class="hl"' if highlight and highlight(r) else ""
        tds = "".join(
            f'<td class="num">{html.escape(str(r.get(c, "")))}</td>' if c in numeric
            else f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in columns)
        body.append(f"<tr{cls}>{tds}</tr>")
    st.markdown(f'<table class="lab-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>',
                unsafe_allow_html=True)
