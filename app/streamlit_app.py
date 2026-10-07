"""Total Domination with GNN + Reinforcement Learning — demo app.

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP.parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Total Domination · GNN + RL", page_icon=None, layout="wide",
                   initial_sidebar_state="collapsed")

from components.theme import header, inject  # noqa: E402
from pages import compare, how, results, solve, watch  # noqa: E402

inject()
header()
nav = st.navigation([st.Page(solve.render, title="Solve", url_path="solve", default=True),
                     st.Page(watch.render, title="Watch the agent", url_path="watch"),
                     st.Page(compare.render, title="Compare methods", url_path="compare"),
                     st.Page(results.render, title="Results", url_path="results"),
                     st.Page(how.render, title="How it works", url_path="how")],
                    position="top")
nav.run()
