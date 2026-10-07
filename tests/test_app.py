"""End-to-end tests of the weighted Streamlit app (headless, streamlit.testing)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "checkpoints" / "best.pt").exists(), reason="needs checkpoints/best.pt")

DRIVER = """
import sys
sys.path.insert(0, r"{app}"); sys.path.insert(0, r"{root}")
import streamlit as st
from components.theme import inject
from components.graph_input import generate, parse_text, example
from components.state import set_current
inject()
{setup}
from pages import {page}
{page}.render()
"""


def app_for(tmp_path, page, setup=""):
    from streamlit.testing.v1 import AppTest
    f = tmp_path / f"drv_{page}.py"
    f.write_text(DRIVER.format(app=ROOT / "app", root=ROOT, page=page, setup=setup), encoding="utf-8")
    return AppTest.from_file(str(f), default_timeout=300)


def text_of(at, cls):
    return " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.markdown if f'class="{cls}' in m.value)


def click(at, label):
    [b for b in at.button if b.label == label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_entry_app_loads():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=300).run()
    assert not at.exception


EXPECTED = {"Two triangles, expensive bridge": 4, "3×3 grid, expensive centre": 4, "Star with heavy centre": 11,
            "Triangle": 3, "Square": 2, "Path (forced)": 9, "Two triangles, unit weights": 2, "Hexagon": 4}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_hand_examples_show_valid_set_and_optimum(tmp_path, name):
    at = app_for(tmp_path, "solve", f'set_current(example({name!r}), ("ex", {name!r}))').run()
    click(at, "Find total dominating set")
    s = " ".join(text_of(at, "stat").split())
    # default method = GNN + RL certified by the exact solver: shows the proven minimum weight
    assert "Valid" in s and "Proven minimum yes" in s
    assert f"Total weight W(S) {EXPECTED[name]} " in s


def test_er_100(tmp_path):
    at = app_for(tmp_path, "solve", 'set_current(generate("er", 100, 6.0, 0), "er")').run()
    click(at, "Find total dominating set")
    assert "Valid" in text_of(at, "stat")


def test_grid_10_random_weights(tmp_path):
    at = app_for(tmp_path, "solve", 'set_current(generate("grid", 100, 0, 0, side=10, weights="uniform_int"), "g")').run()
    click(at, "Find total dominating set")
    s = " ".join(text_of(at, "stat").split())
    assert "Valid" in s and ("Proven minimum yes" in s or "Certified gap" in s)


def test_isolated_vertex_friendly_error(tmp_path):
    at = app_for(tmp_path, "solve", 'set_current(parse_text("0 1\\n1 2\\n7", "0 1"), "i")').run()
    assert not at.exception and "No total dominating set can exist" in text_of(at, "notice")


def test_negative_and_nan_weights_rejected():
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    from components.graph_input import GraphInputError, parse_text
    with pytest.raises(GraphInputError, match="vertex 2"):
        parse_text("1 2\n2 3", "1 4\n2 -3")
    with pytest.raises(GraphInputError):
        parse_text("1 2\n2 3", "1 nan")
    with pytest.raises(GraphInputError):
        parse_text("1 2\n2 3", "1 abc")


def test_messy_input_parsing():
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    from components.graph_input import GraphInputError, parse_text
    lg = parse_text("# c\na b\nb b\na b\nb,c\nthis line has too many\n\nc a", "a 2\nb 3")
    assert lg.graph.n == 3 and lg.graph.num_edges == 3 and lg.labels == ["a", "b", "c"]
    assert list(lg.graph.w) == [2, 3, 1]
    w = " ".join(lg.warnings)
    assert "self-loop" in w and "duplicate" in w and "malformed" in w and "relabelled" in w and "defaults to 1" in w
    with pytest.raises(GraphInputError):
        parse_text("   \n# only a comment\n", "")
    lg = parse_text("v x 2\nv y 3\ne x y\n", "")                  # combined format pasted
    assert list(lg.graph.w) == [2, 3]


def test_large_ba_no_drawing(tmp_path):
    at = app_for(tmp_path, "solve", 'set_current(generate("ba", 5000, 3, 0), "ba")').run()
    click(at, "Find total dominating set")
    assert "Valid" in text_of(at, "stat") and "drawing is skipped" in text_of(at, "notice")


@pytest.mark.parametrize("page,button", [("watch", "Record construction"), ("compare", "Run all methods"),
                                         ("results", None), ("how", None)])
def test_other_pages(tmp_path, page, button):
    at = app_for(tmp_path, page, 'set_current(example("Two triangles, expensive bridge"), "p")').run()
    assert not at.exception
    if button:
        click(at, button)
