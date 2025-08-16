import sys
import types
import pytest

from graphai import Graph
from graphai import node


# build a tiny graph
@node(start=True)
async def a(input: dict):
    return {"x": 1}


@node(end=True)
async def b(input: dict):
    return input


def test_visualize_raises_helpful_error():
    g = Graph()
    g.add_node(a())
    g.add_node(b())
    g.add_edge("a", "b")

    # simulate missing matplotlib
    sys.modules["matplotlib"] = types.ModuleType("matplotlib")  # stub pkg
    sys.modules["matplotlib.pyplot"] = None  # force import failure
    with pytest.raises(ImportError) as ei:
        g.visualize()
    assert "pip install matplotlib" in str(ei.value)


def test_visualize_saves_png(tmp_path):
    # skip if matplotlib actually isn't installed in CI
    try:
        import matplotlib.pyplot as _  # type: ignore # noqa: F401
    except Exception:
        pytest.skip("matplotlib not installed")
    g = Graph()
    g.add_node(a())
    g.add_node(b())
    g.add_edge("a", "b")
    out = tmp_path / "g.png"
    g.visualize(save_path=str(out))
    assert out.exists()
