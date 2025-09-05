import asyncio
import time
import pytest

from graphai import Graph, node


@node(start=True)
async def start(idx: int = 0, x: int = 0, delay: float = 0.0):
    return {"idx": idx, "x": x, "delay": delay}


@node
async def mid(idx: int = 0, x: int = 0, delay: float = 0.0):
    # Simulate work
    if delay and delay > 0:
        await asyncio.sleep(delay)
    return {"idx": idx, "x": x}


@node(end=True)
async def end(idx: int = 0, x: int = 0):
    return {"result": x * 2, "idx": idx}


def build_graph() -> Graph:
    g = Graph().add_node(start()).add_node(mid()).add_node(end())
    g.add_edge("start", "mid").add_edge("mid", "end")
    g.compile()
    return g


@pytest.mark.asyncio
async def test_execute_many_preserves_order():
    g = build_graph()

    inputs = [
        {"idx": 0, "x": 3},
        {"idx": 1, "x": 7},
        {"idx": 2, "x": -1},
        {"idx": 3, "x": 0},
    ]

    results = await g.execute_many(inputs, concurrency=2)

    assert [r["idx"] for r in results] == [i["idx"] for i in inputs]
    assert [r["result"] for r in results] == [6, 14, -2, 0]


@pytest.mark.asyncio
async def test_execute_many_runs_concurrently():
    """
    With 6 items, each sleeping ~0.15s, serial time ~= 6*0.15 = 0.90s.
    With concurrency=3 we expect about ceil(6/3)*0.15 ~ 0.30s (+ overhead).
    We assert the parallel runtime is significantly less than serial (~60%).
    """
    g = build_graph()

    delay = 0.15
    n = 6
    inputs = [{"idx": i, "x": i, "delay": delay} for i in range(n)]

    t0 = time.perf_counter()
    results = await g.execute_many(inputs, concurrency=3)
    parallel_time = time.perf_counter() - t0

    assert [r["idx"] for r in results] == list(range(n))
    assert [r["result"] for r in results] == [i * 2 for i in range(n)]

    serial_time = n * delay

    assert parallel_time <= serial_time * 0.6, (
        f"expected parallel run ({parallel_time:.3f}s) to be significantly "
        f"faster than serial (~{serial_time:.3f}s)"
    )
