"""The browser dial (docs/dial/treelaw.js) computes the same law as the Python package."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bgforest import TreeLaw
from bgforest.ultrametric import single_linkage_ultrametric

JS = Path(__file__).resolve().parents[1] / "docs" / "dial" / "treelaw.js"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_solver_matches_python():
    X = np.random.default_rng(9).uniform(size=(30, 2))
    script = f"""
    const T = require({json.dumps(str(JS))});
    const P = {json.dumps(X.tolist())}, n = P.length, C = T.sqDists(P), out = {{}};
    out.U = Array.from(T.minimax(C, n, T.mst(C, n)));
    for (const lam of [0.5, 50, 5e3, 5e5]) {{
      const law = T.treeLaw(C, n, lam);
      out[lam] = {{P: Array.from(law.P), logTau: law.logTau, trace: law.traceError}};
    }}
    console.log(JSON.stringify(out));
    """
    res = json.loads(subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True).stdout)
    law0 = TreeLaw(X, 1.0)
    assert np.allclose(np.array(res["U"]).reshape(30, 30), single_linkage_ultrametric(law0.costs))
    for lam in (0.5, 50, 5e3, 5e5):
        key = str(int(lam)) if float(lam).is_integer() else str(lam)
        law = TreeLaw(X, lam)
        got = res[key]
        assert abs(got["logTau"] - law.log_partition()) <= 1e-12 * max(1.0, abs(law.log_partition()))
        assert np.abs(np.array(got["P"]).reshape(30, 30) - law.edge_probability()).max() < 1e-10
        assert got["trace"] < 1e-10


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_tree_draws_are_exact():
    X = np.random.default_rng(3).uniform(size=(6, 2))
    draws = 20000
    for lam in (2.0, 40.0):
        script = f"""
        const T = require({json.dumps(str(JS))});
        const P = {json.dumps(X.tolist())}, n = P.length, C = T.sqDists(P), lw = new Float64Array(n * n);
        for (let i = 0; i < n * n; i++) lw[i] = -{lam} * C[i];
        let s = 12345; const rand = () => (s = (s * 16807) % 2147483647) / 2147483647;
        const F = new Array(n * n).fill(0);
        for (let d = 0; d < {draws}; d++) for (const [u, v] of T.drawTree(lw, n, rand)) {{ F[u * n + v]++; F[v * n + u]++; }}
        console.log(JSON.stringify(F));
        """
        F = np.array(json.loads(subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True).stdout))
        F = F.reshape(6, 6) / draws
        P = TreeLaw(X, lam).edge_probability()
        se = np.sqrt(P * (1 - P) / draws) + 1e-12
        assert np.isclose(np.triu(F, 1).sum(), 5)
        assert np.max(np.abs(F - P) / se) < 4.5
