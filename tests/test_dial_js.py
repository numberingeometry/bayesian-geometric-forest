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
