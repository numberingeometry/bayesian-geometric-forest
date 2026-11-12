/* Tempered spanning trees, exact in the browser.
 *
 * Pr(T) is proportional to exp(-lam * sum_{e in T} c_e). kronLogResistance() is a
 * line-by-line port of bgforest.treelaw.kron_log_resistance: a Kron elimination with
 * positive-only updates in log space, then the backward resistance recursion
 *   R(k, j) = 1/d_k + sum_a p_a R(a, j) - p'Rp / 2.
 * tests/test_dial_js.py checks this file against the Python package.
 */
(function (root) {
  'use strict';
  var LOG_HALF = Math.log(0.5);

  function lae(a, b) {                      // log(exp(a) + exp(b))
    if (a === -Infinity) return b;
    if (b === -Infinity) return a;
    return a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b));
  }

  function lse(arr, off, len) {             // log sum exp of arr[off .. off+len)
    var m = -Infinity, i;
    for (i = 0; i < len; i++) if (arr[off + i] > m) m = arr[off + i];
    if (m === -Infinity) return -Infinity;
    var s = 0;
    for (i = 0; i < len; i++) s += Math.exp(arr[off + i] - m);
    return m + Math.log(s);
  }

  /* lw: Float64Array(n*n) of symmetric log weights (-Infinity = no edge).
   * Returns {logTau, logR} with logR a Float64Array(n*n), -Infinity on the diagonal. */
  function kronLogResistance(lwIn, n) {
    var lw = Float64Array.from(lwIn), logR = new Float64Array(n * n).fill(-Infinity);
    if (n <= 1) return { logTau: 0, logR: logR };
    var i, j, a, b, k;
    for (i = 0; i < n; i++) lw[i * n + i] = -Infinity;
    var rows = new Array(n - 1), piv = new Float64Array(n - 1);
    for (k = 0; k < n - 1; k++) {
      var m = n - k - 1, r = lw.slice(k * n + k + 1, k * n + n), d = lse(r, 0, m);
      if (!isFinite(d)) throw new Error('the graph is disconnected');
      rows[k] = r; piv[k] = d;
      for (a = 0; a < m; a++) {
        var ra = r[a] - d, base = (k + 1 + a) * n + k + 1;
        if (ra === -Infinity) continue;
        for (b = a + 1; b < m; b++) {
          var v = lae(lw[base + b], ra + r[b]);
          lw[base + b] = v;
          lw[(k + 1 + b) * n + k + 1 + a] = v;
        }
      }
    }
    var lq = new Float64Array(n), col = new Float64Array(n);
    for (k = n - 2; k >= 0; k--) {
      var mm = n - k - 1, rk = rows[k], dk = piv[k];
      var lp = new Float64Array(mm);
      for (a = 0; a < mm; a++) lp[a] = rk[a] - dk;
      for (j = 0; j < mm; j++) {             // log sum_a p_a R(a, j)
        var mx = -Infinity;
        for (a = 0; a < mm; a++) { var t = lp[a] + logR[(k + 1 + a) * n + k + 1 + j]; col[a] = t; if (t > mx) mx = t; }
        if (mx === -Infinity) { lq[j] = -Infinity; continue; }
        var s = 0;
        for (a = 0; a < mm; a++) s += Math.exp(col[a] - mx);
        lq[j] = mx + Math.log(s);
      }
      for (a = 0; a < mm; a++) col[a] = lp[a] + lq[a];
      var lpRp = lse(col, 0, mm);
      for (j = 0; j < mm; j++) {
        var head = lae(-dk, lq[j]);
        var ratio = Math.min(LOG_HALF + lpRp - head, -1e-300);
        var val = head + Math.log1p(-Math.exp(ratio));
        logR[k * n + k + 1 + j] = val;
        logR[(k + 1 + j) * n + k] = val;
      }
    }
    var logTau = 0;
    for (k = 0; k < n - 1; k++) logTau += piv[k];
    return { logTau: logTau, logR: logR };
  }

  function sqDists(P) {
    var n = P.length, C = new Float64Array(n * n);
    for (var i = 0; i < n; i++) for (var j = i + 1; j < n; j++) {
      var s = 0;
      for (var t = 0; t < P[i].length; t++) { var z = P[i][t] - P[j][t]; s += z * z; }
      C[i * n + j] = s; C[j * n + i] = s;
    }
    return C;
  }

  function mst(C, n) {                      // Prim, O(n^2); returns [[i, j], ...]
    var inT = new Uint8Array(n), best = new Float64Array(n).fill(Infinity), from = new Int32Array(n).fill(-1), out = [];
    best[0] = 0;
    for (var it = 0; it < n; it++) {
      var u = -1;
      for (var v = 0; v < n; v++) if (!inT[v] && (u < 0 || best[v] < best[u])) u = v;
      inT[u] = 1;
      if (from[u] >= 0) out.push([Math.min(u, from[u]), Math.max(u, from[u])]);
      for (v = 0; v < n; v++) if (!inT[v] && C[u * n + v] < best[v]) { best[v] = C[u * n + v]; from[v] = u; }
    }
    return out;
  }

  function minimax(C, n, edges) {           // single-linkage ultrametric of C
    var adj = [], U = new Float64Array(n * n), i;
    for (i = 0; i < n; i++) adj.push([]);
    edges.forEach(function (e) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); });
    for (var s = 0; s < n; s++) {
      var seen = new Uint8Array(n), stack = [s]; seen[s] = 1;
      while (stack.length) {
        var u = stack.pop();
        adj[u].forEach(function (v) {
          if (!seen[v]) { seen[v] = 1; U[s * n + v] = Math.max(U[s * n + u], C[u * n + v]); stack.push(v); }
        });
      }
    }
    return U;
  }

  /* Everything the dial shows at one temperature. */
  function treeLaw(C, n, lam) {
    var lw = new Float64Array(n * n);
    for (var i = 0; i < n * n; i++) lw[i] = -lam * C[i];
    var sol = kronLogResistance(lw, n), P = new Float64Array(n * n), trace = 0;
    for (i = 0; i < n; i++) for (var j = 0; j < n; j++) if (i !== j) {
      var p = Math.exp(lw[i * n + j] + sol.logR[i * n + j]);
      P[i * n + j] = p;
      if (i < j) trace += p;
    }
    return { logTau: sol.logTau, logR: sol.logR, P: P, traceError: Math.abs(trace - (n - 1)) / (n - 1) };
  }

  /* One exact draw from the tree law at any temperature: a port of
   * bgforest.treelaw.chain_rule_tree. Groups of contracted vertices are joined with their
   * exact probability (choosing the original edge in proportion to its weight) or have
   * every edge between them deleted. rand() returns uniforms in [0, 1).
   * Returns the n - 1 edges as [i, j] pairs. */
  function drawTree(lw0, n, rand) {
    var k = n, groups = [], i, j, L = Float64Array.from(lw0), alive = Float64Array.from(lw0), edges = [];
    for (i = 0; i < n; i++) { groups.push([i]); L[i * n + i] = -Infinity; alive[i * n + i] = -Infinity; }
    while (k > 1) {
      var sol = kronLogResistance(L, k), best = -Infinity, a = 0, b = 1;
      for (i = 0; i < k; i++) for (j = i + 1; j < k; j++) {
        var v = L[i * k + j] + sol.logR[i * k + j];
        if (v > best) { best = v; a = i; b = j; }
      }
      var A = groups[a], B = groups[b], x, y;
      if (rand() < Math.exp(Math.min(best, 0))) {
        var m = -Infinity, w = [], s = 0, t = 0;
        for (x = 0; x < A.length; x++) for (y = 0; y < B.length; y++) { var lv = alive[A[x] * n + B[y]]; w.push(lv); if (lv > m) m = lv; }
        for (t = 0; t < w.length; t++) { w[t] = Math.exp(w[t] - m); s += w[t]; }
        var u = rand() * s; t = 0;
        while (t < w.length - 1 && (u -= w[t]) > 0) t++;
        edges.push([A[Math.floor(t / B.length)], B[t % B.length]]);
        var K = k - 1, M = new Float64Array(K * K), idx = [];
        for (i = 0; i < k; i++) if (i !== b) idx.push(i);
        for (var r = 0; r < K; r++) for (var c = 0; c < K; c++) {
          var ii = idx[r], jj = idx[c], val = L[ii * k + jj];
          if (ii === a && jj !== a) val = lae(val, L[b * k + jj]);
          if (jj === a && ii !== a) val = lae(val, L[ii * k + b]);
          M[r * K + c] = ii === jj ? -Infinity : val;
        }
        groups[a] = A.concat(B); groups.splice(b, 1); L = M; k = K;
      } else {
        L[a * k + b] = -Infinity; L[b * k + a] = -Infinity;
        for (x = 0; x < A.length; x++) for (y = 0; y < B.length; y++) { alive[A[x] * n + B[y]] = -Infinity; alive[B[y] * n + A[x]] = -Infinity; }
      }
    }
    return edges;
  }

  var api = { kronLogResistance: kronLogResistance, treeLaw: treeLaw, drawTree: drawTree, sqDists: sqDists, mst: mst, minimax: minimax, lae: lae };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.TreeLawJS = api;
})(typeof self !== 'undefined' ? self : this);
