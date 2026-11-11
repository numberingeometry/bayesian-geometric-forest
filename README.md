# Forest Lab

Which tree grew this point cloud? Exact answers, from random trees to single linkage.

[![Edge probabilities of the tree law as the temperature falls: a fog of possible trees settles into the minimum spanning tree](docs/assets/dial.gif)](https://numberingeometry.github.io/bayesian-geometric-forest/dial/)

Grow a cloud by Gaussian diffusion along an unknown tree, and the posterior over trees
has one dial: the step size. Hot, any tree will do. Cold, only the minimum spanning tree
survives, and its cuts are single linkage. Every edge probability is exact at every
temperature.

**[Turn the dial](https://numberingeometry.github.io/bayesian-geometric-forest/dial/)** ·
[Explore forests](https://numberingeometry.github.io/bayesian-geometric-forest/explorer/) ·
[Mathematics](docs/temperature.md) · [Benchmarks](docs/benchmarks.md)

## Use the package

```sh
python -m pip install "git+https://github.com/numberingeometry/bayesian-geometric-forest"
```

```python
from bgforest import TreeLaw

law = TreeLaw(X, lam=50.0)   # Pr(T) proportional to exp(-lam * sum of squared edge lengths)
law.edge_probability()       # Pr(edge in tree), exact at any temperature
law.free_energy_grad()       # gradient of a soft minimum spanning tree
```

`bgforest` also implements the Bayesian spanning forest of Duan & Roy, where clusters
are trees, with Wilson sampling and reproducible reference experiments. Every identity
is checked against brute-force enumeration on each push.

[Quick start](https://numberingeometry.github.io/bayesian-geometric-forest/api/) ·
[Verification](docs/verification.md) · [MIT license](LICENSE)
