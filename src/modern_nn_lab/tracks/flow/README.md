# Flow Matching

**Claim level: educational implementation.** Continuous flow matching built from first
principles on two-dimensional distributions. The track registry originally declared
`compact reproduction`; it was downgraded when the track was built, because no primary
source was read and there is therefore no paper experiment to be faithful to.

## The idea

Generation as vector-field regression. Pick a probability path that interpolates a source
sample towards a target sample, regress a network onto the path's velocity, then generate by
solving an ODE from the source:

```
x_t = α_t·x₁ + σ_t·x₀           the path
u_t = α̇_t·x₁ + σ̇_t·x₀           its conditional velocity — the regression target
dx/dt = v_θ(x, t)                sampling, from t=0 to t=1
```

The objective regresses onto a *conditional* velocity, yet its minimizer is the *marginal*
field. That is the identity flow matching rests on, and it is not obvious. This repository
does not prove it, but it checks its consequence: on the Gaussian case the learned field
converges to the analytic marginal field, which is known in closed form.

| File | What it holds |
| --- | --- |
| [paths.py](paths.py) | `ProbabilityPath` and the two affine paths |
| [analytic.py](analytic.py) | Closed-form marginal field for Gaussian endpoints |
| [field.py](field.py) | `VectorField` and the flow-matching loss |
| [solver.py](solver.py) | Euler and midpoint integration, with evaluation counting |
| [data.py](data.py) | The three targets, energy distance, mode coverage |

## Separating the two error sources

The acceptance criterion, and the reason `analytic.py` exists. Sample quality alone cannot
distinguish a badly learned field from a solver that took too few steps — a poor sample set
is equally consistent with both. With Gaussian endpoints the marginal velocity is available
in closed form, so each error can be measured with the other held at zero:

| Measurement | What is held fixed | What it isolates |
| --- | --- | --- |
| Integrate the **exact** field | model error is zero | ODE discretization error |
| Compare **learned** field to exact, pointwise | no solver involved | field approximation error |
| Integrate the **learned** field | nothing | both together |

The curved targets (`moons`, `mixture`) have no closed form and are reported on sample
quality alone. For those, the two error sources are **not** separated, and the report says so
rather than implying otherwise.

## The two paths

`linear` — α=t, σ=1−t, so the conditional velocity is `x₁ − x₀`, constant along each
trajectory. This is the optimal-transport-style path: each pair travels a straight line at
uniform speed.

`trigonometric` — α=sin(πt/2), σ=cos(πt/2), so α²+σ²=1. The variance-preserving path from
diffusion models: quarter-circle trajectories with varying speed. The variance claim is
exact only when *both* endpoints are standard normal; for a general target it is not, and no
claim is made that it is.

Straight trajectories at uniform speed are easier to integrate coarsely, so the linear path
should tolerate few solver steps better. That is a prediction the report tests rather than
assumes.

## What the tests check

Beyond the prompt's four mandatory properties, two things are checked more strongly than
required, because the weaker version would pass on broken code:

- **Solver convergence order is measured, not assumed.** Halving the step size must divide
  Euler's error by 2 and midpoint's by 4. A midpoint method that had silently degraded to
  first order would fail rather than merely underperform.
- **The closed-form field is verified two independent ways** — against the defining
  orthogonality property of a conditional expectation, and by integrating it to confirm it
  transports the source onto the target. A third test perturbs the field by 10% and asserts
  the orthogonality check *rejects* it, so the check cannot pass vacuously.

Note on the finite-difference test: it runs in float64. A central difference divides an O(1)
subtraction by 2e-6, which amplifies float32 round-off past the agreement being checked; the
property is mathematical, so the arithmetic should not be the limit.

## Running it

```bash
modern-nn run-track flow          # writes results/flow/
python scripts/report_flow.py     # regenerates every table in reports/flow.md
```

See [reports/flow.md](../../../../reports/flow.md) for results.
