# Circuit models

Circuit strings describe equivalent circuits and determine the exact order of the
numeric parameter vector passed to fitting and simulation functions.

## 🔌 Supported elements

For angular frequency \(\omega = 2\pi f\), `compute_impedance` implements the
following expressions and conventional units.

| Token | Parameters, in order | Implemented impedance | Typical units |
|---|---|---|---|
| `R` | \(R\) | \(R\) | \(\Omega\) |
| `C` | \(C\) | \(1/(j\omega C)\) | F |
| `L` | \(L\) | \(j\omega L\) | H |
| `W` | \(\sigma\) | \(\sigma(1-j)/\sqrt{\omega}\) | chosen to yield \(\Omega\) |
| `Q` | \(Q,n\) | \(1/[Q(j\omega)^n]\) | \(Q\): F s\(^{n-1}\); \(n\): dimensionless |
| `F` | \(F,\tau\) | \(F\tanh(\sqrt{j\omega\tau})/\sqrt{j\omega\tau}\) | \(F\): \(\Omega\); \(\tau\): s |
| `G` | \(G,\tau\) | \(G/\sqrt{1+j\omega\tau}\) | \(G\): \(\Omega\); \(\tau\): s |
| `H` | \(H,\tau,\phi\) | \(H/[\sqrt{1+j\omega\tau}\tanh(\phi\sqrt{1+j\omega\tau})]\) | \(H\): \(\Omega\); \(\tau\): s; \(\phi\): dimensionless |

Element numbers make labels unique in reports while using the same element equation.
For example, `R1` and `R2` are both resistors.

## 🧮 Operators

| Syntax | Meaning | Example |
|---|---|---|
| `+` | Series connection | `R1+C1` |
| `|` | Parallel connection | `(R1|C1)` |
| `(...)` | Grouping | `R1+(R2|Q1)` |

Parenthesize parallel branches to make each circuit structure clear. For example,
use `R1+(R2|Q1)` for a series resistor followed by a parallel `R2|Q1` response.

## 🔢 Parameter order

Parameters are consumed from left to right. Multi-parameter elements consume
adjacent values.

```python
from EISFitpython import circuit_main as circuits

circuit = "R1+(R2|Q1)+F1"
params = [
    5.0,       # R1
    20.0,      # R2
    1.0e-6,    # Q1 coefficient
    0.90,      # Q1 exponent n
    100.0,     # F1 coefficient
    0.01,      # F1 time parameter tau
]

Z = circuits.compute_impedance(params, circuit, [1.0, 10.0, 100.0])
```

For `"(R1|Q1)+(R2|Q2)+Q3"`, the order is:

```text
R1, Q1, n1, R2, Q2, n2, Q3, n3
```

## 🧪 Simulation versus fitting callables

`compute_impedance` and `Z_gen(..., return_type="complex")` return one complex
value per frequency. SciPy fitting expects a real vector, so `Z_curve_fit` and
`Z_gen(..., return_type="concatenated")` return:

```text
[real(Z[0]), ..., real(Z[N-1]), imag(Z[0]), ..., imag(Z[N-1])]
```

```python
import numpy as np
from EISFitpython import circuit_main as circuits

f = np.logspace(6, -1, 50)
model = circuits.Z_gen("R1+(R2|Q1)", return_type="complex")
Z = model(f, 5.0, 20.0, 1e-6, 0.9)
```
