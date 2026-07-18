# `EISFitpython.circuit_main`

Equivalent-circuit evaluation and generation of functions compatible with
`scipy.optimize.curve_fit`.

```python
from EISFitpython import circuit_main as circuits
```

See [Circuit models](../guides/circuit-models.md) for the grammar, equations, and
parameter counts.

## `compute_impedance`

```python
compute_impedance(p, circuit_str, freqs)
```

Evaluate a circuit at one or more frequencies.

| Parameter | Type | Description |
|---|---|---|
| `p` | sequence of float | Element parameters in left-to-right circuit order. |
| `circuit_str` | `str` | Numbered circuit expression such as `"R1+(R2|Q1)"`. |
| `freqs` | array-like | Frequencies in hertz. Positive values are expected. |

Returns a complex NumPy array with the broadcast shape of `freqs`. A scalar
frequency can produce a scalar-like complex value.

```python
Z = circuits.compute_impedance(
    [5.0, 20.0, 1e-6, 0.9],
    "R1+(R2|Q1)",
    [1.0, 10.0, 100.0],
)
```

## `Z_curve_fit`

```python
Z_curve_fit(circuit_str)
```

Create a closure with signature `model(f, *params)` for SciPy fitting. The closure
evaluates the circuit and returns a real vector of length `2 * len(f)` containing
all real components followed by all imaginary components.

| Parameter | Type | Description |
|---|---|---|
| `circuit_str` | `str` | Circuit expression captured by the returned callable. |

Returns a callable that evaluates the selected circuit.

```python
model = circuits.Z_curve_fit("R1|C1")
y = model(f, 100.0, 1e-6)
```

Parenthesize the parallel group in normal use: `"(R1|C1)"`.

## `Z_gen`

```python
Z_gen(circuit_string, return_type="concatenated")
```

Create a fitting or prediction closure.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `circuit_string` | `str` | required | Circuit expression captured by the closure. |
| `return_type` | `str` | `"concatenated"` | `"concatenated"` for a real fit vector or `"complex"` for complex impedance. |

Returns a callable with signature `model(f, *params)`.

```python
complex_model = circuits.Z_gen("R1+(R2|Q1)", return_type="complex")
Z = complex_model(f, 5.0, 20.0, 1e-6, 0.9)
```
