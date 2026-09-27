# B21 · micrograd III: The Backward Pass

**Meetings:** D38–D39 · **Points:** 15 pts

**Watch — 34 min**
Day 1 — 17 min
[Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=5225s) · segment 01:27:05 → 01:43:55, "breaking up tanh, more ops" and "the same thing in PyTorch" from 01:39:31 (17m)
 · `micrograd/engine.py` and `scratch/b21-video.py` open, terminal in the repo root. PyTorch has been installed since B17; today is the first time you import it.

Day 2 — 17 min
[Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=6235s) · segment 01:43:55 → 02:01:12, "MLP library", "tiny dataset, loss function", "collecting parameters" (17m)
 · A new file, `micrograd/nn.py`, open next to the engine.

**During the video**

**Type everything he types, and run it.** Engine changes go into `micrograd/engine.py`. The expressions he builds and the PyTorch cell go into `scratch/b21-video.py`, which starts with the same `sys.path` lines as B19's. His `Neuron`, `Layer` and `MLP` go into `micrograd/nn.py`, starting with `import random` and `from engine import Value`.

**Day 1 · 01:27:05 → 01:39:31.** He adds the operations a real expression needs: numbers on the left of `+` and `*`, `exp`, powers, division, negation and subtraction, each with its own `_backward`. Then he rebuilds `tanh` out of `exp` and division instead of as one node. **Before he runs backward on the rebuilt version, write in your scratch file, as a comment, whether the input's gradient will be the same as with the one-node `tanh`**, and why. Then run it.

**Day 1 · 01:39:31 → 01:43:55.** He builds the same neuron in PyTorch and prints its gradients. Type it exactly, including the `.double()`, and run it. Put his micrograd numbers and his PyTorch numbers on adjacent lines of output and read them digit by digit. Step 4 does this on your neuron.

**Day 2 · 01:43:55 → 02:01:12.** He builds a `Neuron`, a `Layer` of neurons and an `MLP` of layers, runs a four-example dataset through it, writes a loss, and collects every parameter into one list. **When he builds `MLP(3, [4, 4, 1])`, stop and work out on paper how many parameters it has, weights and biases, before he prints the count.** Then run it. You should get the same number he does.

**Notes**

**This week is lined up with your AP Calculus chain-rule unit on purpose.** When your Calc teacher writes `dy/dx = dy/du · du/dx`, that is literally what `self.grad += local_derivative * out.grad` does. `out.grad` is `dy/du`; the local derivative is `du/dx`.

**`+=` not `=`.** You fixed it in B20. Today you add six new `_backward` closures and every one of them needs it. One `=` in `__pow__` gives correct answers on every test that does not reuse a variable, and wrong ones on the loss in B22.

`1 + Value(2.0)` fails with `TypeError: unsupported operand type(s) for +: 'int' and 'Value'`. Python tries `int.__add__` first, which has never heard of `Value`, and then looks for `Value.__radd__`. That is the fix he types. It matters more than it looks: `sum(list_of_values)` starts from the integer 0, so every loss written with `sum` needs `__radd__`.

`__pow__` takes a plain number as the exponent, not a `Value`. His `assert isinstance(other, (int, float))` is there so that `x ** y` with `y` a `Value` fails loudly instead of computing a gradient for only half the expression.

**PyTorch defaults to 32-bit floats; micrograd uses Python's 64-bit floats.** In 32 bits, two correct engines agree to about 7 significant digits and sometimes miss the sixth decimal. That is why he writes `.double()`, and why every comparison you make to 6 decimals uses 64-bit tensors.

`tanh` built from `exp` and division is a longer graph for the same function, so the gradients agree, to about 15 decimals rather than exactly.

**Walkthrough — Finish the engine, and check it against PyTorch**

**Day 1 starts here.**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:
```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/backward
```
*If B20 is not approved yet:* `git switch dev/topo && git switch -c dev/backward`.
```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-track, not main
bash scripts/start-entry.sh
```
Under the timestamp, write the checklist for the whole assignment, and tick what you finish each day:
```markdown
- [ ] Day 1: Karpathy micrograd 01:27:05 → 01:43:55, new ops into engine.py, PyTorch cell into scratch
- [ ] Day 1: Steps 2–4, check the new ops, tanh two ways on my neuron, my neuron against PyTorch
- [ ] Day 2: Karpathy micrograd 01:43:55 → 02:01:12, Neuron/Layer/MLP into nn.py, parameter count on paper first
- [ ] Day 2: Steps 5–6, check nn.py, five tests against hand-computed derivatives
- [ ] Day 2: Extension, how small h should be
- [ ] Push each day; one PR
```

Watch the Day 1 segment and type along before Step 2.

**Step 2. Check the new operations.**

At 01:39:31 your `Value` class should have, in addition to what it had at the end of B20:

```python
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers"
        out = Value(self.data ** other, (self,), f"**{other}")
        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _backward():
            self.grad += out.data * out.grad       # d/dx e^x is e^x, which is out.data
        out._backward = _backward
        return out

    def __rmul__(self, other):        # other * self
        return self * other

    def __radd__(self, other):        # other + self
        return self + other

    def __truediv__(self, other):     # self / other
        return self * other ** -1

    def __neg__(self):                # -self
        return self * -1

    def __sub__(self, other):         # self - other
        return self + (-other)
```

If he did not type `__radd__` in this segment, add it anyway; the Notes say why. Check them all at once:

```bash
uv run python -c "import sys; sys.path.insert(0, 'micrograd'); from engine import Value; a = Value(3.0); b = Value(4.0); print(1 + a, 2 * a, a - b, a / b, a ** 2, (-a).data, sum([a, b]))"
```

*You should see* `Value(data=4.0) Value(data=6.0) Value(data=-1.0) Value(data=0.75) Value(data=9.0) -3.0 Value(data=7.0)`. A `TypeError` names the method you are missing.

**Step 3. `tanh` two ways, on your neuron.**

Create `micrograd/tanh_two_ways.py`:

```python
from engine import Value
from data import load

xs, ys, windows = load()
f = xs[0]

def neuron_n():
    w = [Value(0.5), Value(-0.3), Value(0.8)]
    b = Value(0.1)
    return (Value(f[0]) * w[0] + Value(f[1]) * w[1] + Value(f[2]) * w[2] + b), w, b

n1, w_one, b_one = neuron_n()
o1 = n1.tanh()                                   # one node
n2, w_exp, b_exp = neuron_n()
e = (2 * n2).exp()
o2 = (e - 1) / (e + 1)                           # the same function, built from pieces
o1.backward(); o2.backward()

print(f"o    one node {o1.data: .15f}   from exp {o2.data: .15f}")
for i in range(3):
    print(f"dw{i + 1}  one node {w_one[i].grad: .15f}   from exp {w_exp[i].grad: .15f}")
print(f"db   one node {b_one.grad: .15f}   from exp {b_exp.grad: .15f}")
```

Run it: `uv run python micrograd/tanh_two_ways.py`.

*You should see* each pair agree to about fifteen decimals, perhaps differing in the last digit or two. The weights are the same as your B20 neuron's, so `o` should also match `neuron_b20.py`'s to the precision it printed there, allowing for B20 having rounded the features.

**Step 4. Your neuron against PyTorch.**

This is the 01:39:31 segment on your data. Create `micrograd/vs_torch.py`:

```python
import torch
from engine import Value
from data import load

xs, ys, windows = load()
f = xs[0]
w0, b0 = [0.5, -0.3, 0.8], 0.1

x = [Value(v) for v in f]
w = [Value(v) for v in w0]
b = Value(b0)
o = (x[0] * w[0] + x[1] * w[1] + x[2] * w[2] + b).tanh()
o.backward()

xt = torch.tensor(f, dtype=torch.double, requires_grad=True)
wt = torch.tensor(w0, dtype=torch.double, requires_grad=True)
bt = torch.tensor(b0, dtype=torch.double, requires_grad=True)
ot = torch.tanh((xt * wt).sum() + bt)
ot.backward()

rows = [("o", o.data, ot.item()), ("do/db", b.grad, bt.grad.item())]
rows += [(f"do/dw{i + 1}", w[i].grad, wt.grad[i].item()) for i in range(3)]
rows += [(f"do/dx{i + 1}", x[i].grad, xt.grad[i].item()) for i in range(3)]
for name, mine, theirs in rows:
    print(f"{name:7} micrograd {mine: .6f}   torch {theirs: .6f}   within 5e-7: {abs(mine - theirs) < 5e-7}")
```

*You should see* eight rows, all `True`. PyTorch never saw your `Value` class. Two independent engines agreeing on your numbers to six decimals is the reason to trust either one.

*If it broke:* `False` in one row and `True` everywhere else is a single `_backward` with a wrong local derivative; the row names the variable, and the operation it passes through tells you which closure to read. `False` everywhere at the fifth or sixth decimal means you dropped `dtype=torch.double`.

Commit and push: `git add micrograd/ scratch/ && git commit -m "B21: full op set, tanh two ways, my neuron matches PyTorch"`, then `git push -u origin dev/backward`. Open the pull request now with **jd12** as reviewer. Sign off the log.

**Day 2 starts here.** Open a new entry with `bash scripts/start-entry.sh` in the log repo and copy over what is left of the checklist. Watch the Day 2 segment first, parameter count on paper before he prints it.

**Step 5. Check `nn.py`.**

At 02:01:12, `micrograd/nn.py` should be this, in substance:

```python
import random
from engine import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

Check it: `uv run python -c "import sys; sys.path.insert(0, 'micrograd'); from nn import MLP; print(len(MLP(3, [4, 4, 1]).parameters()))"`.

*You should see* `41`, the number you worked out on paper during the video. If your paper said something else, write the arithmetic out again in your log, weights and biases per layer, and find the layer you miscounted.

**Step 6. Five tests against derivatives you did by hand.**

Each test below states a derivative you can do on paper in a line. Do each one on paper first; the assertion is the paper's answer. Create `micrograd/test_engine.py`:

```python
import math
import pytest
from engine import Value

def test_karpathy_expression():
    a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
    L = (a * b + c) * f
    L.backward()
    assert (a.grad, b.grad, c.grad, f.grad) == (6.0, -4.0, -2.0, 4.0)

def test_reuse():
    a, b = Value(2.0), Value(-3.0)
    d = a * b + a                        # a arrives by two paths
    d.backward()
    assert a.grad == pytest.approx(b.data + 1)
    assert b.grad == pytest.approx(a.data)

def test_tanh_one_node_equals_exp_form():
    n1, n2 = Value(0.7), Value(0.7)
    o1 = n1.tanh()
    e = (2 * n2).exp()
    o2 = (e - 1) / (e + 1)
    o1.backward(); o2.backward()
    assert o1.data == pytest.approx(o2.data)
    assert n1.grad == pytest.approx(1 - math.tanh(0.7) ** 2)
    assert n2.grad == pytest.approx(n1.grad)

def test_division():
    a, b = Value(3.0), Value(4.0)
    q = a / b
    q.backward()
    assert a.grad == pytest.approx(1 / 4)
    assert b.grad == pytest.approx(-3 / 16)

def test_radd_for_sum():
    vals = [Value(1.0), Value(2.0), Value(3.0)]
    s = sum(vals)
    s.backward()
    assert s.data == 6.0 and all(v.grad == 1.0 for v in vals)
```

Run from the repo root: `uv run pytest micrograd -q`.

*You should see* `5 passed`. Then break the engine on purpose: change the `+=` in `__mul__`'s `_backward` to `=` for `self.grad` only, and run again. Write in your log which test failed and what it said, then put the `+=` back and run again to `5 passed`.

*If it broke:* the reuse test is the one that catches a `=`. If it passes with the `=` in place, you changed the wrong line; the reused variable in that test is multiplied, not added.

**Extension — How small should `h` be?** *(assigned)*

Since B17 you have checked derivatives by nudging an input by a small `h`. Your engine now gives exact derivatives, so you can measure how wrong nudging is, on your neuron. Before you run anything, write in your log the `h` you expect to give the smallest error, and whether a two-sided nudge, `(f(x + h) - f(x - h)) / 2h`, will do better or worse than the one-sided one. Commit the log. I will check your commit timestamps.

Create `micrograd/nudge_sweep.py`:

```python
from engine import Value
from data import load

xs, _, _ = load()
f = xs[0]

def out(w3):
    return (Value(f[0]) * 0.5 + Value(f[1]) * -0.3 + Value(f[2]) * w3 + 0.1).tanh()

w3 = Value(0.8)
o = out(w3)
o.backward()
exact = w3.grad
print(f"engine: do/dw3 = {exact:.12f}")
print(f"{'h':>7}  {'one-sided error':>16}  {'two-sided error':>16}")
for k in range(1, 13):
    h = 10.0 ** -k
    one = (out(0.8 + h).data - out(0.8).data) / h
    two = (out(0.8 + h).data - out(0.8 - h).data) / (2 * h)
    print(f"{h:7.0e}  {abs(one - exact):16.3e}  {abs(two - exact):16.3e}")
```

Run it: `uv run python micrograd/nudge_sweep.py`.

*You should see* the one-sided error shrink by about ten times for every ten times smaller `h`, bottom out somewhere around `h = 1e-7` or `1e-8`, and then grow again. The two-sided error shrinks about a hundred times per step, bottoms out much lower and sooner, around `h = 1e-5`, and then also grows. The growth at the bottom of the table is floating-point: `f(x + h)` and `f(x)` agree in so many digits that subtracting them leaves only rounding noise, and dividing by a tiny `h` magnifies it. Where exactly your minimums fall is yours.

Write `micrograd/FINDINGS-b21.md`: the table; your baseline, which is the engine's exact gradient, checked against PyTorch in Step 4; your committed predictions and whether they held; and where it got worse, with numbers: the smallest error each method reached, the `h` where it did, and how much worse each was at `h = 1e-12`.

Commit, push, and sign off:

```bash
git add micrograd/ scratch/
git commit -m "B21: nn.py, five engine tests, finite-difference error sweep"
git push
```

The pull request you opened on Day 1 picks this up. Say in the PR body where this is weakest.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

Push again each day; one PR, not two; one log entry per meeting.

**Deliverable**
`micrograd/engine.py` with a complete `.backward()` and the full op set + `micrograd/nn.py` + `micrograd/test_engine.py` (five derivatives verified against hand work, including the reuse case) + `micrograd/tanh_two_ways.py` + `micrograd/vs_torch.py` + `micrograd/nudge_sweep.py` + `micrograd/FINDINGS-b21.md` + `scratch/b21-video.py`.

**Reflection Questions**

1. Paste the eight rows from `vs_torch.py`. Pick `do/dw3` and write out, with your own numbers, the chain of local derivatives your engine multiplied to get it, from `o` back to `w3`, and match each factor to a line of code in `engine.py` (file and line). Then say which factor PyTorch computed that your B17 `dloss` would have had no equivalent for.

2. Paste the `pytest` failure from Step 6 with `=` in `__mul__`, and the comment you wrote in `scratch/b21-video.py` predicting whether `tanh` from `exp` would give the same gradient. Say, for the reuse test, which two paths the gradient took to reach `a`, what each path contributed with `a = 2` and `b = -3`, and which of the two the `=` threw away.

3. Paste your `nudge_sweep.py` table beside your committed predictions. Name the `h` where each method was best and its error there. Say where it got worse and why, and then say which `h` you will use from now on when you want to check a gradient by nudging, and what error you now expect from it.
