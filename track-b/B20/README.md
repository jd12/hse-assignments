# B20 · micrograd II: Topological Sort

**Meetings:** D36–D37 · **Points:** 15 pts

**Watch — 34 min**
Day 1 — 16 min
[Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=3172s) · segment 00:52:52 → 01:09:02, "manual backprop #2, a neuron" (16m)
 · `micrograd/engine.py`, `scratch/b20-video.py` (same three import lines as B19's scratch file, plus `from viz import draw_dot`), and paper.

Day 2 — 18 min
[Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4142s) · segment 01:09:02 → 01:27:05, "backward function per operation", "backward for a whole graph" and "fixing the multi-use node bug" (18m)
 · Same files. This is the segment where the engine starts computing gradients by itself, so type slowly.

**During the video**

**Type everything he types, and run it.** Changes to the `Value` class go into `micrograd/engine.py`; the expressions he builds go into `scratch/b20-video.py`. Run the scratch file after every change he evaluates.

**Day 1 · 00:52:52 → 01:09:02.** He builds a single neuron with two inputs, adds `tanh` to `Value` so the neuron has a squashing function at the end, and then backpropagates through it by hand, node by node. Type `tanh` into the engine as he writes it. Then, for every node, **write the local derivative on the edge on paper before he says it**: for a `+`, what does each input get; for a `*`, what does each input get; for `tanh`, what is `d tanh(n)/dn` in terms of the output. Unpause and check. Your page from this segment is practice for Step 3, where you do the same thing on your own neuron with nobody to check you.

**Day 2 · 01:09:02 → 01:27:05.** He moves each local derivative into the operation that creates it, as a small function called `_backward` stored on the output `Value`, and then calls them one at a time in the right order. Then he writes the ordering itself: a topological sort. At 01:22:28 he finds a bug in his own engine when a node is used twice. **Before he fixes it, pause and predict** in your scratch file, as a comment, what `a.grad` should be for `b = a + a` and what his engine printed instead. Then type the fix. Of everything in this segment, that one fix causes the most lost evenings.

**Notes**

Topological sort answers one question: in what order is it safe to process the nodes? You cannot compute a node's gradient until every node that *uses* it has passed its share back. Sorting the graph so children come before parents, then walking it in reverse, guarantees that.

The implementation is a depth-first traversal with a `visited` set:
```python
topo = []
visited = set()
def build(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build(child)
        topo.append(v)
```
`topo.append(v)` comes **after** the loop. Put it before and your ordering is wrong in a way that produces garbage with no error. The extension makes you do exactly that and measure it.

The `visited` set is not an optimization. A node reachable along two paths would otherwise be added twice, and its `_backward` would run twice.

**`+=` not `=`.** When a node is used twice, gradient arrives from two places and both have to be kept. With `=`, the second arrival overwrites the first. Your loss would still go down a little, which makes it worse: a wrong gradient that half works is harder to find than one that crashes.

`.backward()` sets `self.grad = 1.0` on the output before walking the reversed list. Forget that and every gradient in your graph is zero, with no error.

`tanh` has the derivative `1 - tanh(n)²`. The `_backward` for `tanh` uses the output it already computed, `t`, rather than recomputing anything. Write `1 - t**2`, not `1 - n**2`; the second one is a bug that gives plausible numbers.

**Walkthrough — Your neuron's gradients, by hand and by the engine**

**Day 1 starts here.**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:
```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/topo
```
*If B19 is not approved yet:* `git switch dev/value && git switch -c dev/topo`. The engine grows in every assignment from here to the milestone, so this branch has to start from the newest one.
```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-track, not main
bash scripts/start-entry.sh
```
Under the timestamp, write the checklist for the whole assignment, and tick what you finish each day:
```markdown
- [ ] Day 1: Karpathy micrograd 00:52:52 → 01:09:02, tanh into engine.py, local derivatives on paper before he says them
- [ ] Day 1: Steps 2–3, my neuron with tanh, and its gradients by hand on one page, committed
- [ ] Day 2: Karpathy micrograd 01:09:02 → 01:27:05, _backward per op, topo sort, the multi-use fix
- [ ] Day 2: Steps 4–5, check engine.py, then let .backward() check my hand page
- [ ] Day 2: Extension, break the ordering two ways and count what goes wrong
- [ ] Push each day; one PR
```

Watch Day 1's segment and type along before Step 2.

**Step 2. Your neuron, with `tanh`.**

Your B19 neuron read three features from a window of your corpus and added them up. Now it squashes the sum with `tanh`, which is what Karpathy's neuron does. Create `micrograd/neuron_b20.py`:

```python
from engine import Value
from data import load

xs, ys, windows = load()
f = [round(v, 2) for v in xs[0]]             # your first real window, to 2 decimals so the page is doable
print("window:", repr(windows[0]))
print("features:", f)

x1, x2, x3 = Value(f[0], label="x1"), Value(f[1], label="x2"), Value(f[2], label="x3")
w1, w2, w3 = Value(0.5, label="w1"), Value(-0.3, label="w2"), Value(0.8, label="w3")
b = Value(0.1, label="b")

n = x1 * w1 + x2 * w2 + x3 * w3 + b; n.label = "n"
o = n.tanh(); o.label = "o"
print(f"n = {n.data:.4f}   o = {o.data:.4f}")
```

Run it: `uv run python micrograd/neuron_b20.py`.

*You should see* your window, your three features rounded to two places, and an `o` strictly between -1 and 1 with the same sign as `n`. `tanh` squashes any number into that range and keeps its sign.

**Step 3. The hand page, committed before the engine can check it.**

On one sheet of paper, draw your neuron's graph: seven leaves, three products, the sums, `n`, and `o`. Using the numbers Step 2 printed, and a calculator for `tanh`, work backwards from `o` and **label every edge with its local derivative**, then write the gradient of `o` with respect to every node next to the node. Start with `do/do = 1`. The one that needs care is `do/dn = 1 - o²`: use your `o`, not your `n`.

Photograph it as `micrograd/hand-b20.jpg` and commit it by itself:

```bash
git add micrograd/neuron_b20.py micrograd/hand-b20.jpg scratch/b20-video.py micrograd/engine.py
git commit -m "B20: my neuron with tanh, gradients by hand"
git push -u origin dev/topo
```

Open the pull request now with **jd12** as reviewer. I will check your commit timestamps: the hand page has to exist before `.backward()` does. Sign off the log.

**Day 2 starts here.** Open a new entry with `bash scripts/start-entry.sh` in the log repo and copy over what is left of the checklist. Watch Day 2's segment first and type along.

**Step 4. Check `engine.py` against where he stops.**

At 01:27:05, your engine should have a `grad` and a `_backward` on every `Value`, a `_backward` closure inside `__add__`, `__mul__` and `tanh`, and a `backward` method. Compare yours with this, line by line:

```python
import math

class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

Then check the fix you just typed:

```bash
uv run python -c "import sys; sys.path.insert(0, 'micrograd'); from engine import Value; a = Value(3.0); b = a + a; b.backward(); print(a.grad)"
```

*You should see* `2.0`. If you see `1.0`, one of your closures still uses `=`.

*If it broke:* `AttributeError: 'Value' object has no attribute '_backward'` means a `Value` was created somewhere without going through `__init__`'s new line. `math` not defined means the `import math` at the top is missing.

**Step 5. Let the engine check your page.**

Add to the bottom of `neuron_b20.py`:

```python
o.backward()
for v in (o, n, x1, x2, x3, w1, w2, w3, b):
    print(f"do/d{v.label:3} = {v.grad: .4f}")
```

Run it and hold your page next to the output.

*You should see* every gradient on your page match to the precision you worked in; `do/do = 1.0000`, `do/db` equal to `do/dn`, `do/dw3` equal to `x3` times `do/dn`, and `do/dx3` equal to `w3` times `do/dn`. Where your page and the engine disagree, do not change the page. Write the correct number next to the wrong one on paper, in a different color, rephotograph it as `micrograd/hand-b20-corrected.jpg`, and say in your log which edge you got wrong and why.

**Extension — What the ordering buys you** *(assigned)*

The engine gets the right answer because of two details in `build_topo`: `topo.append(v)` after the loop, and the `visited` set. Remove each one, separately, and measure what goes wrong. Before you run anything, write in your log, for each break, whether you expect an error, all-zero gradients, or wrong numbers that look plausible. Commit the log. I will check your commit timestamps.

To see the `visited` set matter, the graph needs a node used twice, so this one uses your `n` in two places. Create `micrograd/topo_check.py`:

```python
from engine import Value
from data import load

def build_order(root, append_first=False, use_visited=True):
    topo, visited = [], set()
    def build(v):
        if use_visited:
            if v in visited:
                return
            visited.add(v)
        if append_first:
            topo.append(v)                   # the bug: parent before its children
        for child in v._prev:
            build(child)
        if not append_first:
            topo.append(v)                   # correct: parent after its children
    build(root)
    return topo

def violations(topo):
    first = {}
    for i, v in enumerate(topo):
        first.setdefault(v, i)
    return sum(1 for v in topo for c in v._prev if first[c] > first[v])   # a child listed after its parent

def grads_with(root, order, leaves):
    for v in build_order(root):
        v.grad = 0.0
    root.grad = 1.0
    for v in reversed(order):
        v._backward()
    return [leaf.grad for leaf in leaves]

xs, ys, windows = load()
f = xs[0]
w1, w2, w3, b, w4 = Value(0.5), Value(-0.3), Value(0.8), Value(0.1), Value(-1.5)
n = Value(f[0]) * w1 + Value(f[1]) * w2 + Value(f[2]) * w3 + b
o = (n * w4 + n).tanh()                      # n is used by two different nodes
leaves = [w1, w2, w3, b, w4]

right = grads_with(o, build_order(o), leaves)
print(f"{'order':28} {'nodes':>5} {'violations':>10}  gradients of w1 w2 w3 b w4")
for name, kw in (("correct", {}),
                 ("append before the loop", {"append_first": True}),
                 ("no visited set", {"use_visited": False})):
    order = build_order(o, **kw)
    g = grads_with(o, order, leaves)
    wrong = sum(abs(a - r) > 1e-12 for a, r in zip(g, right))
    print(f"{name:28} {len(order):5d} {violations(order):10d}  {[round(x, 5) for x in g]}  wrong: {wrong}")
```

Run it: `uv run python micrograd/topo_check.py`.

*You should see* three rows. **Correct**: 17 nodes, 0 violations, 0 wrong; the gradients are yours and nobody else's, because `f` is your window. **Append before the loop**: the same 17 nodes, many violations, and gradients that are all zero, because walking the reversed list reaches the leaves first, before anything has passed gradient down to them. **No visited set**: more than 17 nodes, because `n` and everything under it is listed twice; 0 violations by this checker; and gradients that are wrong but look like real numbers, most of them larger than the correct ones, because the part of the graph under `n` passed its gradient down twice. `w4` may come out right, since it sits above the reused node.

That last row is the dangerous one. It has no error, no zeros, and it passes the violations check you just wrote. Write `micrograd/FINDINGS-b20.md`: all three rows as printed; your committed predictions and whether they held; and where it got worse, with numbers: the break that fooled your own checker, and by what factor its gradient for `w3` differed from the correct one. Then write one sentence on what check *would* have caught it.

Commit, push, and sign off:

```bash
git add micrograd/ scratch/
git commit -m "B20: backward checked against hand page, topo order broken two ways and measured"
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
A working `topo` ordering and `backward()` in `micrograd/engine.py` + `micrograd/hand-b20.jpg` (every edge labeled with its local derivative, committed before `.backward()` checked it) + `micrograd/neuron_b20.py` + `micrograd/topo_check.py` + `micrograd/FINDINGS-b20.md` + `scratch/b20-video.py`.

**Reflection Questions**

1. Paste the `do/d...` lines from Step 5 and point to one edge on your hand page, with the commit hash of the page. State the local derivative you wrote on it, where that number came from in your own neuron, and whether the engine agreed. If any edge disagreed, say which one and what you did wrong; if none did, say which edge took you longest and why.

2. Paste the comment you wrote in `scratch/b20-video.py` predicting `a.grad` for `b = a + a`, and the `2.0` from Step 4. Say, in terms of the two lines inside `__add__`'s `_backward`, exactly what `=` did to the first contribution when `self` and `other` were the same object, and why the answer came out as it did rather than as zero.

3. Paste the three rows from `topo_check.py` beside your committed predictions. For the "no visited set" row, say how many extra nodes appeared in the order and which ones they were, then explain why `violations` reported 0 for an order that gave wrong gradients. What did your checker measure, and what should it have measured?
