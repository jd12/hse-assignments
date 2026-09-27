# B19 · micrograd I: The Value Class and the Forward Pass

**Meetings:** D34–D35 · **Points:** 15 pts

**Watch — 34 min**
Day 1 — 13 min
[Karpathy, The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1149s) · segment 00:19:09 → 00:32:10, "Value object and visualization" (13m)
 · `micrograd/engine.py` and `scratch/b19-video.py` open side by side, terminal in the repo root.

Day 2 — 21 min
[Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1930s) · segment 00:32:10 → 00:52:52, "manual backprop #1, simple expression" through "preview of one optimization step" (21m)
 · Same two files, plus paper: he does the chain rule by hand and so do you.

**During the video**

**Type everything he types, and run it.** He is in a notebook and you are not. His `Value` class goes into `micrograd/engine.py`, which is the file you will keep adding to until the milestone. Everything else he types, the expressions he builds out of `Value`s and the calls that draw them, goes into `scratch/b19-video.py`, which starts with these three lines so it can find your engine:

```python
import sys
sys.path.insert(0, "micrograd")
from engine import Value
```

Run it with `uv run python scratch/b19-video.py` from the repo root every time he evaluates a cell.

**Day 1 · 00:19:09 → 00:32:10.** He builds `Value` one piece at a time: the number it holds, how to print it, `+`, `*`, the record of which `Value`s produced it, the operation that produced it, and a label. Then he writes a function that draws the whole expression as a graph. The drawing code goes into `micrograd/viz.py`, not the engine: add `from viz import draw_dot` to the top of your scratch file under the other import. Where he displays the graph inline, you write `draw_dot(L).render("scratch/b19-graph", format="svg", cleanup=True)` and open `scratch/b19-graph.svg` in a browser. Step 2 installs what that needs; do it before you press play.

**Day 2 · 00:32:10 → 00:52:52.** He adds a `grad` to every `Value`, and then does the chain rule by hand, node by node, from the output back to the leaves, checking each number by nudging an input and re-running. Before he fills in each node's gradient, pause and write your own answer on paper, then unpause and check. Photograph the page as `micrograd/hand-b19.jpg`. At 00:51:10 he previews one optimization step: every leaf nudged a little in the direction of its gradient, and the output moves. Type it and watch which way yours moves.

**Notes**

`__add__` and `__mul__` are Python's operator-overloading hooks: `a + b` calls `a.__add__(b)`. That is why `Value` objects can be written with ordinary math syntax.

Each `Value` stores `data`, `_prev` (the set of `Value`s that produced it) and `_op` (which operation). Those three fields are the entire computational graph. There is no separate graph object.

Common first bug: `Value(2.0) + 1` crashes with `AttributeError: 'int' object has no attribute 'data'`, because `1` is an int, not a `Value`. The fix is one line at the top of `__add__` and `__mul__`: `other = other if isinstance(other, Value) else Value(other)`. Add it the first time you hit it, not the fifth. `1 + Value(2.0)`, the other way round, is a different crash with a different fix, and it arrives in B21.

**Graphviz is two installs, not one.** `uv add graphviz` installs the Python package that writes the drawing instructions. The program that turns them into a picture is `dot`, and it comes from `brew install graphviz`. With only the first, `render` fails with `ExecutableNotFound: failed to execute 'dot'`. If Homebrew will not install it, Step 3 gives you a text version of the graph instead. The picture is worth the fight, so try.

**Walkthrough — A neuron's forward pass on your corpus**

**Day 1 starts here.**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:
```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/value
```
```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-track, not main
bash scripts/start-entry.sh
```
Under the timestamp, write the checklist for the whole assignment, and tick what you finish each day:
```markdown
- [ ] Day 1: install graphviz, both halves
- [ ] Day 1: Karpathy micrograd 00:19:09 → 00:32:10, Value into engine.py, drawing into viz.py, expressions into scratch
- [ ] Day 1: Steps 3–5, check engine.py, build data.py, a neuron's forward pass on one real window, drawn
- [ ] Day 2: Karpathy micrograd 00:32:10 → 00:52:52, with my own answers on paper before his
- [ ] Day 2: Steps 6–7, gradients of my neuron by hand and by nudging, and one optimization step
- [ ] Day 2: Extension, one neuron with weights I choose against real and shuffled text
- [ ] Push each day; one PR
```

**Step 2. Install graphviz.**

```bash
brew install graphviz
uv add graphviz
dot -V
uv run python -c "import graphviz; print(graphviz.__version__)"
```

*You should see* `dot - graphviz version 12.2.1` or similar from the first line (the program), and a version like `0.21` from the second (the Python package). The numbers do not matter; two versions do. With only the package installed, `render` and `graphviz.version()` both fail with `ExecutableNotFound: failed to execute 'dot'`. Then watch Day 1's segment and type along.

**Step 3. Check `engine.py` and `viz.py` against where he stops.**

At 00:32:10, `micrograd/engine.py` has one class, `Value`, with `__init__` storing `data`, `_prev` (as a `set` of the children), `_op` and `label`; a `__repr__`; and `__add__` and `__mul__`, each returning a new `Value` built from the two inputs, with `"+"` or `"*"` as its `_op`. The two `isinstance` lines from the Notes are the one thing that is yours rather than his; keep them. Check it:

```bash
uv run python -c "import sys; sys.path.insert(0, 'micrograd'); from engine import Value; a = Value(2.0); d = a * -3.0 + 10; print(d, d._op, len(d._prev))"
```

*You should see* `Value(data=4.0) + 2`. If yours differs anywhere else, his is the one to match; fix it now, because every later segment edits this file.

`micrograd/viz.py` should be this; `print_graph` at the bottom is yours:

```python
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})    # left to right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | data %.4f }" % (n.label, n.data), shape="record")
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot

def print_graph(v, depth=0):
    print("    " * depth + f"{v.label or '?'} = {v.data:.4f}" + (f"   ({v._op})" if v._op else ""))
    for child in v._prev:
        print_graph(child, depth + 1)
```

`print_graph` is the fallback if the `dot` program will not install: the same graph, as indented text, read from the output back to the leaves.

**Step 4. Your corpus, as numbers a neuron can read.**

The dataset for the rest of micrograd comes from your corpus, and it asks a question a model can learn: is this 60-character window real text from your corpus, or the same characters shuffled? Each window becomes three numbers: its share of spaces, its share of vowels, and its share of character pairs that are among your corpus's 50 most common. Create `micrograd/data.py`:

```python
import random
from collections import Counter

def load(n_each=4, width=60, seed=0):
    text = open("data/corpus.txt", encoding="utf-8").read().lower()
    text = " ".join(text.split())                        # every run of whitespace becomes one space
    pairs = Counter(text[i:i + 2] for i in range(len(text) - 1))
    common = {p for p, _ in pairs.most_common(50)}       # your corpus's 50 most common character pairs

    def features(s):
        bigrams = [s[i:i + 2] for i in range(len(s) - 1)]
        return [s.count(" ") / len(s),                             # share of spaces
                sum(c in "aeiou" for c in s) / len(s),             # share of vowels
                sum(b in common for b in bigrams) / len(bigrams)]  # share of common pairs

    def sample(rng, n):
        out = []
        for _ in range(n):
            start = rng.randrange(len(text) - width)
            real = text[start:start + width]
            chars = list(real)
            rng.shuffle(chars)
            out.append((real, 1.0))                      # a real window of your corpus: +1
            out.append(("".join(chars), -1.0))           # the same characters, shuffled: -1
        return out

    # scale each feature so it averages 0 with a spread of 1, using a fixed reference sample
    ref = [features(s) for s, _ in sample(random.Random(12345), 500)]
    cols = list(zip(*ref))
    means = [sum(c) / len(c) for c in cols]
    stds = [(sum((v - m) ** 2 for v in c) / len(c)) ** 0.5 for c, m in zip(cols, means)]

    xs, ys, windows = [], [], []
    for s, label in sample(random.Random(seed), n_each):
        xs.append([(v - m) / sd for v, m, sd in zip(features(s), means, stds)])
        ys.append(label)
        windows.append(s)
    return xs, ys, windows

if __name__ == "__main__":
    xs, ys, windows = load()
    for x, y, w in zip(xs, ys, windows):
        print(f"{y:+.0f}  [{x[0]:+.3f} {x[1]:+.3f} {x[2]:+.3f}]  {w[:40]!r}")
```

Run it: `uv run python micrograd/data.py`.

*You should see* eight rows in pairs: a `+1` row of your real text, then a `-1` row of the same characters scrambled. Every number is roughly between -3 and +3. The invariant to check with your own eyes: **within each pair, the first two numbers are identical.** Shuffling moves characters around but does not change how many spaces or vowels there are. Only the third number can tell the pair apart, and it is higher for the real row in every pair.

*If it broke:* a `ZeroDivisionError` in `stds` means your corpus is tiny or almost all one character. Tell me.

**Step 5. A neuron's forward pass, drawn.**

Create `micrograd/forward.py`:

```python
from engine import Value
from viz import draw_dot, print_graph
from data import load

xs, ys, windows = load()
f = xs[0]                                    # your first real window
print("window:", repr(windows[0]))

x1, x2, x3 = Value(f[0], label="x1 spaces"), Value(f[1], label="x2 vowels"), Value(f[2], label="x3 pairs")
w1, w2, w3 = Value(1.0, label="w1"), Value(1.0, label="w2"), Value(1.0, label="w3")
b = Value(0.0, label="b")

x1w1 = x1 * w1; x1w1.label = "x1*w1"
x2w2 = x2 * w2; x2w2.label = "x2*w2"
x3w3 = x3 * w3; x3w3.label = "x3*w3"
s1 = x1w1 + x2w2; s1.label = "s1"
s2 = s1 + x3w3;   s2.label = "s2"
n = s2 + b;       n.label = "n"

print("n =", n)
print_graph(n)
draw_dot(n).render("micrograd/forward", format="svg", cleanup=True)
```

Run it: `uv run python micrograd/forward.py`, and open `micrograd/forward.svg`.

*You should see* `n` equal to the sum of your three feature values, since every weight is 1 and the bias is 0: check it against row one of Step 4 by hand. The picture reads left to right from seven leaves (three features, three weights, the bias) to `n`. Count the longest path from a leaf to `n`: `x1 → x1*w1 → s1 → s2 → n` is five nodes, which is the depth the deliverable asks for.

Commit and push: `git add micrograd/ scratch/ pyproject.toml uv.lock && git commit -m "B19: Value with + and *, graph drawing, corpus features, neuron forward pass"`, then `git push -u origin dev/value`. Open the pull request now with **jd12** as reviewer. Sign off the log.

**Day 2 starts here.** Open a new entry with `bash scripts/start-entry.sh` in the log repo and copy over what is left of the checklist. Watch Day 2's segment first; it adds `self.grad = 0.0` to `__init__` and a `grad` field to the drawing's label. Make both changes in your files as he does.

**Step 6. Gradients of your neuron, by hand and by nudging.**

`n = x1·w1 + x2·w2 + x3·w3 + b`. On the same paper as the video, write `∂n/∂w1`, `∂n/∂w2`, `∂n/∂w3` and `∂n/∂b`, as numbers from your own Step 4 row. Then add to the bottom of `forward.py`:

```python
# set the gradients by hand, the way he did, from the output back
n.grad = 1.0
s2.grad = n.grad; b.grad = n.grad
s1.grad = s2.grad; x3w3.grad = s2.grad
x1w1.grad = s1.grad; x2w2.grad = s1.grad
w1.grad = x1.data * x1w1.grad; w2.grad = x2.data * x2w2.grad; w3.grad = x3.data * x3w3.grad
x1.grad = w1.data * x1w1.grad; x2.grad = w2.data * x2w2.grad; x3.grad = w3.data * x3w3.grad
draw_dot(n).render("micrograd/forward-grads", format="svg", cleanup=True)

# check each weight's gradient by nudging it, his way
def n_with(dw1=0.0, dw2=0.0, dw3=0.0, db=0.0):
    return (f[0] * (1.0 + dw1) + f[1] * (1.0 + dw2) + f[2] * (1.0 + dw3)) + (0.0 + db)

h = 1e-6
for name, v, kw in (("w1", w1, "dw1"), ("w2", w2, "dw2"), ("w3", w3, "dw3"), ("b", b, "db")):
    nudge = (n_with(**{kw: h}) - n_with()) / h
    print(f"dn/d{name}  by hand {v.grad: .6f}   by nudging {nudge: .6f}")
```

*You should see* four lines where the two columns agree to five or six decimals, and the gradient of each weight equal to its own feature value: `dn/dw3` is your `x3`. A weight's gradient is how much its input shows up in the output. The bias always gets exactly 1, because it is added straight in.

**Step 7. One optimization step.**

Add:

```python
step = 0.01
for v in (w1, w2, w3, b):
    v.data += step * v.grad
n_new = (x1 * w1 + x2 * w2 + x3 * w3) + b
predicted = step * (x1.data ** 2 + x2.data ** 2 + x3.data ** 2 + 1)
print(f"n before {n.data:.6f}   after {n_new.data:.6f}   change {n_new.data - n.data:.6f}   predicted {predicted:.6f}")
```

*You should see* `n` go up, and the change equal the predicted number to six decimals. Each weight moved by `0.01` times its gradient, and its gradient is its input, so the output moved by `0.01` times the sum of the inputs squared, plus `0.01` for the bias. Moving along the gradient always increases the output; B22 moves the other way to make a loss go down.

**Extension — Can one neuron you set by hand tell real text from shuffled?** *(assigned)*

A neuron says "real" when `n > 0` and "shuffled" when `n < 0`. With all weights 1 and bias 0 it is a guess. Before you run anything, write in your log the weights and bias you think will get all eight right, chosen by looking at your Step 4 table, and how many of 100 new windows you expect that same neuron to get right. Commit the log. I will check your commit timestamps.

Create `micrograd/hand_neuron.py`:

```python
from engine import Value
from data import load

def neuron(x, w, b):
    return (Value(x[0]) * w[0] + Value(x[1]) * w[1] + Value(x[2]) * w[2]) + b

def score(xs, ys, w, b):
    return sum((neuron(x, w, b).data > 0) == (y > 0) for x, y in zip(xs, ys))

train_x, train_y, _ = load()                          # your 8
test_x, test_y, _ = load(n_each=50, seed=99)          # 100 windows it has never seen

for name, w, b in (("baseline", (1.0, 1.0, 1.0), 0.0),
                   ("mine",     (0.0, 0.0, 0.0), 0.0)):     # put your choice here
    print(f"{name:9} w={w} b={b}   train {score(train_x, train_y, w, b)}/8"
          f"   new windows {score(test_x, test_y, w, b)}/100")
```

Replace the `mine` row with your committed choice and run it: `uv run python micrograd/hand_neuron.py`.

*You should see* the baseline get some of your eight wrong. A good hand choice gets 8 of 8: the pairs share their first two features, so only the weight on `x3` can separate a pair, and the bias shifts where the dividing line falls. On 100 new windows your neuron may do a little worse than on the eight you looked at, or may still get all 100: with a prose corpus the third feature separates the two classes very cleanly. If you get 100 of 100, run the same neuron on `load(n_each=500, seed=7)` and report that score as the place it got worse; there will be misses in a thousand. The baseline's score on the 100 may surprise you, since it has `x3` in it too; compare your row with the baseline on the 100, not only on the eight.

Write `micrograd/FINDINGS-b19.md`: both rows as printed, your committed prediction, whether it held, and where it got worse with the number: the drop from your 8 to the 100, or to the 1,000 if the 100 were perfect. Say which weight did the work and what the other two weights could never do, given the invariant from Step 4.

Commit, push, and sign off:

```bash
git add micrograd/ scratch/
git commit -m "B19: neuron gradients by hand and by nudging, one step, hand-set neuron on real vs shuffled"
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
`micrograd/engine.py` (a working `Value` with `+`, `*`, `grad`) + `micrograd/viz.py` + `micrograd/data.py` + `micrograd/forward.py` with `forward.svg` and `forward-grads.svg` (an expression at least five nodes deep, rendered) + `micrograd/hand_neuron.py` + `micrograd/FINDINGS-b19.md` + `micrograd/hand-b19.jpg` + `scratch/b19-video.py`.

**Reflection Questions**

1. Paste the eight rows `data.py` printed. Point to one pair and say, using its characters, why its first two numbers are identical and its third is not. Then say what would happen to your extension if your corpus's 50 most common pairs were mostly pairs that contain a space, and check whether they are: add `print(sorted(common))` inside `load` right after `common` is built, run `data.py` once, paste the line, and take the print back out.

2. Paste the four `dn/d...` lines from Step 6 and the `n before ... predicted` line from Step 7. Using your own `x3` and the gradient of `w3`, explain in two sentences why nudging `w3` by 0.01 changed `n` by exactly what it did. Then say what `_prev` held for your node `s2` and how `draw_dot` found `x3` from `n` without being told about it.

3. Paste both lines from `hand_neuron.py` and the prediction you committed. Where did it get worse, and by how much? Find one of the new windows your neuron got wrong (from the 100, or from the 1,000 if it was perfect on the 100; print it), paste it, and say what about that window's three numbers fooled a neuron that was perfect on your eight.
