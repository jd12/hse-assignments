# T08 · `Value`: Your Counts as an Expression Graph, and the Chain Rule by Hand

**Meetings:** D31 · **Points:** 15 pts

**Watch — 32 min**
[Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1149s) · 00:19:09 → 00:51:10, "starting the core Value object of micrograd and its visualization" and "manual backpropagation example #1: simple expression" (32m)
 · `micrograd/engine.py` (new, empty) and `scratch/t08-video.py` open side by side, terminal in the repo root. Step 2 installs graphviz; do it before you press play.

**During the video**

**Type everything he types, and run it.** He is in a notebook and you are not. His `Value` class goes into `micrograd/engine.py`, the file you keep adding to until the milestone. Everything else he types, the expressions he builds out of `Value`s, goes into `scratch/t08-video.py`, which starts with these lines so it can find your engine and the drawing code that came with the template:

```python
import sys
sys.path.insert(0, "micrograd")
from engine import Value
from viz import draw_dot
```

Run it with `uv run python scratch/t08-video.py` from the repo root every time he evaluates a cell.

**00:19:09 → 00:32:10.** He builds `Value` one piece at a time: the number it holds, how to print it, `+`, `*`, the record of which `Value`s produced it, the operation that produced it, and a label. Then he writes `draw_dot`, which draws the whole expression as a graph. **Do not type `draw_dot`**: it is `micrograd/viz.py`, shipped with the template, and you import it. Where he displays the graph inline, you write `draw_dot(L).render("scratch/t08-graph", format="svg", cleanup=True)` and open `scratch/t08-graph.svg` in a browser.

**00:32:10 → 00:51:10 · compute before he does.** He adds `grad` to every `Value`, then does the chain rule by hand on `L = (a*b + c) * f`, from `L` back to the leaves, checking each number by nudging. Before he fills in each node's gradient, pause, compute it yourself and write it under `## During the video` in `evidence/T08.md` with the node's name, then unpause and check. Seven numbers: `dL/dL`, `dL/dd`, `dL/df`, `dL/dc`, `dL/de`, `dL/da`, `dL/db`. Step 4 does the same on an expression built from your corpus, with nobody to check you until Step 5.

**Notes**

**`__add__` and `__mul__` are Python's operator hooks.** `a + b` calls `a.__add__(b)`. That is why `Value` objects can be written with ordinary math syntax, and why `d = a * b + c` builds a graph without anyone calling a "build graph" function.

**Each `Value` stores `data`, `_prev` and `_op`.** `_prev` is the set of `Value`s that produced it; `_op` is which operation. Those three fields are the entire graph. There is no separate graph object, and `draw_dot` finds every node by starting at the output and following `_prev` backwards.

**Common first bug:** `Value(2.0) + 1` crashes with `AttributeError: 'int' object has no attribute 'data'`, because `1` is an `int`, not a `Value`. The fix is one line at the top of `__add__` and `__mul__`: `other = other if isinstance(other, Value) else Value(other)`. It reads "if `other` is already a `Value` keep it, otherwise wrap it in one." Add it the first time you hit it, not the fifth. `1 + Value(2.0)`, the other way round, is a different crash with a different fix, and it arrives in T10.

**Graphviz is two installs, not one.** `uv add graphviz` installs the Python package that writes the drawing instructions. The program that turns them into a picture is `dot`, and it comes from `brew install graphviz`. With only the first, `render` fails with `ExecutableNotFound: failed to execute 'dot'`. If Homebrew will not install it, `print_graph` in `viz.py` prints the same graph as indented text, and every script today falls back to it.

**Walkthrough — the chain rule on your counts**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T07 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/value
```

*If T07 is not approved yet:* `git switch dev/slope && git switch -c dev/value`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Step 2: graphviz, both halves
- [ ] micrograd 19:09 → 51:10 watched, Value typed into engine.py, the seven gradients computed before he did
- [ ] Step 3: engine.py checked, expr.py run and drawn
- [ ] Step 4: the five gradients of my expression by hand, in evidence/T08.md and committed
- [ ] Step 5: expr.py grads checks my page
- [ ] Fill every slot in evidence/T08.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Install graphviz.**

```bash
brew install graphviz
uv add graphviz
dot -V
uv run python -c "import graphviz; print(graphviz.__version__)"
```

*You should see* `dot - graphviz version 12.2.1` or similar from the third line (the program), and a version like `0.21` from the fourth (the Python package). The numbers do not matter; two versions do. *If it broke* with `brew: command not found` or a Homebrew error you cannot fix in two minutes, skip the program: `uv add graphviz` alone lets every script today run, with text graphs instead of pictures. Then watch the segment and type along.

**Step 3. Check `engine.py`, then build the expression from your counts.**

At 00:51:10, `micrograd/engine.py` has one class, `Value`, with `__init__` storing `data`, `grad` (starting at `0.0`), `_prev` (as a `set` of the children), `_op` and `label`; a `__repr__`; and `__add__` and `__mul__`, each returning a new `Value` built from the two inputs, with `"+"` or `"*"` as its `_op`. The two `isinstance` lines from the Notes are the one thing that is yours rather than his; keep them. Check it:

```bash
uv run python -c "import sys; sys.path.insert(0, 'micrograd'); from engine import Value; a = Value(2.0); d = a * -3.0 + 10; print(d, d._op, len(d._prev), d.grad)"
```

*You should see* `Value(data=4.0) + 2 0.0`. If yours differs anywhere else, his is the one to match; fix it now, because every later assignment edits this file.

Now the expression. `micrograd/expr.py` came with the template; open it and read the top. It takes T01's two-word counts for chapter 2 and chapter 7, per 1,000 words and rounded to one decimal so a hand page is doable, and builds the two-word dot product out of your `Value`s: `p1 = a1 * b1`, `p2 = a2 * b2`, `d = p1 + p2`, then `L = d * s` with `s = 0.01`, a scale, so the chain rule has a product to pass through at the top, the way his `L = d * f` does.

```bash
uv run python micrograd/expr.py
```

*You should see* the four counts, the four node values, the graph as indented text, and `drawn: micrograd/expr.svg`. On Homer, `WORD2 = jove`:

```text
chapter 2 per 1,000 words: the 49.9  jove 2.0
chapter 7 per 1,000 words: the 68.0  jove 3.7
p1 = 3393.20   p2 = 7.40   d = 3400.60   L = 34.0060
L = 34.0060   grad 0.0000   (*)
    d = 3400.6000   grad 0.0000   (+)
        p1 = 3393.2000   grad 0.0000   (*)
            a1 = 49.9000   grad 0.0000
            b1 = 68.0000   grad 0.0000
        p2 = 7.4000   grad 0.0000   (*)
            a2 = 2.0000   grad 0.0000
            b2 = 3.7000   grad 0.0000
    s = 0.0100   grad 0.0000
drawn: micrograd/expr.svg
```

Open `micrograd/expr.svg`. It reads left to right from five leaves (`a1`, `b1`, `a2`, `b2`, `s`) through two `*` nodes and a `+` to `d`, then one more `*` to `L`: the same shape as his `L = (a*b + c) * f`, with your corpus's numbers in the boxes. The order of the children under each node in the text version can differ from Homer's; a `set` has no order. Paste the three value lines under `## Step 3` in `evidence/T08.md`.

*If it broke* with `could not draw (is graphviz installed with brew?): ExecutableNotFound`, the `dot` program is missing; the text graph is your picture today. *If it broke* with `TypeError: Value.__init__() got an unexpected keyword argument 'label'`, your `__init__` is missing `label=""`; he adds it while building the drawing, so the boxes have names.

**Step 4. The five gradients, by hand, committed before the engine checks them.**

`L = (a1·b1 + a2·b2) · s`. Under `## Step 4` in `evidence/T08.md`, work from `L` back to the leaves the way he did at 00:32:10, one line per node, each with the number and the one-line reason:

- `dL/dL = 1`
- `dL/dd` and `dL/ds`: `L = d * s`, so each gets the other one's value
- `dL/dp1` and `dL/dp2`: `d = p1 + p2`, so a `+` passes `dL/dd` straight through
- `dL/da1 = b1 · dL/dp1`, `dL/db1 = a1 · dL/dp1`, and the same for `a2`, `b2`

Nine numbers, from your own four counts. Then commit the file by itself:

```bash
git add evidence/T08.md
git commit -m "T08: the gradients of my expression by hand"
```

I will check your commit timestamps: the hand page has to exist before `expr.py grads` does.

**Step 5. Let the script check your page.**

```bash
uv run python micrograd/expr.py grads
```

It sets every gradient by hand in code, from the output back, exactly the lines you just did on paper, then checks each leaf by nudging it and re-running `L` with plain floats, his way. *You should see* five lines where the two columns agree to four decimals, and `drawn: micrograd/expr-grads.svg`. On Homer:

```text
dL/da1  by hand     0.6800   by nudging     0.6800
dL/da2  by hand     0.0370   by nudging     0.0370
dL/db1  by hand     0.4990   by nudging     0.4990
dL/db2  by hand     0.0200   by nudging     0.0200
dL/ds  by hand  3400.6000   by nudging  3400.6000
```

Hold your page next to it. `dL/da1` is `b1 · s`: one more `the` per 1,000 words in chapter 2 raises `L` by chapter 7's `the` rate times the scale. `dL/ds` is `d` itself, the biggest number on the page, because `s` multiplies everything. Where your page and the script disagree, do not change the page: write the correct number next to the wrong one under `## Step 4`, with one line on which rule you applied wrongly. Paste the five lines under `## Step 5`. Open `micrograd/expr-grads.svg`: every box now shows its `grad`.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T08.md` (the questions are below) and run `uv run python scripts/check_evidence.py T08` until it says `all slots filled`. Then:

```bash
git add micrograd/engine.py micrograd/expr.svg micrograd/expr-grads.svg scratch/t08-video.py evidence/T08.md pyproject.toml uv.lock
git commit -m "T08: Value with + and *, my counts as an expression graph, gradients by hand checked by nudging"
git push -u origin dev/value
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the gradient that took you longest.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`micrograd/engine.py` (a working `Value` with `+`, `*`, `grad` and a label) + `micrograd/expr.svg` and `micrograd/expr-grads.svg` (or the text graphs pasted in the evidence file if `dot` would not install) + `scratch/t08-video.py` + `evidence/T08.md` with every slot filled, the Step 4 page committed before Step 5 ran.

**Reflection Questions** (answer under `## Reflection` in `evidence/T08.md`)

1. Paste your Step 5 lines next to your Step 4 page. Pick `dL/da1` and write the chain of two factors your page multiplied to get it, with your numbers, naming the node each factor came from. Then say what `_prev` held for your node `d`, and how `draw_dot` found `a1` from `L` without being told about it.

   *How to get it:* `dL/da1 = dL/dp1 × dp1/da1 = (dL/dd × dd/dp1) × b1`; on Homer `0.01 × 1 × 68.0 = 0.68`. `d._prev` is the set `{p1, p2}`; `draw_dot` starts at `L`, reads `L._prev`, and keeps reading `_prev` until it reaches nodes with none.

2. Paste the seven numbers you wrote during the video for his `L = (a*b + c) * f`. Say which of them you got wrong before he showed it, if any, and which rule you misapplied. If none, say which one you were least sure of and why `dL/dc` came out as it did.

   *How to get it:* his numbers are `dL/dL = 1`, `dL/dd = -2`, `dL/df = 4`, `dL/dc = -2`, `dL/de = -2`, `dL/da = 6`, `dL/db = -4`. `dL/dc` equals `dL/dd` because `d = e + c` and a `+` passes the gradient through unchanged.

3. Your `L` is `0.01` times a two-word dot product. Say what `dL/ds = 3400.6` means in words, and why one gradient on your page is a thousand times bigger than the others even though `s` is the smallest number in the expression.

   *How to get it:* `dL/ds` is how much `L` moves when `s` moves by 1, and `L = d × s` moves by `d` per unit of `s`. The size of a gradient is set by what multiplies the input, not by the input's own size.
