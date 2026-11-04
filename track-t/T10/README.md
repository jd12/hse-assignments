# T10 · `backward()` Finds the Order Itself, and PyTorch Agrees With Your Engine

**Meetings:** D33 · **Points:** 15 pts

**Watch — 26 min**
[Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4652s) · 01:17:32 → 01:43:55, "implementing the backward function for a whole expression graph", "fixing a backprop bug when one node is used multiple times", "breaking up a tanh, exercising with more operations" and "doing the same thing but in PyTorch: comparison" (26m)
 · `micrograd/engine.py` and `scratch/t10-video.py` (the same four import lines) open, terminal in the repo root. Step 2 installs PyTorch, a large download; start it before you press play.

**During the video**

**Type everything he types, and run it.** Engine changes go into `micrograd/engine.py`; the expressions and the PyTorch cell go into `scratch/t10-video.py`.

**01:17:32 → 01:22:28.** He writes the ordering itself, a topological sort, and a `backward` method that sets `self.grad = 1.0` and walks the sorted list in reverse calling every `_backward`. That replaces T09 Step 4's hand-ordered calls.

**01:22:28 · compute before he does.** He finds a bug in his own engine when a node is used twice: `b = a + a`. **Before he fixes it, pause** and write under `## During the video` in `evidence/T10.md` what `a.grad` should be and what his engine printed instead, and which line of `__add__` is the reason. Then type the fix. Of everything in micrograd, that one fix causes the most lost evenings.

**01:27:05 → 01:39:31.** He adds the operations a real expression needs: numbers on the left of `+` and `*`, `exp`, powers, division, negation and subtraction, each with its own `_backward`. Then he rebuilds `tanh` out of `exp` and division instead of as one node. **Before he runs backward on the rebuilt version**, write under `## During the video` whether the input's gradient will be the same as with the one-node `tanh`, and why.

**01:39:31 → 01:43:55.** He builds the same neuron in PyTorch and prints its gradients. Type it exactly, including the `.double()`, and run it. Step 4 does this on your neuron.

**Notes**

**Topological sort answers one question: in what order is it safe to call the `_backward`s?** A node cannot pass gradient to its inputs until every node that uses it has passed gradient to it. Sorting the graph so that every node comes after its children, then walking the list backwards, guarantees that. His `build_topo` is a depth-first walk with a `visited` set, and `topo.append(v)` comes **after** the loop over `v._prev`. Put it before the loop and every gradient comes out `0.0` with no error.

**`+=` not `=`.** When a node is used twice, gradient arrives from two places and both have to be kept. With `=`, the second arrival overwrites the first. The Extension breaks exactly one of the two lines in `__add__` and shows what the `a + a` check cannot see.

**`1 + Value(2.0)` fails** with `TypeError: unsupported operand type(s) for +: 'int' and 'Value'`. Python tries `int.__add__` first, which has never heard of `Value`, then looks for `Value.__radd__`. That is the fix he types, and it matters more than it looks: `sum(list_of_values)` starts from the integer `0`, so every loss written with `sum` needs `__radd__`.

**PyTorch defaults to 32-bit floats; your engine uses Python's 64-bit floats.** In 32 bits, two correct engines agree to about seven digits and sometimes miss the sixth decimal. That is why he writes `.double()`, and why `vs_torch.py` builds its tensors with `dtype=torch.double`.

**`uv add torch` is large.** On a Mac it gets the CPU build, which is the right one; nothing this year uses a GPU. On Windows or Linux the `pyproject.toml` from T01 points it at the CPU index. If the download is over a gigabyte, it is pulling the CUDA build: stop and tell me.

**Walkthrough — the whole graph at once, then a second opinion**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T09 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/backward
```

*If T09 is not approved yet:* `git switch dev/neuron && git switch -c dev/backward`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Step 2: uv add torch, CPU build, imports
- [ ] micrograd 1:17:32 → 1:43:55 watched, backward() and the new ops typed into engine.py, two predictions written before he ran
- [ ] Step 3: engine.py checked against where he stops, a + a gives 2.0, neuron.py backward matches T09
- [ ] Step 4: vs_torch.py, three engines agree
- [ ] Extension: one = in __add__, and what the a + a check misses
- [ ] Fill every slot in evidence/T10.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Install PyTorch.**

```bash
uv add torch
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

*You should see* a version number and `False`. `False` is correct: CUDA is NVIDIA's GPU platform, and your laptop is not running it. The version ends in `+cpu` on some machines and in nothing on a Mac; both are the CPU build. *If it broke* with `uv add` sitting on a download with a size in the gigabytes, press Ctrl-C and tell me. Then watch the segment and type along.

**Step 3. Check `engine.py` against where he stops.**

At 01:43:55 your `Value` class has, on top of T09's: a `backward` method with `build_topo` inside it; `__pow__` (a plain number as the exponent, with his `assert`), `exp`, `__rmul__`, `__radd__`, `__truediv__`, `__neg__` and `__sub__`; and `+=` in every `_backward`. If he did not type `__radd__` in this segment, add it anyway; the Notes say why. Three checks:

```bash
uv run python -c "import sys; sys.path.insert(0, 'micrograd'); from engine import Value; a = Value(3.0); b = a + a; b.backward(); print(a.grad)"
uv run python -c "import sys; sys.path.insert(0, 'micrograd'); from engine import Value; a = Value(3.0); b = Value(4.0); print(1 + a, 2 * a, a - b, a / b, a ** 2, (-a).data, sum([a, b]))"
uv run python micrograd/neuron.py backward
```

*You should see* `2.0` from the first; `Value(data=4.0) Value(data=6.0) Value(data=-1.0) Value(data=0.75) Value(data=9.0) -3.0 Value(data=7.0)` from the second; and from the third, T09's two lines and the same ten gradients as T09 Step 4 printed, now from one call to `o.backward()` instead of five hand-ordered calls. Paste the `2.0` and the ten gradient lines under `## Step 3` in `evidence/T10.md`.

*If it broke* with `1.0` from the first check, one of the two lines in `__add__`'s `_backward` still uses `=`. A `TypeError` from the second names the method you are missing. *If it broke* with every gradient `0.0000` from the third, `topo.append(v)` is before the loop in `build_topo`, or `self.grad = 1.0` is missing from `backward`.

**Step 4. Your neuron three ways.**

`micrograd/vs_torch.py` came with the template; open it and read the top. It builds T09's neuron three times: your engine with `tanh` as one node, your engine with `tanh` built from `exp` and division, and PyTorch, which has never seen your `Value` class. Before you run it, write under `## Step 4` whether the `from exp` column will match the `tanh node` column exactly or only to some number of decimals, and why. Then:

```bash
uv run python micrograd/vs_torch.py
```

*You should see* six rows, every one `True`. On Homer:

```text
           tanh node     from exp        torch   all within 5e-7
o          -0.406991    -0.406991    -0.406991   True
do/db       0.834358     0.834358     0.834358   True
do/dw1     -1.067978    -1.067978    -1.067978   True
do/dw2     -0.300369    -0.300369    -0.300369   True
do/dx1      0.417179     0.417179     0.417179   True
do/dx2     -0.250307    -0.250307    -0.250307   True
```

Three ways of computing the same six numbers, two of them yours and one of them a library used by every lab in the world, agreeing to six decimals. That is the reason to trust any of them. The `from exp` column is a longer graph for the same function, `exp`, two `+`s, a division and the `**-1` inside it, so it agrees to about fifteen decimals rather than exactly; the table rounds to six. Paste the table under `## Step 4`.

*If it broke* with `False` in one row and `True` elsewhere, a single `_backward` has a wrong local derivative; the row names the variable, and the column says which graph. `False` in the whole `from exp` column with the other two agreeing is a bug in one of the new operations, since the `tanh node` column never touches them. `False` everywhere at the sixth decimal means `dtype=torch.double` was edited out.

**Extension — What the `a + a` check cannot see** *(assigned)*

`__add__`'s `_backward` has two lines, `self.grad += ...` and `other.grad += ...`. Change the first one only, to `self.grad = 1.0 * out.grad`, leaving the second as `+=`. Before you run anything, write under `## Extension` in `evidence/T10.md` what the `a + a` check will print now, and which of `vs_torch.py`'s three columns will go wrong. Commit the file. I will check your commit timestamps. Then run both:

```bash
uv run python -c "import sys; sys.path.insert(0, 'micrograd'); from engine import Value; a = Value(3.0); b = a + a; b.backward(); print(a.grad)"
uv run python micrograd/vs_torch.py
```

*You should see* the `a + a` check still print `2.0`, and the `from exp` column wrong in every gradient row while the other two columns agree. On Homer:

```text
           tanh node     from exp        torch   all within 5e-7
o          -0.406991    -0.406991    -0.406991   True
do/db       0.834358     0.241349     0.834358   False
do/dw1     -1.067978    -0.308927    -1.067978   False
do/dw2     -0.300369    -0.086886    -0.300369   False
```

The check passed because in `a + a`, `self` and `other` are the same object: the `=` set `a.grad` to `1.0` and the untouched `+=` on the next line added the second `1.0`, so `2.0` came out by luck. The `from exp` graph uses `e` in two different `+` nodes, `e - 1` and `e + 1`, so `e` is `self` in both and the second arrival overwrote the first. The `tanh node` graph reuses nothing, so it never noticed. Write where it got worse under `## Extension`: the factor by which `do/dw1` in the `from exp` column shrank, and one sentence on what the `a + a` check measured and what it should have. **Then put the `+=` back** and run `vs_torch.py` to six `True`s before you commit anything else.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T10.md` (the questions are below) and run `uv run python scripts/check_evidence.py T10` until it says `all slots filled`. Then:

```bash
git add micrograd/engine.py scratch/t10-video.py evidence/T10.md pyproject.toml uv.lock
git commit -m "T10: backward() with topological sort, the full op set, my neuron matches PyTorch three ways"
git push -u origin dev/backward
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the operation whose `_backward` you are least sure of.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`micrograd/engine.py` with a complete `backward()` and the full op set, `+=` everywhere + `scratch/t10-video.py` + `evidence/T10.md` with every slot filled, with `torch` in `pyproject.toml`. The engine is finished; T11 builds a network out of it.

**Reflection Questions** (answer under `## Reflection` in `evidence/T10.md`)

1. Paste the six rows from `vs_torch.py`. Pick `do/dw1` and write out, with your numbers, the chain of local derivatives your engine multiplied to get it, from `o` back to `w1`, and name the line of `engine.py` each factor comes from. Then say which of those factors PyTorch computed for you that T07's `slope_by_rule` had to be written by hand for.

   *How to get it:* `do/dw1 = (1 - o²) × 1 × 1 × x1`: the `_backward` in `tanh`, then the one in `__add__` twice, then the one in `__mul__` with `other.data` as `x1`. On Homer `0.834358 × (-1.28) = -1.067978`. T07's rule was derived on paper for one specific function; PyTorch and your engine derive it from the graph, one local rule per operation.

2. Paste what you wrote during the video for `b = a + a` and the `2.0` from Step 3. Say, in terms of the two lines inside `__add__`'s `_backward`, what `=` did to the first contribution when `self` and `other` were the same object, and why his engine printed `1.0` rather than `0.0`.

   *How to get it:* with `=` on both lines, the first line set `a.grad` to `1.0` and the second set it to `1.0` again; nothing was added, so the answer was one path's worth instead of two. `0.0` would have needed both lines to be skipped.

3. Paste your Extension table beside your committed predictions. Say what the `a + a` check measured, what it should have measured, and write the one-line check that would have caught a `=` on `self.grad` alone.

   *How to get it:* `a + a` has `self` and `other` as the same `Value`, so a `=` on one line is repaired by the `+=` on the other. A check needs a node reused as `self` in two different `+` nodes: `a = Value(3.0); b = (a + 1) + (a + 2); b.backward(); print(a.grad)` should print `2.0` and prints `1.0` with the `=` in place.
