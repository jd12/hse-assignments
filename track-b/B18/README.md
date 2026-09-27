# B18 · Gradients and Gradient Descent

**Meetings:** D32–D33 · **Points:** 15 pts


**Watch — 35 min**
Day 1 — 24 min
[Khan Academy, Multivariable derivatives: Partial derivatives, introduction](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives) · whole video (10:56, 11m)
[Khan Academy, Multivariable derivatives: Partial derivatives and graphs](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives) · whole video (6:54, 7m)
[Khan Academy, Multivariable derivatives: Gradient and graphs](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives) · whole video (6:11, 6m)
 · This series is by Grant Sanderson, who makes 3Blue1Brown. Paper on the desk.

Day 2 — 11 min
[Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=852s) · segment 00:14:12 → 00:19:09, "derivative with multiple inputs" (5m)
[Khan Academy, Multivariable derivatives: Gradient and contour maps](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives) · whole video (6:17, 6m)
 · `scratch/b18-video.py` open for Karpathy. Your Day 1 `calc/trajectory.png` open for Khan.

**During the video**

**Partial derivatives, introduction · compute the other one.** He works one partial derivative of his example function in full. Pause the moment he finishes it and compute the *other* partial, with respect to the other variable, yourself, on paper. Then check it against what he does next.

**Partial derivatives and graphs · draw the slice.** When he cuts the surface with a plane that holds one variable still, draw it: the surface, the plane, and the curve where they meet. Then draw the tangent line to that curve. Label it "∂f/∂x is the slope of this line".

**Gradient and graphs · draw the arrow.** When he shows the gradient as a vector in the input plane, draw a bowl, a point on its side, and the gradient at that point as an arrow on the floor, not on the surface. Write next to it: "entries are the partials; points uphill; minus it points downhill".

**Karpathy 00:14:12 → 00:19:09 · type and run.** **Type everything he types into `scratch/b18-video.py`, and run it.** He builds an expression of three inputs, nudges one input at a time by `h`, and reads off each slope. Every time he gets a slope, print next to it what the rule says it should be: for a product, the other factor; for a sum, 1. They should agree to the digits `h` allows.

**Gradient and contour maps · compare with your picture.** He shows the gradient crossing contour lines at right angles. Stop when he does and look at your Day 1 `calc/trajectory.png`. Your first step leaves the start at right angles to the contour it is on; `set_aspect("equal")` in the plotting code is what lets you see that, since with unequal axes a right angle does not look like one. Circle that step on a printout or a screenshot and commit it as `calc/contour-note.png`.

**Notes**

The gradient `∇L` is a vector whose entries are the partial derivatives. That is it. It points in the direction of steepest increase, so `-∇L` points steepest downhill.

The update rule, which you will now see every week for the rest of the course:
```
w ← w − lr · ∂L/∂w
b ← b − lr · ∂L/∂b
```

**Update both from the same old point.** If you overwrite `w` and then use the new `w` to compute the `b` update, you have written a different algorithm. Compute both slopes first, then apply both moves. The code below does it in one line on purpose, and Step 6 shows you what the other version does.

**Your surface is a long, thin valley, and that is the lesson.** The slope `w` multiplies numbers up to 10; the intercept `b` multiplies 1. So the loss is very steep across `w` and very shallow along the valley floor. A learning rate small enough not to blow up across the valley is tiny along it. Expect a sharp zig-zag in the first few steps and then a long crawl.

**The learning-rate trap.** Too small and you crawl; too large and you overshoot the valley and bounce outward, the loss increasing every step until it is `inf`. You are required to demonstrate this: keep the diverging run's output and its plot.

`plt.contour` with evenly spaced levels leaves the bottom of the bowl blank; the code below spaces them logarithmically.

**Walkthrough — Gradient descent on two parameters**

**Day 1 starts here.**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:
```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/gradient-descent
```
*If B17 is not approved yet:* `git switch dev/derivative && git switch -c dev/gradient-descent`, because today imports `calc/windows.py`.
```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-track, not main
bash scripts/start-entry.sh
```
Under the timestamp, write the checklist for the whole assignment, and tick what you finish each day:
```markdown
- [ ] Day 1: Khan partial derivatives introduction, computing the other partial
- [ ] Day 1: Khan partial derivatives and graphs, drawing the slice
- [ ] Day 1: Khan gradient and graphs, drawing the arrow
- [ ] Day 1: Steps 2–4, the gradient by hand, checked by nudging, and the first contour run
- [ ] Day 2: Karpathy micrograd 00:14:12 → 00:19:09, typing along in scratch/b18-video.py
- [ ] Day 2: Khan gradient and contour maps, circling my first step
- [ ] Day 2: Steps 5–7, the diverging run, the wrong update order, SURFACE.md
- [ ] Day 2: Extension, the learning-rate sweep
- [ ] Push each day; one PR
```

**Step 2. The gradient, by hand, before the code.**

The model is now a line with an intercept: `y ≈ w · x + b`, on the same windows as B17. The loss is `L(w, b) = mean((w·x + b − y)²)`. On paper, in `calc/GRADIENT.md`, write `∂L/∂w` and `∂L/∂b`. Treat `b` as a constant for the first and `w` as a constant for the second, exactly as on your Khan slice. Commit it: `git add calc/GRADIENT.md && git commit -m "B18: gradient by hand"`. I will check your commit timestamps.

**Step 3. The gradient in code, checked by nudging.**

Create `calc/gradient_descent.py`:

```python
import sys
import numpy as np
import matplotlib.pyplot as plt
from windows import load_windows

x, y = load_windows()

def loss(w, b):
    return np.mean((w * x + b - y) ** 2)

def grad(w, b):
    err = w * x + b - y
    return np.mean(2 * err * x), np.mean(2 * err)     # (dL/dw, dL/db)

def descend(lr, steps, w=0.0, b=0.0):
    path = [(w, b)]
    for _ in range(steps):
        dw, db = grad(w, b)                  # both slopes from the SAME old point
        w, b = w - lr * dw, b - lr * db      # then both moves at once
        path.append((w, b))
    return np.array(path)
```

Then create `calc/check_grad.py`:

```python
from gradient_descent import loss, grad

h = 1e-6
for w, b in ((0.0, 0.0), (0.3, 0.5), (1.0, -1.0)):
    dw, db = grad(w, b)
    nw = (loss(w + h, b) - loss(w, b)) / h
    nb = (loss(w, b + h) - loss(w, b)) / h
    print(f"({w:4}, {b:4})  dL/dw rule {dw:10.5f} nudge {nw:10.5f}   dL/db rule {db:9.5f} nudge {nb:9.5f}")
```

Run it: `uv run python calc/check_grad.py`.

*You should see* three rows where each rule agrees with its nudge to about five significant figures, and where `dL/dw` is several times larger than `dL/db` at the same point. That ratio is the long thin valley from the Notes, measured. If the rule disagrees with the nudge, your code and your `GRADIENT.md` should both be checked; fix the code, and write the correction under the original on the page.

**Step 4. The first run, on a contour map.**

Add to `gradient_descent.py`:

```python
def contour(path, filename, title):
    w_star, b_star = np.polyfit(x, y, 1)             # the answer, found without descending
    ws = np.linspace(min(0, w_star) - 0.3, max(0, w_star) + 0.3, 200)
    bs = np.linspace(min(0, b_star) - 0.6, max(0, b_star) + 0.6, 200)
    W, B = np.meshgrid(ws, bs)
    L = np.mean((W[..., None] * x + B[..., None] - y) ** 2, axis=-1)
    plt.figure(figsize=(7, 6))
    plt.contour(W, B, L, levels=np.logspace(np.log10(L.min()), np.log10(L.max()), 25))
    plt.plot(path[:, 0], path[:, 1], ".-", color="red", markersize=3, lw=0.8)
    plt.plot(w_star, b_star, "k*", markersize=12)
    plt.xlim(ws[0], ws[-1]); plt.ylim(bs[0], bs[-1])
    plt.gca().set_aspect("equal")                   # so a right angle looks like one
    plt.xlabel("w (slope)"); plt.ylabel("b (intercept)"); plt.title(title)
    plt.savefig(filename, dpi=110); plt.close()

if __name__ == "__main__":
    lr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
    path = descend(lr, 2000)
    for k in (0, 1, 2, 3, 10, 100, 500, 2000):
        w, b = path[k]
        print(f"step {k:5d}  w {w: .5f}  b {b: .5f}  loss {loss(w, b):.5f}")
    print("polyfit     w {:.5f}  b {:.5f}".format(*np.polyfit(x, y, 1)))
    contour(path, "calc/trajectory.png", f"lr = {lr}")
```

`np.polyfit` is the closed-form answer, used only to put a star at the bottom of the bowl. With 41 parameters in B22 there is no closed form, which is why you are learning to walk.

Run it: `uv run python calc/gradient_descent.py`.

*You should see*, in the printout, `w` jump well past its final value on step 1 and come back on step 2, while `b` barely moves. Then `w` slowly drifts back down while `b` climbs, and by step 2,000 both match the `polyfit` line to four or five decimals. In `calc/trajectory.png`, the contours are long thin ellipses tilted across the plot; the red path makes one or two sharp zig-zags across the valley and then crawls along its floor to the star. Your `b` at the end is the line's intercept: how many distinct words, in hundreds, the line predicts for a window of zero words, which is not zero. That is the first sign a straight line is the wrong shape for this data.

Commit and push: `git add calc/ && git commit -m "B18: gradient checked, first contour run" && git push -u origin dev/gradient-descent`. Open the pull request now, **jd12** as reviewer, and leave it open. Sign off the log: `bash scripts/sign-off.sh`, then `git add logs && git commit && git push`.

**Day 2 starts here.** Open a new entry with `bash scripts/start-entry.sh` in the log repo and copy over what is left of the checklist. Watch the Day 2 videos first.

**Step 5. The diverging run.**

Run the same script with a learning rate past the edge:

```bash
uv run python calc/gradient_descent.py 0.03
```

Then add `calc/diverge.py`, which plots loss per step for the good run and the bad one:

```python
import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import descend, loss

for lr, color in ((0.02, "tab:blue"), (0.03, "tab:red")):
    path = descend(lr, 60)
    plt.semilogy([loss(w, b) for w, b in path], color=color, label=f"lr = {lr}")
    print(f"lr {lr}: loss at steps 0, 5, 10, 30, 60 =",
          [f"{loss(*path[k]):.3g}" for k in (0, 5, 10, 30, 60)])
plt.xlabel("step"); plt.ylabel("loss (log scale)"); plt.legend()
plt.savefig("calc/diverge.png", dpi=110)
```

*You should see* `lr 0.02` fall and flatten, and `lr 0.03` climb from the very first step, in a nearly straight line on the log plot, which means the loss is multiplying by about the same factor every step. In the printout of `gradient_descent.py 0.03`, `w` swings to either side of its final value and further out each time, and by step 500 the numbers run to dozens of digits; by step 2,000 the loss is `inf` with a `RuntimeWarning: overflow`. That warning is the divergence, not a bug in your code. The contour picture from that run is overwritten on `trajectory.png`, so run `uv run python calc/gradient_descent.py` once more afterwards to put the good one back. Keep both printouts; they are required.

**Step 6. The wrong update order.**

Copy `descend` into `calc/sequential.py` and change the update so `b` uses the new `w`:

```python
from gradient_descent import grad

w, b, lr = 0.0, 0.0, 0.02
for k in range(3):
    dw, _ = grad(w, b)
    w = w - lr * dw
    _, db = grad(w, b)          # computed AFTER w already moved
    b = b - lr * db
    print(f"sequential step {k + 1}  w {w: .5f}  b {b: .5f}")
```

*You should see* the same `w` on step 1 as your Step 4 run and a different `b`, quite possibly with the opposite sign, because `b` was pushed by a slope measured at a point the algorithm was never actually at. The two runs are different algorithms from the first step on.

**Step 7. Half a page on the shape.**

In `calc/SURFACE.md`, half a page, in your own words: what the trajectory's shape tells you about the surface. Use your own numbers: how far step 1 overshot in `w`, how many steps the crawl took, what the ratio of `dL/dw` to `dL/db` was in Step 3, and what the diverging run's loss was doing by step 30.

**Extension — How fast, and where does it break?** *(assigned)*

Before you run anything, write in your log which learning rate you expect to reach the bottom fastest, and the smallest learning rate you expect to diverge. Commit the log. I will check your commit timestamps.

Create `calc/lr_sweep.py`:

```python
import numpy as np
from gradient_descent import x, y, loss, grad

w_star, b_star = np.polyfit(x, y, 1)
floor = loss(w_star, b_star)

def steps_to_floor(lr, cap=20000, tol=1e-4):
    w = b = 0.0
    for k in range(1, cap + 1):
        dw, db = grad(w, b)
        w, b = w - lr * dw, b - lr * db
        L = loss(w, b)
        if not np.isfinite(L) or L > 1e6:
            return f"diverged at step {k}"
        if L - floor < tol:
            return k
    return f"not there after {cap}"

print(f"floor (polyfit loss) {floor:.5f}")
for lr in (0.001, 0.005, 0.01, 0.02, 0.025, 0.027, 0.028, 0.03):
    print(f"lr {lr:<6} {steps_to_floor(lr)}")
```

*You should see* the steps needed fall roughly in proportion as the learning rate grows, 0.001 taking several thousand and 0.02 a few hundred, then a best rate somewhere just under the edge, then one that is slower again because it zig-zags across the valley on nearly every step, and then divergence. The edge is close to 0.027 or 0.028 for everyone, because it depends on your window lengths and not on your corpus. Where exactly your best rate and your first diverging rate fall is yours.

Write `calc/FINDINGS-b18.md`: the sweep table; your baseline, which is `lr 0.02` from Step 4; your predictions copied unchanged with a sentence each on whether they held; and where it got worse, named with numbers: the learning rate just below the edge that was slower than a smaller one, and why the zig-zag costs steps.

Commit, push, and sign off:

```bash
git add calc/ scratch/b18-video.py
git commit -m "B18: diverging run, update order, learning-rate sweep"
git push
```

The pull request you opened on Day 1 picks this up; one PR, not two. Say in the PR body where this is weakest.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

Push again each day; one PR, not two; one log entry per meeting.

**Deliverable**
`calc/gradient_descent.py` (no autograd, no optimizer library) + `calc/trajectory.png` (trajectory on a contour plot) + `calc/diverge.png` (a diverging learning rate) + `calc/SURFACE.md` (half a page) + `calc/GRADIENT.md` + `calc/check_grad.py` + `calc/sequential.py` + `calc/lr_sweep.py` + `calc/FINDINGS-b18.md` + `calc/contour-note.png` + `scratch/b18-video.py`.

**Reflection Questions**

1. Paste the step 0 to step 3 lines from your Step 4 run and the first row of `check_grad.py`. Using your own `dL/dw` at `(0, 0)` and your learning rate, compute by hand where `w` should be after step 1, and check it against the printout. Then say, from the ratio of your two slopes, why `b` hardly moved while `w` overshot.

2. Paste the `lr 0.03` line from `diverge.py` and the first four steps of `gradient_descent.py 0.03`. Describe, step by step with your numbers, what happens to `w` and to the loss, and name the factor the loss is multiplying by each step once the climb is steady. Say what one step would have had to do differently for the run to stay stable.

3. Paste your `lr_sweep.py` table beside your committed predictions. Name the learning rate that was fastest, the one just above it that was slower, and the first one that diverged. Say which of your two predictions was further off, and what about your `trajectory.png` would have told you the answer before you ran the sweep.
