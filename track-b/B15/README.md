# B15 · Matrices Are Functions

**Meetings:** D29 · **Points:** 15 pts

**Watch — 30 min**
[3Blue1Brown, Essence of Linear Algebra, Ch. 3: Linear transformations and matrices](https://www.youtube.com/watch?v=kYB8IZa5AuE) · whole video (10:59, 11m)
[Jon Krohn, Machine Learning Foundations: Affine Transformations, Topic 27](https://www.youtube.com/watch?v=6H-fSbV-Jzw) · whole video (18:53, 19m)
 · Paper for 3Blue1Brown. `scratch/b15-video.py` open for Krohn.

**During the video**

**Ch. 3 · compute before he does.** He works an example in which he tells you where î and ĵ land and then asks where some other vector ends up. Pause before he answers. Write that vector as `x · (where î lands) + y · (where ĵ lands)`, with his numbers, and do the arithmetic on paper. Then unpause and check. That one line of arithmetic is the whole chapter, and it is the line Step 4 checks in code on your own chapter vectors.

**Krohn · type and run.** He demonstrates scaling, shear and rotation with NumPy on plotted vectors. **Type everything he types in NumPy into `scratch/b15-video.py`, and run it** with `uv run python scratch/b15-video.py`. For every matrix he applies, add a line that prints the matrix times `[1, 0]` and times `[0, 1]`, and check the two results against the matrix's columns before he tells you what the transformation does. If he plots with a helper function of his own, type it; if you would rather not, use the `quiver` call from your B13 `vectors.py` with `angles="xy", scale_units="xy", scale=1`. Krohn's title says **affine**: an affine map is a linear one followed by a shift. Hold on to that; the extension catches one on your corpus.

**Notes**

This is the single most important idea in B13–B16, and it is one sentence: a matrix is a function that moves space, and its columns are where the basis vectors land.

To find the matrix for a transformation, ask where î = (1, 0) goes and where ĵ = (0, 1) goes. Those two answers, written as columns, *are* the matrix. Written as rows, they are the transpose, and the picture you get is a different transformation. This is the most common mistake on today's hand page, and your `NOTATION.md` from B13 exists for it.

"Linear" has a precise meaning: grid lines stay parallel and evenly spaced, and the origin does not move. Equivalently, `f(u + v) = f(u) + f(v)` and `f(c·u) = c·f(u)`. A transformation that shifts everything two units right is not linear, because the origin moves. The extension tests four functions you might apply to a count vector and one of them is exactly that.

`np.column_stack([a, b])` puts `a` and `b` in as columns. `np.array([a, b])` puts them in as rows. They print differently and they are transposes of each other; when a result looks rotated the wrong way, check which one you used.

The L1 norm is the sum of absolute values; the L2 norm is the straight-line length. They come back in B43 when you look at gradient clipping. Step 3 shows you a transformation that keeps one of them and changes the other, so do not treat them as filler.

**Walkthrough — Three matrices, by hand and then on your chapters**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:
```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/transforms
```
*If B14 is not approved yet:* branch from the newest of your open branches, since today needs `linear/chapter_vectors.py`: `git switch dev/matmul && git switch -c dev/transforms`.
```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-track, not main
bash scripts/start-entry.sh
```
Under the timestamp, write today's checklist:
```markdown
- [ ] Watch Essence of Linear Algebra Ch. 3, computing his example before he does
- [ ] Watch Krohn Topic 27, typing along in scratch/b15-video.py
- [ ] Step 2: the three matrices by hand, committed before any code
- [ ] Steps 3–4: landing.py checks my matrices and moves my chapter vectors
- [ ] Extension: which of four functions of a count vector are linear
- [ ] Push dev/transforms and open the PR
```

**Step 2. Three matrices, by hand, committed first.**

By hand, no code. In `linear/MATRICES.md` (typed) or `linear/matrices.jpg` (photographed), give the 2×2 matrix for:

| | Transformation | Write down |
|---|---|---|
| (a) | 90° counterclockwise rotation | where î lands, where ĵ lands, the matrix, what happens to areas |
| (b) | horizontal shear that sends ĵ to (1, 1) and leaves î alone | the same four things |
| (c) | scale by 3 in x and by 0.5 in y | the same four things |

One sentence each on areas. That sentence is the bridge to determinants in B16; do not skip it.

Commit this page by itself before you write any code: `git add linear/MATRICES.md` (or the `.jpg`) and `git commit -m "B15: three matrices by hand"`. Step 3 checks it, and a hand page committed after the check is a copy of the check. I will check your commit timestamps.

**Step 3. Check your matrices, and move your chapters with them.**

Create `linear/landing.py`. Fill in the three matrices from your page, as rows of the matrix, which is how NumPy is written:

```python
import numpy as np
import matplotlib.pyplot as plt
from chapter_vectors import V2, WORD1, WORD2

c = np.cos(np.pi / 4)
MATRICES = {
    "rotate 90": np.array([[0.0, 0.0], [0.0, 0.0]]),   # fill in from your page
    "shear":     np.array([[0.0, 0.0], [0.0, 0.0]]),
    "scale":     np.array([[0.0, 0.0], [0.0, 0.0]]),
    "rotate 45": np.array([[c, -c], [c, c]]),          # given, not on your page
}

e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
U = V2 / np.linalg.norm(V2, axis=1)[:, None]   # each chapter scaled to length 1, (10, 2)
ratio = V2[:, 1] / V2[:, 0]
lo, hi = ratio.argmin(), ratio.argmax()         # your two most different chapters

def angle(a, b):
    return np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, M) in zip(axes, MATRICES.items()):
    print(f"{name:10} i lands at {M @ e1}   j lands at {M @ e2}")
    moved = U @ M.T                              # every chapter through M at once
    v, mv = U[2], moved[2]
    print(f"{'':10} chapter 2: L2 {np.linalg.norm(v):.4f} -> {np.linalg.norm(mv):.4f}"
          f"   L1 {np.abs(v).sum():.4f} -> {np.abs(mv).sum():.4f}"
          f"   angle(ch{lo}, ch{hi}) {angle(U[lo], U[hi]):.3f} -> {angle(moved[lo], moved[hi]):.3f} degrees")
    for row, color in ((U, "lightgray"), (moved, "tab:blue")):
        ax.quiver(np.zeros(10), np.zeros(10), row[:, 0], row[:, 1],
                  angles="xy", scale_units="xy", scale=1, color=color)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal"); ax.grid(True)
    ax.set_title(f"{name}  (gray: {WORD1}/{WORD2} chapters before)")
plt.savefig("linear/transforms.png", dpi=110)
```

Run it: `uv run python linear/landing.py`.

*You should see*, for each matrix, the two landing points equal to your two columns. If they are not, the page from Step 2 is wrong or you typed columns as rows; fix the code, not the committed page, and say which it was in your log. In the picture, the gray fan of your ten chapter arrows (unit length, all pointing nearly the same way, from B13) turns a quarter circle under the 90° rotation, leans under the shear, and stretches sideways and flattens under the scale. The printed `angle` line uses the two of your chapters that are furthest apart. Both rotations keep that angle to three decimals and keep L2 at `1.0000`. The shear and the scale change both, and the scale squeezes your fan much narrower, because it multiplies the `WORD2` direction by 0.5 and the `WORD1` direction by 3. The L1 norm is the odd one: the 90° rotation keeps it, because a quarter turn only swaps the two coordinates, and the 45° rotation changes it. So only L2 survives every rotation, which is why L2 is the one called length.

*If it broke:* a picture where the rotation turns the fan clockwise is the transpose of the matrix you meant. `U @ M.T` applies `M` to every row of `U`; `U @ M` applies the transpose.

**Step 4. `M @ v` is a combination of the columns.**

Add to the bottom of `landing.py`:

```python
M = MATRICES["shear"]
v = V2[2]
print("M @ v                    ", M @ v)
print("v[0]*col0 + v[1]*col1    ", v[0] * M[:, 0] + v[1] * M[:, 1])
```

*You should see* the same two numbers twice, with your chapter 2 counts in them. That is the Ch. 3 example you computed on paper during the video, done on your corpus: the count of `WORD1` says how much of the first column to take, and the count of `WORD2` says how much of the second.

**Extension — Which ways of processing a count vector are linear?** *(assigned)*

Four things people do to word counts before a model sees them. Before you run anything, write in your log, for each of the four below, "linear" or "not linear", and for the ones you call linear, the 2×2 matrix, found by asking where î and ĵ land. Commit the log. I will check your commit timestamps.

| Name | What it does to a chapter vector `v` |
|---|---|
| per 1000 words | `v * 1000 / size`: counts as a rate |
| total and difference | `[v[0] + v[1], v[0] - v[1]]` |
| centered | `v - mean`, where `mean` is the average of your ten chapter vectors |
| log counts | `log(1 + v)`, each coordinate |

Create `linear/linearity.py`:

```python
import numpy as np
from chapter_vectors import V2, size

vA, vB = V2[2], V2[7]
mean = V2.mean(axis=0)

candidates = {
    "per 1000 words":       lambda v: v * 1000 / size,
    "total and difference": lambda v: np.array([v[0] + v[1], v[0] - v[1]]),
    "centered":             lambda v: v - mean,
    "log counts":           lambda v: np.log1p(v),
}

e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
print("mean chapter vector:", mean)
print(f"{'function':22} {'add gap':>10} {'scale gap':>10} {'matrix gap':>11}")
for name, f in candidates.items():
    add_gap = np.abs(f(vA + vB) - (f(vA) + f(vB))).max()      # f(u + v) vs f(u) + f(v)
    scale_gap = np.abs(f(3 * vA) - 3 * f(vA)).max()           # f(3u) vs 3 f(u)
    M = np.column_stack([f(e1), f(e2)])                        # the matrix IF f were linear
    matrix_gap = np.abs(M @ vA - f(vA)).max()                  # does that matrix reproduce f?
    print(f"{name:22} {add_gap:10.4g} {scale_gap:10.4g} {matrix_gap:11.4g}")
    print(f"{'':22} matrix from where i and j land: {M.round(4).tolist()}")
```

Run it: `uv run python linear/linearity.py`.

*You should see* two functions with all three gaps at zero or at floating-point dust (`1e-14` or smaller), and two with gaps nowhere near zero: thousands for **centered**, and for **log counts** anything from a few units (add and scale) to hundreds (the matrix). For **centered**, the add gap is exactly the larger entry of your mean vector, printed on the first line (to the four significant figures the format shows), because `(u + v) - mean` and `(u - mean) + (v - mean)` differ by one `mean`: that is Krohn's affine map, a linear step plus a shift. For **log counts**, every gap is large and the "matrix" built from î and ĵ is nonsense, because no matrix can do what `log` does. For the two linear ones, the printed matrix should be the one you wrote in your log.

Write `linear/FINDINGS-b15.md`: the table as printed, your four predictions copied unchanged beside it, and one sentence per function on whether you were right. Name the function where your prediction was worst and give its gap. Then one sentence on what the centered result means for anyone who "normalizes" data before feeding it to a model and assumes the result is still a matrix away from the original.

Commit, push, open the pull request:

```bash
git add linear/ scratch/b15-video.py
git commit -m "B15: hand matrices checked, chapters transformed, linearity of four count transforms"
git push -u origin dev/transforms
```

On GitHub: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**. Say in the PR body where this is weakest.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`linear/MATRICES.md` (or `matrices.jpg`, committed before the code) + `linear/landing.py` + `linear/transforms.png` + `linear/linearity.py` + `linear/FINDINGS-b15.md` + `scratch/b15-video.py`.

**Reflection Questions**

1. Paste the three `lands at` lines from `landing.py` and the commit hash of your hand page. Was any matrix on the page wrong when the code checked it? If yes, paste the wrong one and say whether it was the transpose or a different mistake. If none was, paste the shear line and say what the picture would have looked like if you had typed its columns as rows.

2. Paste the four `chapter 2:` lines, with the chapter numbers your code picked for the `angle` column. Give the before and after angle for the scale, and explain from your own counts why that matrix narrowed your fan by as much as it did while both rotations left it alone. Then say which of the two rotations changed L1, with the numbers, and why the other one could not.

3. Paste your `linearity.py` table and the predictions you committed. Pick the function you got wrong, or if you got none wrong, the centered one. Show, with your own mean vector, the one line of algebra that produces its add gap, and say whether that function is "a matrix away" from your raw counts.
