# B16 · Composition, Determinants, and the Visualizer

**Meetings:** D30 · **Points:** 15 pts

**Watch — 20 min**
[3Blue1Brown, Essence of Linear Algebra, Ch. 4: Matrix multiplication as composition](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (10:04, 10m)
[3Blue1Brown, Essence of Linear Algebra, Ch. 6: The determinant](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (10:03, 10m)
 · Paper, and your B15 hand page open next to it, because you are about to compose those three matrices.
 · Optional, not counted: Ch. 7, Inverse matrices, column space and null space (12:09), if you want the name for what happened to your singular system in B13.

**During the video**

Both chapters are pictures, and the walkthrough is the code-along: `transform_viz.py` draws what he draws.

**Ch. 4 · multiply before he does.** He composes two transformations and then works out the single matrix that does both. Pause when he sets it up. Using only "where does î land, then where does that land", write the composed matrix yourself, on paper, then unpause and check. Then take your B15 rotation and shear and do the same thing both ways round: rotation first then shear, and shear first then rotation. You now have two matrices on paper that Step 4 draws, and they should not be equal.

**Ch. 6 · draw the square every time.** Each time he gives a transformation and its determinant, sketch the unit square and the shape it becomes, and write the area next to it. Do it for your three B15 matrices too, before Step 3 prints their determinants. Commit this page with the rest as `linear/video-notes-b16.jpg`.

**Notes**

Ch. 4 answers the question B14 left open: matrix multiplication is function composition. `AB` means "apply `B` first, then `A`". Read right to left, like `f(g(x))`. That is why order matters, and it is also why your B21 backward pass runs in reverse.

The determinant is the area scaling factor. Determinant 0 means space got flattened onto a line or a point, which is exactly the "singular" from B13. A negative determinant means orientation flipped: î and ĵ swapped which side of each other they are on. These are the same fact told three ways; say so in your write-up.

`np.linalg.det` does floating-point arithmetic, so a determinant that is exactly 1 on paper can print as `0.9999999999999998`, and one that is exactly 0 can print as `2.2e-16` or, on a bigger matrix of counts, `2e-11`. Print with `:.4f`, and compare with `np.isclose`, never `==`.

The visualizer draws the grid with `line @ M.T`. `line` is a stack of points as rows, and `@ M.T` applies `M` to each row. Using `@ M` instead draws the transpose, and for a rotation that means turning the wrong way. If your 90° rotation turns clockwise, that is the bug.

`transform_viz.py` saves a picture rather than opening a window. `plt.show()` in a script blocks until you close the window, and a script that never finishes looks like a hang.

**Walkthrough — The visualizer**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:
```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/determinant
```
*If B15 is not approved yet:* `git switch dev/transforms && git switch -c dev/determinant`, because the extension uses your B15 matrices and chapter vectors.
```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-track, not main
bash scripts/start-entry.sh
```
Under the timestamp, write today's checklist:
```markdown
- [ ] Watch Essence of Linear Algebra Ch. 4, composing his example and my B15 rotation and shear by hand
- [ ] Watch Ch. 6, drawing the unit square every time
- [ ] Steps 2–3: transform_viz.py draws one matrix and prints its determinant
- [ ] Step 4: composition mode shows AB and BA side by side
- [ ] Step 5: det(AB) = det(A) det(B) on my own matrices
- [ ] Extension: one option from the menu, with a prediction committed first
- [ ] Push dev/determinant and open the PR
```

**Step 2. The drawing function.**

Create `linear/transform_viz.py`. Type it:

```python
import sys
import numpy as np
import matplotlib.pyplot as plt

def draw(ax, M, title):
    M = np.asarray(M, dtype=float)
    for t in np.linspace(-3, 3, 13):
        for line in (np.array([[t, -3], [t, 3]]), np.array([[-3, t], [3, t]])):
            ax.plot(*line.T, color="lightgray", lw=0.6)             # the grid before
            ax.plot(*(line @ M.T).T, color="steelblue", lw=0.6)     # the same grid after M
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    ax.fill(*(square @ M.T).T, color="gold", alpha=0.4)             # where the unit square went
    ax.quiver(0, 0, M[0, 0], M[1, 0], angles="xy", scale_units="xy", scale=1, color="green")  # i lands here
    ax.quiver(0, 0, M[0, 1], M[1, 1], angles="xy", scale_units="xy", scale=1, color="red")    # j lands here
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    ax.set_title(f"{title}   det = {np.linalg.det(M):.4f}")
```

Green is where î lands, red is where ĵ lands, and gold is the unit square after the move. Its area is the determinant, and the title prints it so you can check the picture against the number.

**Step 3. One matrix from the command line.**

Add to the bottom:

```python
if __name__ == "__main__":
    nums = [float(x) for x in sys.argv[1:]]
    if len(nums) == 4:
        M = np.array(nums).reshape(2, 2)
        fig, ax = plt.subplots(figsize=(6, 6))
        draw(ax, M, "M")
        plt.savefig("linear/viz.png", dpi=110)
        print("det =", np.linalg.det(M))
```

The four numbers are the matrix read across its rows: `a b c d` is `[[a, b], [c, d]]`. Run your three B15 matrices, one at a time, and look at `linear/viz.png` after each:

```bash
uv run python linear/transform_viz.py 0 -1 1 0
```

*You should see*, for the rotation, the grid landing on itself (a quarter turn of a square grid does that; the arrows and the gold square show the turn), green pointing straight up, red pointing left, and `det = 1.0` (or `0.9999999999999998`). For the shear, the gold square leaned into a parallelogram with the same base and height, and a determinant of 1. For your scale, a rectangle 3 wide and 0.5 tall and a determinant of 1.5, printed as `1.5000000000000002`; the title rounds it. Compare each with the area sentences on your B15 page and the squares you drew during Ch. 6. Then run `0 1 1 0`, which swaps î and ĵ, and look for the minus sign.

**Step 4. Composition: `AB` against `BA`.**

Replace the `if __name__` block so it also takes eight numbers:

```python
if __name__ == "__main__":
    nums = [float(x) for x in sys.argv[1:]]
    if len(nums) == 4:
        M = np.array(nums).reshape(2, 2)
        fig, ax = plt.subplots(figsize=(6, 6))
        draw(ax, M, "M")
        plt.savefig("linear/viz.png", dpi=110)
        print("det =", np.linalg.det(M))
    elif len(nums) == 8:
        A = np.array(nums[:4]).reshape(2, 2)
        B = np.array(nums[4:]).reshape(2, 2)
        fig, axes = plt.subplots(2, 2, figsize=(11, 11))
        draw(axes[0, 0], A, "A")
        draw(axes[0, 1], B, "B")
        draw(axes[1, 0], A @ B, "AB  (B first, then A)")
        draw(axes[1, 1], B @ A, "BA  (A first, then B)")
        plt.savefig("linear/composition.png", dpi=110)
        print("AB =\n", A @ B, "\nBA =\n", B @ A)
        print("det(A) det(B) =", np.linalg.det(A) * np.linalg.det(B),
              "  det(AB) =", np.linalg.det(A @ B), "  det(BA) =", np.linalg.det(B @ A))
    else:
        print("give 4 numbers for one matrix or 8 for two")
```

Run it with your B15 rotation as `A` and your shear as `B`:

```bash
uv run python linear/transform_viz.py 0 -1 1 0  1 1 0 1
```

*You should see* two bottom panels that are visibly different pictures, and printed `AB` and `BA` that match the two matrices you wrote on paper during Ch. 4. If one of them does not match, your paper did the composition in the other order. All three determinants print as the same number, 1 here, even though the two products are different matrices.

**Step 5. `det(AB) = det(A) · det(B)`, on a pair that is not 1.**

Run it again with your scale as `A` and your shear as `B`, and then with your scale and `0 1 1 0`. Write the three determinants from each run in your log.

*You should see* `det(A) det(B)`, `det(AB)` and `det(BA)` agree every time: `1.5` for scale and shear, `-1.5` for scale and the swap (printed as `1.5000000000000002` and `-1.5000000000000002`). The area scale of doing two things is the product of the area scales, whichever order you do them in, even though the order changes where everything goes.

**Extension — Determinants of your corpus** *(choose one; write which and why in `linear/FINDINGS-b16.md`)*

Whichever you pick, write your prediction in the log and commit it before you run anything. I will check your commit timestamps. Every option ends in a number, a baseline to compare it with, and the place it got worse.

| Option | What you do | Baseline | The number |
|---|---|---|---|
| **1. The most nearly dependent pair** | Scale each of your ten B13 chapter vectors to length 1, and for all 45 pairs compute `abs(np.linalg.det(np.column_stack([u_i, u_j])))`, which is the sine of the angle between them. Draw the smallest pair with `draw`. | Your B13 pair, chapters 2 and 7 | The smallest and largest of the 45, and where 2–7 ranks |
| **2. A better second word** | Keep `WORD1`. For every word ranked 20 to 200 in your corpus as `WORD2`, compute the median of those 45 values. | Your B13 `WORD2` | The word with the largest median, its median, and yours |
| **3. Rates, then centering** | Take `M = np.column_stack([vA, vB])` for chapters 2 and 7 in raw counts. Predict `det` after converting both to per-1000-word rates (B15's scale), then check. Then center both (B15's affine map) and try to predict that one. | `det(M)` in raw counts | The ratio of the two determinants, against `(1000 / size) ** 2` |
| **4. Six orders of three** | Multiply your B15 rotation, shear and scale in all six orders (`itertools.permutations`). | The single product `R @ H @ S` | How many distinct matrices you got, and all six determinants |

For option 1, all 45 values are small for two slices of the same text, the smallest often a thousandth or less, and your 2–7 pair may or may not be the smallest; if your B13 coefficients were large, this pair's small value is why; if they were small, your chapter 5 happened to sit between 2 and 7. For option 3, the rate conversion multiplies the determinant by exactly `(1000 / size) ** 2`, because it scales area in both directions, and centering has no such rule, because it is not a matrix. For option 4, all six determinants are equal and the number of distinct matrices is not one.

Write `linear/FINDINGS-b16.md`: which option and why; the code's output; your prediction, copied unchanged, and whether it held; and the place it got worse, named with numbers. Save any picture you drew as `linear/extension.png`. Put the code in `linear/extension_b16.py`.

Commit, push, open the pull request:

```bash
git add linear/
git commit -m "B16: transform visualizer with composition, det(AB) check, corpus determinants"
git push -u origin dev/determinant
```

On GitHub: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**. Say in the PR body where this is weakest.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`linear/transform_viz.py` (draws before/after with î and ĵ highlighted, prints the determinant, composes two matrices) + `linear/viz.png` + `linear/composition.png` + `linear/extension_b16.py` + `linear/FINDINGS-b16.md` + `linear/video-notes-b16.jpg`.

**Reflection Questions**

1. Paste the printed `AB` and `BA` from Step 4 and the composed matrices from your Ch. 4 paper. Did your paper match, and in which order had you composed them? Then describe, from `composition.png`, one specific place in the picture where `AB` and `BA` differ: where the gold square sits, or where green or red points.

2. Paste the three determinant lines from Step 5 for your scale and the swap. Say what the minus sign means for the gold square in that picture, and then say why "area scale" alone could not have told you the product would be `-1.5` rather than `1.5`, while "area scale with orientation" could.

3. Paste your extension output and your committed prediction. Say where it got worse, with the number: the pair or word that was closest to singular, the rule that centering broke, or the orders that did not commute. Tie it back to one specific number from your B13 `FINDINGS.md` and say whether this assignment explains it.
