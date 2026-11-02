# T03 · Composition: Two Matrices, One Matrix

**Meetings:** D26 · **Points:** 15 pts

**Watch — 15 min**
[3Blue1Brown, Essence of Linear Algebra, Ch. 4: Matrix multiplication as composition](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (10:04, 10m)
[3Blue1Brown, Essence of Linear Algebra, Ch. 5: Three-dimensional linear transformations](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (4:46, 5m)
 · Paper, and your T02 matrices beside it, because you are about to compose them.

**During the video**

**Ch. 4 · multiply before he does.** He composes two transformations and then works out the single matrix that does both. Pause when he sets it up. Using only "where does î land, then where does that land", write the composed matrix yourself, then unpause and check. Then take your T02 rotation and shear and do the same thing both ways round: rotation first then shear, and shear first then rotation. You now have two matrices on paper that Step 3 draws, and they are not equal. Step 2 starts by committing them.

**Ch. 5 · nothing to draw.** The point of the five minutes is that everything from Ch. 3 and 4 holds with three coordinates instead of two: a 3×3 matrix is three columns, each saying where a basis vector lands. Your count vectors in T05 have fifty coordinates; nothing changes but the size.

**Notes**

**Matrix multiplication is function composition.** `AB` means "apply `B` first, then `A`". Read right to left, like `f(g(x))`. That is why order matters, and it is why micrograd's backward pass runs in reverse.

**The visualizer draws the grid with `line @ M.T`.** `line` is a stack of points as rows, and `@ M.T` applies `M` to each row. The file is shipped and has nothing to edit. If your 90° rotation turns clockwise, you typed the four numbers in the wrong order: the command takes the matrix across its rows, `0 -1 1 0`, not down its columns.

**`transform_viz.py` saves a picture rather than opening a window.** `plt.show()` in a script blocks until you close the window, and a script that never finishes looks like a hang.

**Walkthrough — the visualizer**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T02 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/composition
```

*If T02 is not approved yet:* `git switch dev/transforms && git switch -c dev/composition`, because the extension reads your T02 matrices.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Ch. 4 watched, his example and my two compositions done by hand; Ch. 5 watched
- [ ] Step 2: transform_viz.py on my three matrices and the swap
- [ ] Step 3: AB against BA
- [ ] Extension: prediction in evidence/T03.md and committed, then compose3.py run
- [ ] Fill every slot in evidence/T03.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. One matrix at a time.**

First, the two matrices from your page, before any code runs: type them under `## During the video` in `evidence/T03.md` as rows, and commit that file on its own, so the timestamp shows the paper came first:

```bash
git add evidence/T03.md && git commit -m "T03: two compositions by hand"
```

Open `linear/transform_viz.py` and read `draw`: thirteen grid lines each way, drawn gray before and blue after `M`; the unit square after `M` in gold; green and red arrows for where î and ĵ land. Then run it on your three T02 matrices, one at a time, and look at `linear/viz.png` after each. The four numbers are the matrix read across its rows: `a b c d` is `[[a, b], [c, d]]`.

```bash
uv run python linear/transform_viz.py 0 -1 1 0
```

*You should see* `det = 1.0` and, in the picture, the grid landing on itself (a quarter turn of a square grid does that; the arrows and the gold square show the turn), green pointing straight up, red pointing left. For the shear, the gold square leans into a parallelogram with the same base and height, and `det = 1.0`. For your scale, a rectangle 3 wide and 0.5 tall and `det = 1.5000000000000002`; the title rounds it. Then run `0 1 1 0`, which swaps î and ĵ, and look for the minus sign: `det = -1.0`. Paste the four `det =` lines under `## Step 2` in `evidence/T03.md`. The determinant is tomorrow's video; today, read it as the area of the gold square, with a sign.

**Step 3. `AB` against `BA`.**

Eight numbers draw two matrices and both products. Run it with your rotation as `A` and your shear as `B`:

```bash
uv run python linear/transform_viz.py 0 -1 1 0  1 1 0 1
```

*You should see* `AB` and `BA` printed, three determinants that all equal `1.0`, and `linear/composition.png` with four panels: `A`, `B`, `AB` (B first, then A) and `BA` (A first, then B). The two bottom panels are visibly different pictures.

```text
AB =
 [[ 0. -1.]
 [ 1.  1.]]
BA =
 [[ 1. -1.]
 [ 1.  0.]]
det(A) det(B) = 1.0   det(AB) = 1.0   det(BA) = 1.0
```

Those two matrices are the two you wrote on paper during Ch. 4, if you composed in the right order; if one of them does not match, your paper did it the other way round. Paste the block under `## Step 3` and say whether your paper matched.

**Extension — three in a row (ASSIGNED)**

Your rotation, shear and scale can be multiplied in six orders. Before you run anything, write under `## Extension` in `evidence/T03.md` how many different matrices you expect from the six, and whether their determinants differ. Commit that file. I will check your commit timestamps.

```bash
uv run python linear/compose3.py
```

*You should see* six lines, one per order, each with the product and its determinant and the word `new` or `same as one above`, then a count. On Homer's matrices, which are everyone's matrices today, all six are `new`: `6 distinct matrices from 6 orders`, and every determinant is `1.5000`. The order changes where everything lands and does not change the area scale. Paste the lines under `## Extension` in `evidence/T03.md` and say whether your prediction held.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T03.md` (the questions are below) and run `uv run python scripts/check_evidence.py T03` until it says `all slots filled`. Then:

```bash
git add linear/ evidence/T03.md
git commit -m "T03: visualizer on three matrices, AB against BA, six orders of three"
git push -u origin dev/composition
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body says which of the two compositions you got wrong on paper, or that both were right.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/T03.md` with every slot filled + `linear/viz.png` + `linear/composition.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/T03.md`)

1. From `linear/composition.png`, name one specific place where `AB` and `BA` differ: where the gold square sits, or where green or red points.

   *How to get it:* compare the two bottom panels. Green is column 0 of the matrix printed above the picture; in `AB` it is `(0, 1)` and in `BA` it is `(1, 1)`.

2. `AB` means "apply B first, then A". Say why, using where î goes under rotation-then-shear, which is `BA` in Step 3's run, step by step.

   *How to get it:* in Step 3, `A` is the rotation and `B` is the shear, so rotation first is `BA`. î goes to `(0, 1)` under the rotation, and `(0, 1)` is ĵ, which the shear sends to `(1, 1)`. That is column 0 of the printed `BA`: the shear's matrix times the rotation's matrix, the one applied first written on the right.

3. All three determinants were the same number in Step 3 even though `AB` and `BA` are different matrices. Say why, in terms of the area of the gold square.

   *How to get it:* each transformation multiplies every area by its own factor, and multiplying by two factors gives the same product in either order. Where the square lands depends on the order; how big it is does not.
