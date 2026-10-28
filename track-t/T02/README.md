# T02 · Matrices Are Functions

**Meetings:** D25 · **Points:** 10 pts

**Watch — 11 min**
[3Blue1Brown, Essence of Linear Algebra, Ch. 3: Linear transformations and matrices](https://www.youtube.com/watch?v=kYB8IZa5AuE) · whole video (10:59, 11m)
 · Paper.

**During the video**

**Ch. 3 · compute before he does.** He works an example in which he tells you where î and ĵ land and then asks where some other vector ends up. Pause before he answers. Write that vector as `x · (where î lands) + y · (where ĵ lands)`, with his numbers, and do the arithmetic on paper. Unpause and check. That one line of arithmetic is the whole chapter, and Step 3 checks it in code on your own chapter vectors. Photograph the page and save it as `linear/video-notes-t02.jpg`; it goes in with the final commit.

**Notes**

**A matrix is a function that moves space, and its columns are where the basis vectors land.** To find the matrix for a transformation, ask where î = (1, 0) goes and where ĵ = (0, 1) goes. Those two answers, written as columns, are the matrix. Written as rows, they are the transpose, and the picture you get is a different transformation. This is the usual mistake on today's hand matrices; your `NOTATION.md` exists for it.

**A shear slides every point sideways by an amount set by its height.** Points on the x axis do not move, and the higher a point sits the further it is pushed, so vertical grid lines lean and horizontal ones stay put. Ask the two questions: î = (1, 0) is on the x axis, so it stays; ĵ = (0, 1) is at height 1, so it slides to (1, 1). That is row (b) in Step 2, and the one he draws at the end of Ch. 3.

**"Linear" has a precise meaning:** grid lines stay parallel and evenly spaced, and the origin does not move. In symbols, `f(u + v) = f(u) + f(v)` and `f(c·u) = c·f(u)`. A transformation that shifts everything two units right is not linear, because the origin moves. The extension tests four things people do to word counts, and one of them is exactly that.

**`np.column_stack([a, b])` puts `a` and `b` in as columns. `np.array([a, b])` puts them in as rows.** They are transposes of each other. When a result looks rotated the wrong way, check which one you used.

**Walkthrough — three matrices, by hand and then on your chapters**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T01 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/transforms
```

*If T01 is not approved yet:* today needs its `chapter_vectors.py`, so branch from it instead: `git switch dev/vectors && git switch -c dev/transforms`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Ch. 3 watched, his example computed before he did
- [ ] Step 2: three matrices by hand, in evidence/T02.md before any code runs
- [ ] Step 3: landing.py checks them and moves my chapters
- [ ] Extension: four predictions in the log, linearity.py run
- [ ] Fill every slot in evidence/T02.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Three matrices, by hand, written down first.**

No code. On paper, give the 2×2 matrix for each of these by asking where î lands and where ĵ lands:

| | Transformation |
|---|---|
| (a) | 90° counterclockwise rotation |
| (b) | horizontal shear that sends ĵ to (1, 1) and leaves î alone |
| (c) | scale by 3 in x and by 0.5 in y |

Type the three matrices under `## Step 2` in `evidence/T02.md` before you open any code. Step 3 checks them, and a matrix written after the check is a copy of the check.

**Step 3. The code checks your matrices, then moves your chapters.**

Open `linear/landing.py` and read it down to `if __name__`; the rest runs the checks. The three `np.array(...)` rows under `MATRICES` are zeros; fill them in from your Step 2 matrices, as rows of the matrix, which is how NumPy writes it. The fourth, a 45° rotation, is given. Then:

```bash
uv run python linear/landing.py
```

*You should see*, for each of the four matrices, a line saying where î and ĵ land, and under it a `chapter 2:` line with three before-and-after numbers: the L2 length of chapter 2's arrow, its L1 length (the sum of the absolute values), and the angle between your two furthest-apart chapters from T01's extension. `landing.py` first scales every chapter arrow to length 1 (the `U = ...` line), so L2 starts at `1.0000` for every corpus and L1 starts a little above 1. The landing points have to equal the columns of your Step 2 matrices. If they do not, the matrix is wrong or you typed columns as rows: fix the code, leave the Step 2 slot as you wrote it, and say which it was in `evidence/T02.md`. On Homer, with `jove`:

```text
rotate 90  i lands at [0. 1.]   j lands at [-1.  0.]
           chapter 2: L2 1.0000 -> 1.0000   L1 1.0400 -> 1.0400   angle(ch1, ch7) 1.821 -> 1.821 degrees
shear      i lands at [1. 0.]   j lands at [1. 1.]
           chapter 2: L2 1.0000 -> 1.0408   L1 1.0400 -> 1.0808   angle(ch1, ch7) 1.821 -> 1.689 degrees
scale      i lands at [3. 0.]   j lands at [0.  0.5]
           chapter 2: L2 1.0000 -> 2.9976   L1 1.0400 -> 3.0179   angle(ch1, ch7) 1.821 -> 0.304 degrees
rotate 45  i lands at [0.70710678 0.70710678]   j lands at [-0.70710678  0.70710678]
           chapter 2: L2 1.0000 -> 1.0000   L1 1.0400 -> 1.4130   angle(ch1, ch7) 1.821 -> 1.821 degrees
M @ v                     [1477.   58.]
v[0]*col0 + v[1]*col1     [1477.   58.]
```

Whatever your corpus: both rotations keep the angle and keep L2 at `1.0000`; the shear and the scale change both, and the scale squeezes your bundle of arrows to a sliver, because it multiplies the `WORD2` direction by 0.5 and the `WORD1` direction by 3. L1 is the odd one. The 90° rotation keeps it, because a quarter turn only swaps the two coordinates, and the 45° rotation changes it. Only L2 survives every rotation, which is why L2 is the one called length.

The last two lines are the Ch. 3 example on your corpus: `M @ v` and `v[0]` copies of column 0 plus `v[1]` copies of column 1 print the same two numbers, with your chapter 2 counts in them. Open `linear/transforms.png`. Your ten arrows are within a couple of degrees of each other, so the gray "fan" looks like one arrow; what you can see is the whole bundle turning a quarter circle under the rotation, stretching along the x axis under the scale, and barely moving under the shear, which is the point: a shear that sends ĵ to (1, 1) hardly touches arrows that lie almost along î. Paste everything it printed under `## Step 3`.

*If it broke:* `i lands at [ 0. -1.]` for the rotation, or a picture where it turns clockwise, means you typed the matrix down its columns instead of across its rows; `np.array([[0, -1], [1, 0]])` is the first row then the second.

**Extension — which ways of processing a count vector are linear? (ASSIGNED)**

Four things people do to word counts before a model sees them. Before you run anything, write in your log, for each of the four, "linear" or "not linear", and for the ones you call linear, the 2×2 matrix, found by asking where î and ĵ land. Commit the log. I will check your commit timestamps.

| Name | What it does to a chapter vector `v` |
|---|---|
| per 1000 words | `v * 1000 / size`: counts as a rate |
| total and difference | `[v[0] + v[1], v[0] - v[1]]` |
| centered | `v - mean`, where `mean` is the average of your ten chapter vectors |
| log counts | `log(1 + v)`, each coordinate |

```bash
uv run python linear/linearity.py
```

*You should see* the mean chapter vector, then one row per function with three gaps: `f(u + v)` against `f(u) + f(v)`, `f(3u)` against `3 f(u)`, and the matrix built from where î and ĵ land against `f` itself. Two functions have all three gaps at zero, and two have gaps that are not zero. On Homer:

```text
mean chapter vector: [1691.2   66.3]
function                  add gap  scale gap  matrix gap
per 1000 words                0.0        0.0         0.0
                       matrix from where i and j land: [[0.0352, 0.0], [0.0, 0.0352]]
total and difference          0.0        0.0         0.0
                       matrix from where i and j land: [[1.0, 1.0], [1.0, -1.0]]
centered                   1691.2     3382.4   2496211.2
                       matrix from where i and j land: [[-1690.2, -1691.2], [-66.3, -65.3]]
log counts                    6.7       13.4       976.3
                       matrix from where i and j land: [[0.6931, 0.0], [0.0, 0.6931]]
```

For **centered**, the add gap is exactly the larger entry of your mean vector, because `(u + v) - mean` and `(u - mean) + (v - mean)` differ by one `mean`: a linear step plus a shift, which is called affine. For **log counts**, every gap is large and the "matrix" built from î and ĵ is nonsense, because no matrix does what `log` does. For the two linear ones, the printed matrix is the one you wrote in your log. Paste the table under `## Extension` in `evidence/T02.md` and say, for each function, whether your prediction held.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T02.md` (the questions are below) and run `uv run python scripts/check_evidence.py T02` until it says `all slots filled`. Then:

```bash
git add linear/ evidence/T02.md
git commit -m "T02: hand matrices checked, chapters transformed, linearity of four count transforms"
git push -u origin dev/transforms
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the matrix you were least sure of before the code checked it.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/T02.md` with every slot filled + `linear/landing.py` with your three matrices + `linear/transforms.png` + `linear/video-notes-t02.jpg`.

**Reflection Questions** (answer under `## Reflection` in `evidence/T02.md`)

1. Paste the four `chapter 2:` lines. Give the before and after angle for the scale, and explain from your own counts why that matrix narrowed your bundle of arrows by that much while both rotations left it alone.

   *How to get it:* the angle is the last number pair on each `chapter 2:` line. The scale multiplies every arrow's `WORD2` coordinate by 0.5 and its `WORD1` coordinate by 3, so every slope from T01's extension is divided by 6, and arrows whose slopes were close to begin with end up closer still. A rotation turns every arrow by the same amount, so the angle between any two of them cannot change.

2. Which of the two rotations changed L1, with the numbers, and why could the other one not?

   *How to get it:* the L1 pair is the middle of each `chapter 2:` line. L1 is `|x| + |y|`. A 90° turn sends `(x, y)` to `(-y, x)`, and the sum of the absolute values is the same two numbers added; a 45° turn mixes `x` and `y` into each coordinate, and the sum changes.

3. Pick the function you got wrong in the extension, or if you got none wrong, `centered`. Show, with your own mean vector, the one line of algebra that produces its add gap, and say whether that function is "a matrix away" from your raw counts.

   *How to get it:* `f(u + v) - (f(u) + f(v))` with `f(v) = v - mean` simplifies to `mean`, so the add gap is the larger entry of the printed mean vector; check it against the table. A function with a nonzero gap has no matrix, so centered counts are a matrix away from raw counts only after you add the mean back.
