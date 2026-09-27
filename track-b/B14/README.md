# B14 · Dot Product and Matrix Multiplication, Built From Nothing

**Meetings:** D28 · **Points:** 15 pts

**Watch — about 24 min**
[3Blue1Brown, Essence of Linear Algebra, Ch. 9: Dot products and duality](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (14:12, 14m)
[Jon Krohn, Machine Learning Foundations: Matrix Multiplication, Topic 19](https://www.youtube.com/watch?v=Kqh7stbGakg) · the first ten minutes, up to the first code demo; stop when that demo is done (about 10m of a 25:00 video)
 · Have `scratch/b14-video.py` open for Krohn. Paper for 3Blue1Brown.

**During the video**

**Ch. 9 · draw.** When he projects one vector onto another, stop and do it with the two vectors from your B13 page: `v = (2, 1)` and `w = (-1, 3)`. Draw both from the origin, drop the perpendicular from `w` onto the line through `v`, and write the dot product next to it: `2·(-1) + 1·3 = 1`. Then write whether the angle between them is under or over 90°, and check that the sign of `1` agrees. Photograph the page with the rest of today's work as `linear/video-notes-b14.jpg`. The duality half of the chapter, where a 1×2 matrix becomes a projection, is worth watching closely and has nothing to draw; it comes back as the reason `dot` and a one-row `matmul` give the same number in Step 6.

**Krohn · type and run.** He is teaching from a notebook and you are typing into `scratch/b14-video.py`. **Type everything he types in NumPy, and run it** with `uv run python scratch/b14-video.py` after each new line, with a `print` around anything he evaluates in a cell. The shapes are the point: every time he multiplies two things, add `print(X.shape, Y.shape, (X @ Y).shape)` and read the three shapes before you read the numbers. If he repeats a product in PyTorch or TensorFlow, watch that part and do not install either one; PyTorch arrives in B17 and TensorFlow never does.

**Notes**

**NumPy is banned for the deliverable.** `linear/linalg.py` uses lists of lists only. NumPy is allowed in the tests, to check your answers, and in the extension, to time yourself against it.

Dimension rule, memorize it: `(n × m) @ (m × p) → (n × p)`. The inner dimensions must match and they vanish; the outer dimensions survive. If your code runs on square matrices but explodes on a 2×3 @ 3×4, you wrote the loops in the wrong order.

Row-major means `A[i][j]` is row `i`, column `j`. Getting column `j` out of `B` is `[row[j] for row in B]`. Most people's first `matmul` is silently transposed, and a square symmetric test matrix cannot catch it, because a symmetric matrix is its own transpose. Test with a non-square, non-symmetric pair.

Raise a clear `ValueError` on a dimension mismatch instead of letting Python throw an `IndexError` twelve frames deep. The mismatch test is the one people skip and the one that catches their bugs.

`pytest` finds `linear/linalg.py` because it puts the test file's own folder on the import path. That only works if you run it from the repo root as `uv run pytest linear -q`. Running `pytest` inside `linear/` works too; running `python linear/test_linalg.py` runs nothing and prints nothing, which looks like success.

**Walkthrough — `dot` and `matmul` from nothing, then on your chapters**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:
```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/matmul
```
*If B13 is not approved yet:* today needs its `chapter_vectors.py`, so branch from it instead: `git switch dev/vectors && git switch -c dev/matmul`.
```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-track, not main
bash scripts/start-entry.sh
```
Under the timestamp, write today's checklist:
```markdown
- [ ] Watch Essence of Linear Algebra Ch. 9 and draw the projection of (-1, 3) onto (2, 1)
- [ ] Watch Krohn Topic 19, first ten minutes, typing along in scratch/b14-video.py
- [ ] Steps 2–3: dot and matmul in linear/linalg.py, no NumPy
- [ ] Steps 4–5: five tests green, then break matmul on purpose and see which test catches it
- [ ] Step 6: dot products of my chapters
- [ ] Extension: the chapter similarity matrix in two vocabulary bands, and the timing against NumPy
- [ ] Push dev/matmul and open the PR
```

**Step 2. `dot`.**

Create `linear/linalg.py`. Type it:

```python
def dot(a, b):
    if len(a) != len(b):
        raise ValueError(f"dot: lengths {len(a)} and {len(b)} do not match")
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total
```

That is the whole of the algebraic definition: multiply matching coordinates and add. 3Blue1Brown spent fourteen minutes showing you that this sum is also a projection, which is why the next function can be built from nothing but this one.

**Step 3. `matmul`.**

Add this below `dot`:

```python
def matmul(A, B):
    n, m = len(A), len(A[0])
    m2, p = len(B), len(B[0])
    if m != m2:
        raise ValueError(f"matmul: ({n} x {m}) @ ({m2} x {p}): inner dimensions {m} and {m2} differ")
    C = []
    for i in range(n):                          # each row of A
        row = []
        for j in range(p):                      # each column of B
            col = [B[k][j] for k in range(m)]   # k runs over the inner dimension, the one that vanishes
            row.append(dot(A[i], col))
        C.append(row)
    return C
```

Every entry of the product is one `dot`: row `i` of `A` against column `j` of `B`. The loop variable `k` is the inner dimension, and it never appears in the shape of `C`.

**Step 4. Five tests against NumPy.**

```bash
uv add --dev pytest
```

Create `linear/test_linalg.py`:

```python
import numpy as np
import pytest
from linalg import dot, matmul

def test_dot_small():
    assert dot([1, 2, 3], [4, 5, 6]) == 32

def test_dot_perpendicular():
    assert dot([2, 0], [0, 7]) == 0

def test_square():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    assert np.allclose(matmul(A, B), np.array(A) @ np.array(B))

def test_non_square():
    A = [[1, 2, 3], [4, 5, 6]]                          # 2 x 3
    B = [[1, 0, 2, 1], [0, 1, 1, 3], [2, 1, 0, 1]]      # 3 x 4
    got = matmul(A, B)
    assert np.array(got).shape == (2, 4)
    assert np.allclose(got, np.array(A) @ np.array(B))

def test_mismatch():
    with pytest.raises(ValueError):
        matmul([[1, 2, 3]], [[1, 2], [3, 4]])
```

Run from the repo root: `uv run pytest linear -q`.

*You should see* `5 passed`.

*If it broke:* `ModuleNotFoundError: No module named 'linalg'` means you ran it from somewhere other than the repo root, or you put an `__init__.py` in `linear/`. Delete the `__init__.py`; this folder is not a package.

**Step 5. Break it on purpose.**

In `matmul`, change `col = [B[k][j] for k in range(m)]` to `col = [B[j][k] for k in range(m)]`, which is the transposed bug. Before you run the tests, write in your log which of the five you expect to fail. Then run `uv run pytest linear -q` and write down which ones did, and the first line of each error. Put the line back and run again to `5 passed`. Commit with a message that names the test that caught it.

*You should see* the square test fail on a wrong answer and the non-square test fail with an `IndexError` or a wrong shape, and the two `dot` tests and the mismatch test pass, because they never reach the bad line. If only one test failed, look at which one survived and why its matrices could not tell a row from a column.

**Step 6. Dot products of your chapters.**

Create `linear/gram.py`. It uses your B13 chapters and your own `dot`:

```python
from collections import Counter
from chapter_vectors import chapters, count_vector, words
from linalg import dot, matmul

ranked = [w for w, _ in Counter(words).most_common()]
vocab = ranked[:50]
vA = count_vector(chapters[2], vocab).tolist()
vB = count_vector(chapters[7], vocab).tolist()

d = dot(vA, vB)
cos = d / (dot(vA, vA) ** 0.5 * dot(vB, vB) ** 0.5)
print(f"chapter 2 . chapter 7 = {d:,.0f}   cosine = {cos:.4f}")
print("one-row matmul:", matmul([vA], [[x] for x in vB]))
```

Run it: `uv run python linear/gram.py`.

*You should see* a dot product in the hundreds of thousands or millions, because `the` alone appears a thousand or two times in each chapter and contributes its count squared; a cosine above 0.9; and a one-row `matmul` that prints the same dot product wrapped as `[[...]]`. That last line is the duality half of Ch. 9 on your own data: a 1×50 matrix times a 50×1 matrix is a dot product.

**Extension — Which of your chapters are alike, and what did the common words hide?** *(assigned)*

Before you run anything, write two predictions in your log and commit them: the smallest cosine you expect between any two of your ten chapters using the 50 most common words, and whether that number goes up or down if you throw the 50 most common words away. I will check your commit timestamps.

Add this to the bottom of `linear/gram.py`:

```python
import time
import numpy as np

N = len(chapters)

def cosine_matrix(vocab):
    X = [count_vector(ch, vocab).tolist() for ch in chapters]      # 10 x V
    Xt = [list(col) for col in zip(*X)]                             # V x 10
    G = matmul(X, Xt)                                               # 10 x 10, every chapter against every chapter
    norms = [G[i][i] ** 0.5 for i in range(N)]
    return np.array([[G[i][j] / (norms[i] * norms[j]) for j in range(N)] for i in range(N)])

for name, band in (("top 50", ranked[:50]), ("ranks 51-250", ranked[50:250])):
    C = cosine_matrix(band)
    off = C[~np.eye(N, dtype=bool)]
    i, j = np.unravel_index(np.argmax(C - 2 * np.eye(N)), C.shape)
    k, l = np.unravel_index(np.argmin(C), C.shape)
    print(f"{name:13} min {off.min():.3f}  mean {off.mean():.3f}  max {off.max():.3f}"
          f"  most alike {i},{j}  least alike {k},{l}")

def best_of_5(f):
    times = []
    for _ in range(5):
        t = time.perf_counter(); f(); times.append(time.perf_counter() - t)
    return min(times)

for V in (50, 500, 2000):
    X = [count_vector(ch, ranked[:V]).tolist() for ch in chapters]
    Xt = [list(col) for col in zip(*X)]
    A = np.array(X)
    tp = best_of_5(lambda: matmul(X, Xt))
    tn = best_of_5(lambda: A @ A.T)
    print(f"V={V:5d}  pure Python {tp * 1000:8.2f} ms   NumPy {tn * 1000:7.3f} ms   ratio {tp / tn:7.0f}x"
          f"   same answer: {np.allclose(matmul(X, Xt), A @ A.T)}")
```

*You should see* two rows of similarity and three rows of timing. With the top 50 words, every pair of chapters is very alike, all of them somewhere above 0.9, because every chapter of any book is mostly `the`, `of` and `and` in about the same proportions. With ranks 51 to 250 the numbers spread out and drop, and the most-alike and least-alike pairs may change. The timing ratio is large at every size and it grows as `V` grows; `same answer: True` on all three rows. If your corpus has fewer than 2,000 distinct words, the last row uses all of them and says so by being the same as the one before.

Write `linear/FINDINGS-b14.md`: both similarity rows; which pair of chapters is most alike in each band, and one sentence on why, from what you know about your corpus; your two predictions, copied unchanged, with one sentence each on whether they held; and the timing table. Name what got worse: the top-50 band cannot tell your chapters apart, and pure Python gets worse relative to NumPy as the vocabulary grows. Give the numbers for both.

Commit, push, open the pull request:

```bash
git add linear/ scratch/b14-video.py pyproject.toml uv.lock
git commit -m "B14: dot and matmul from lists, tests vs NumPy, chapter similarity in two bands"
git push -u origin dev/matmul
```

On GitHub: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**. Say in the PR body where this is weakest.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`linear/linalg.py` (no NumPy) + `linear/test_linalg.py` (five cases, one non-square, one mismatch, all against NumPy) + `linear/gram.py` + `linear/FINDINGS-b14.md` + `scratch/b14-video.py` + `linear/video-notes-b14.jpg`.

**Reflection Questions**

1. Paste the `pytest` output from Step 5 with the transposed line in place, and the commit message you wrote after it. Name the test that caught the bug and the one you predicted would, and explain from the actual matrices in `test_linalg.py` why the square test gave a wrong answer rather than a crash while the non-square one did something else.

2. Paste both similarity rows from `gram.py` with your prediction beside them. Say which pair of chapters was most alike in the ranks 51–250 band and what, in your corpus, those two chapters share. Then say what the top-50 band was really measuring, using its min and max as evidence.

3. Paste your timing table. For `V=2000`, work out how many multiply-adds your `matmul` did (`n × m × p`, with your real shapes) and divide by your pure-Python time to get multiply-adds per second. Say what that rate predicts for one 512×512 @ 512×512 product, and why that number should worry you about pure-Python code before B19.
