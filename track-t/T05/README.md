# T05 · Dot Product and Matrix Multiplication, Built From Nothing

**Meetings:** D28 · **Points:** 10 pts

**Watch — 19 min**
[3Blue1Brown, Essence of Linear Algebra, Ch. 8: Nonsquare matrices as transformations between dimensions](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (4:27, 5m)
[3Blue1Brown, Essence of Linear Algebra, Ch. 9: Dot products and duality](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (14:12, 14m)
 · Paper.

**During the video**

**Ch. 8 · write the rule.** A 2×3 matrix takes a vector with three coordinates and lands it in two. Write the dimension rule on your page before Step 2 needs it: `(n × m) @ (m × p) → (n × p)`. The inner dimensions must match, and they vanish.

**Ch. 9 · draw.** When he projects one vector onto another, stop and do it with your T01 pair: `v = (2, 1)` and `w = (-1, 3)`. Draw both from the origin, drop the perpendicular from `w` onto the line through `v`, and write the dot product next to it: `2·(-1) + 1·3 = 1`. Write whether the angle between them is under or over 90°, and check that the sign of `1` agrees. Commit the page as `linear/video-notes-t05.jpg`. The duality half of the chapter, where a 1×2 matrix becomes a projection, has nothing to draw; it comes back in Step 5 as the reason `dot` and a one-row `matmul` give the same number.

**Notes**

**NumPy is banned in `linear/linalg.py`.** Lists of lists only. NumPy is allowed in the tests, to check your answers, and in `gram.py`, to time yourself against it.

**Row-major means `A[i][j]` is row `i`, column `j`.** Getting column `j` out of `B` is `[row[j] for row in B]`. Most people's first `matmul` is silently transposed, and a square symmetric test matrix cannot catch it, because a symmetric matrix is its own transpose. The tests use a non-square, non-symmetric pair for that reason.

**Raise a clear `ValueError` on a dimension mismatch** instead of letting Python throw an `IndexError` twelve frames deep. The mismatch test is the one people skip and the one that catches their bugs.

**`pytest` finds `linear/linalg.py` because it puts the test file's own folder on the import path.** That works only from the repo root, as `uv run pytest linear -q`. `python linear/test_linalg.py` runs nothing and prints nothing, which looks like success.

**Walkthrough — `dot` and `matmul` from nothing, then on your chapters**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T04 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/matmul
```

*If T04 is not approved yet:* `git switch dev/determinant && git switch -c dev/matmul`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Ch. 8 watched, the dimension rule written; Ch. 9 watched, the projection drawn
- [ ] Step 2: dot and matmul typed into linear/linalg.py
- [ ] Step 3: five tests green
- [ ] Step 4: prediction in the log, the transposed bug, which test caught it
- [ ] Step 5: gram.py, two chapters, one number
- [ ] Fill every slot in evidence/T05.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. `dot`, then `matmul`.**

This is the one file in the linear algebra week you type rather than run, because these two functions are the lesson. Create `linear/linalg.py` and type it:

```python
# linear/linalg.py  (T05)
# dot and matmul from nothing: lists of lists, no NumPy.

# In: two lists of numbers of the same length.  Out: one number, the sum of the products of matching entries.
def dot(a, b):
    if len(a) != len(b):
        raise ValueError(f"dot: lengths {len(a)} and {len(b)} do not match")
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

# In: A, an n x m list of lists, and B, an m x p list of lists.  Out: the n x p product, as a list of lists.
def matmul(A, B):
    n, m = len(A), len(A[0])                    # rows and columns of A
    m2, p = len(B), len(B[0])                   # rows and columns of B
    if m != m2:
        raise ValueError(f"matmul: ({n} x {m}) @ ({m2} x {p}): inner dimensions {m} and {m2} differ")
    C = []
    for i in range(n):                          # each row of A
        row = []
        for j in range(p):                      # each column of B
            col = [B[k][j] for k in range(m)]   # column j of B, pulled out as a list; k runs over the inner dimension
            row.append(dot(A[i], col))          # one entry of the product is one dot: row i of A against column j of B
        C.append(row)
    return C
```

`dot` is the whole algebraic definition: multiply matching coordinates and add. Ch. 9 spent fourteen minutes showing that this sum is also a projection, which is why `matmul` can be built from nothing but `dot`: every entry of the product is row `i` of `A` against column `j` of `B`. The loop variable `k` is the inner dimension, and it never appears in the shape of `C`.

**Step 3. Five tests against NumPy.**

`linear/test_linalg.py` shipped with the repo: two `dot` cases, a square product, a 2×3 times 3×4, and a mismatch that has to raise. From the repo root:

```bash
uv run pytest linear -q
```

*You should see* `5 passed`. Paste the last line under `## Step 3` in `evidence/T05.md`. *If it broke* with `ModuleNotFoundError: No module named 'linalg'`, you ran it from somewhere other than the repo root, or you put an `__init__.py` in `linear/`; delete it, the folder is not a package.

**Step 4. Break it on purpose.**

In `matmul`, change `col = [B[k][j] for k in range(m)]` to `col = [B[j][k] for k in range(m)]`, the transposed bug. Before you run the tests, write in your log which of the five you expect to fail. Commit the log. Then run `uv run pytest linear -q`.

*You should see* `2 failed, 3 passed`: the square test fails on a wrong answer (`assert False`), the non-square test fails with `IndexError: list index out of range`, and the two `dot` tests and the mismatch test pass, because they never reach the bad line. Paste the `FAILED` lines and the summary line under `## Step 4`. Put the line back, run again to `5 passed`, and commit with a message that names the test that caught it. *If it broke* with the tests still failing after you put the line back, delete `linear/__pycache__`; Python kept the old compiled file because the edit landed within the same second.

**Step 5. Two chapters, one number.**

```bash
uv run python linear/gram.py
```

`gram.py` takes chapters 2 and 7 over the 50 most common words of your corpus and calls your `dot` on them. *You should see* a dot product in the millions, a cosine above 0.9, and a one-row `matmul` that prints the same dot product wrapped as `[[...]]`. On Homer: `chapter 2 . chapter 7 = 7,470,275   cosine = 0.9538` and `one-row matmul: [[7470275.0]]`. The dot product is in the millions because `the` alone appears a thousand or two times in each chapter and contributes its count squared. The last line is the duality half of Ch. 9 on your own data: a 1×50 matrix times a 50×1 matrix is a dot product. Paste both lines under `## Step 5`.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T05.md` (the questions are below) and run `uv run python scripts/check_evidence.py T05` until it says `all slots filled`. Then:

```bash
git add linear/ evidence/T05.md
git commit -m "T05: dot and matmul from lists, five tests, two chapters dotted"
git push -u origin dev/matmul
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the line of `matmul` you understand least.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/T05.md` with every slot filled + `linear/linalg.py` (no NumPy) + the commit that names the test that caught the bug + `linear/video-notes-t05.jpg`.

**Reflection Questions** (answer under `## Reflection` in `evidence/T05.md`)

1. Name the test that caught the bug and the one you predicted would. Explain from the actual matrices in `test_linalg.py` why the square test gave a wrong answer rather than a crash, while the non-square one crashed.

   *How to get it:* open `linear/test_linalg.py`. In the square test `B` is 2×2, so `B[j][k]` with `j` and `k` both under 2 is always a real entry, just the wrong one; in the non-square test `B` is 3×4, so `B[j][k]` with `j` up to 3 asks for a fourth row that does not exist.

2. The one-row `matmul` printed the same number as `dot`, wrapped as `[[...]]`. Say why a 1×50 matrix times a 50×1 matrix is a dot product, using the loops in your own `matmul`.

   *How to get it:* with `n = 1` and `p = 1` the two outer loops each run once, `col` is the whole of `B` as a list, and the one entry of `C` is `dot(A[0], col)`.

3. Your cosine between chapters 2 and 7 is above 0.9. Say what in the 50 most common words makes two slices of one text look that alike, and what you would have to change about the vocabulary to tell them apart. T06 checks your answer.

   *How to get it:* `gram.py` builds `vocab` from `Counter(words).most_common()`; print the first ten and look at them. They are the words every chapter of any book uses in about the same proportions.
