# T05 · Dot Product and Matrix Multiplication, Built From Nothing

**Meetings:** D28 · **Points:** 15 pts

**Watch — 17 min**
[3Blue1Brown, Essence of Linear Algebra, Ch. 7: Inverse matrices, column space and null space](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (12:09, 12m)
[3Blue1Brown, Essence of Linear Algebra, Ch. 8: Nonsquare matrices as transformations between dimensions](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (4:27, 5m)
 · Paper, and your T04 `span.py` paste.

**During the video**

**Ch. 7 · one word.** When he says a system `A x = b` has a solution only if `b` is in the column space of `A`, write on your page which of his two pictures your T04 singular system is: `b` off the line (no solution) or `b` on it (infinitely many). The column space is the span of the columns, Ch. 2's word for the same set.

**Ch. 8 · write the rule.** A 2×3 matrix takes a vector with three coordinates and lands it in two. Write the dimension rule on your page before Step 2 needs it: `(n × m) @ (m × p) → (n × p)`. The inner dimensions must match, and they vanish. Put both answers under `## During the video` in `evidence/T05.md`.

**Notes**

**NumPy is banned in `linear/linalg.py`.** Lists of lists only. NumPy is allowed in the tests, to check your answers, and in `gram.py`, to time yourself against it.

**Row-major means `A[i][j]` is row `i`, column `j`.** Getting column `j` out of `B` is `[row[j] for row in B]`. Most people's first `matmul` is silently transposed, and a square symmetric test matrix cannot catch it, because a symmetric matrix is its own transpose. The tests use a non-square, non-symmetric pair for that reason.

**Raise a clear `ValueError` on a dimension mismatch** instead of letting Python throw an `IndexError` twelve frames deep. The mismatch test is the one people skip and the one that catches their bugs.

**`pytest` finds `linear/linalg.py` because it puts the test file's own folder on the import path.** Run it from the repo root, as `uv run pytest linear -q`. `python linear/test_linalg.py` runs nothing and prints nothing, which looks like success.

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
- [ ] Ch. 7 watched, my singular system named; Ch. 8 watched, the dimension rule written
- [ ] Step 2: dot and matmul typed into linear/linalg.py
- [ ] Step 3: five tests green
- [ ] Step 4: prediction written first, the transposed bug, which test caught it
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

`dot` is the whole algebraic definition: multiply matching coordinates and add. `matmul` is built from nothing but `dot`: every entry of the product is row `i` of `A` against column `j` of `B`, which is Ch. 8's rule with the arithmetic filled in. The loop variable `k` is the inner dimension, and it never appears in the shape of `C`. Tomorrow's Ch. 9 is about what that sum means geometrically.

**Step 3. Five tests against NumPy.**

`linear/test_linalg.py` shipped with the repo: two `dot` cases, a square product, a 2×3 times 3×4, and a mismatch that has to raise. From the repo root:

```bash
uv run pytest linear -q
```

*You should see* `5 passed`. Paste the last line under `## Step 3` in `evidence/T05.md`. *If it broke* with `ERROR: file or directory not found: linear`, you are not in the repo root; `cd` back to it. *If it broke* with `ModuleNotFoundError: No module named 'linalg'`, you put an `__init__.py` in `linear/`; delete it, the folder is not a package.

**Step 4. Break it on purpose.**

In `matmul`, change `col = [B[k][j] for k in range(m)]` to `col = [B[j][k] for k in range(m)]`, the transposed bug. Before you run the tests, write under `## Step 4` in `evidence/T05.md` which of the five you expect to fail, and commit that file. Then run `uv run pytest linear -q`.

*You should see* `2 failed, 3 passed`: the square test fails on a wrong answer (`assert False`), the non-square test fails with `IndexError: list index out of range`, and the two `dot` tests and the mismatch test pass, because they never reach the bad line. Paste the `FAILED` lines and the summary line under `## Step 4`. Put the line back, run again to `5 passed`, and commit with a message that names the test that caught it. *If it broke* with the tests still failing after you put the line back, delete `linear/__pycache__`; Python kept the old compiled file because the edit landed within the same second.

**Step 5. Two chapters, one number.**

```bash
uv run python linear/gram.py
```

`gram.py` takes chapters 2 and 7 over the 50 most common words of your corpus and calls your `dot` on them. *You should see* a dot product in the millions, a cosine above 0.9, a one-row `matmul` that prints the same dot product wrapped as `[[...]]`, and the first ten of the fifty words. On Homer: `chapter 2 . chapter 7 = 7,470,275   cosine = 0.9538`, `one-row matmul: [[7470275.0]]` and `top 10 of the 50: ['the', 'and', 'of', 'to', 'he', 'in', 'his', 'a', 'you', 'i']`. The dot product is in the millions because `the` alone appears a thousand or two times in each chapter and contributes the product of its two counts, `1419 × 1935` on Homer, over a third of the total. The `matmul` line is Ch. 8's rule at its smallest: a 1×50 matrix times a 50×1 matrix is a 1×1 matrix, and the one entry is a dot product. Paste all three lines under `## Step 5`. *If it broke* with `IndexError` inside `matmul`, the Step 4 line is still transposed, or `linear/__pycache__` kept the transposed version; put the line back and delete that folder.

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
`evidence/T05.md` with every slot filled + `linear/linalg.py` (no NumPy) + the commit that names the test that caught the bug.

**Reflection Questions** (answer under `## Reflection` in `evidence/T05.md`)

1. Name the test that caught the bug and the one you predicted would. Explain from the actual matrices in `test_linalg.py` why the square test gave a wrong answer rather than a crash, while the non-square one crashed.

   *How to get it:* open `linear/test_linalg.py`. In the square test `B` is 2×2, so `B[j][k]` with `j` and `k` both under 2 is always a real entry, just the wrong one; in the non-square test `B` is 3×4, so `B[j][k]` with `j` up to 3 asks for a fourth row that does not exist.

2. The one-row `matmul` printed the same number as `dot`, wrapped as `[[...]]`. Say why a 1×50 matrix times a 50×1 matrix is a dot product, using the loops in your own `matmul`.

   *How to get it:* with `n = 1` and `p = 1` the two outer loops each run once, `col` is the whole of `B` as a list, and the one entry of `C` is `dot(A[0], col)`.

3. Your cosine between chapters 2 and 7 is above 0.9. Say what in the 50 most common words makes two slices of one text look that alike, and what you would have to change about the vocabulary to tell them apart. T06 checks your answer.

   *How to get it:* the `top 10 of the 50` line in your Step 5 paste. They are the words every chapter of any book uses in about the same proportions, so any two chapters agree on most of the fifty coordinates before a single content word is counted.
