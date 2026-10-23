# T04 · The Determinant, and a Singular System You Built Yourself

**Meetings:** D27 · **Points:** 10 pts

**Watch — 22 min**
[3Blue1Brown, Essence of Linear Algebra, Ch. 6: The determinant](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (10:03, 10m)
[3Blue1Brown, Essence of Linear Algebra, Ch. 7: Inverse matrices, column space and null space](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (12:09, 12m)
 · Paper, and your T02 hand page.

**During the video**

**Ch. 6 · draw the square every time.** Each time he gives a transformation and its determinant, sketch the unit square and the shape it becomes, and write the area next to it. Do it for your three T02 matrices too, and write the three areas down before Step 2 prints them. Commit the page as `linear/video-notes-t04.jpg`.

**Ch. 7 · name what happened in T01.** When he says a system `A x = b` has a solution only if `b` is in the column space of `A`, stop and write under your two T01 systems which of the two pictures he draws each one is: `b` off the line (no solution) or `b` on it (infinitely many). The two systems on your T01 page were both the determinant-zero kind, and Step 3 builds a third one from your own counts.

**Notes**

**The determinant is the area scaling factor.** Determinant 0 means space got flattened onto a line or a point, which is exactly "singular" from T01. A negative determinant means orientation flipped: î and ĵ swapped which side of each other they are on. These are the same fact told three ways.

**`np.linalg.det` does floating-point arithmetic.** A determinant that is exactly 1 on paper can print as `0.9999999999999998`, and one that is exactly 0 can print as `2.2e-16` or, on a matrix of counts, `-2e-11`. Compare with `np.isclose`, never `==`. Step 3 shows you what "zero" looks like on your own counts.

**`solve` and `lstsq` ask two different questions.** `solve` wants exactly as many equations as unknowns and finds the one answer; with more equations than unknowns there is usually no exact answer, and `lstsq` finds the closest point instead.

**Walkthrough — determinants on your matrices and your counts**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T03 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/determinant
```

*If T03 is not approved yet:* `git switch dev/composition && git switch -c dev/determinant`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Ch. 6 watched, the unit square drawn every time, my three areas written down first; Ch. 7 watched
- [ ] Step 2: det(AB) = det(A) det(B) on two pairs
- [ ] Step 3: span.py, chapter 5 in the span of 2 and 7, the singular system
- [ ] Extension: prediction in the log, dependent.py run
- [ ] Fill every slot in evidence/T04.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. `det(AB) = det(A) · det(B)`, on a pair that is not 1.**

T03's pair had every determinant at 1, which proves nothing. Run the visualizer with your scale as `A` and your shear as `B`, then with your scale and the swap:

```bash
uv run python linear/transform_viz.py 3 0 0 0.5  1 1 0 1
uv run python linear/transform_viz.py 3 0 0 0.5  0 1 1 0
```

*You should see* `det(A) det(B)`, `det(AB)` and `det(BA)` agree on each run: `1.5000000000000002` three times for scale and shear, `-1.5000000000000002` three times for scale and the swap. The area scale of doing two things is the product of the area scales, whichever order you do them in, even though the order changes where everything goes. Paste the two determinant lines under `## Step 2` in `evidence/T04.md`, and compare the areas with the ones you wrote during Ch. 6.

**Step 3. Is chapter 5 in the span of chapters 2 and 7?**

Open `linear/span.py` and read it; it asks T01's Ch. 2 question of your own chapters and then builds a singular system. Then:

```bash
uv run python linear/span.py
```

*You should see* four things. On Homer, with `jove`:

```text
2 words  coefficients [0.901 0.312]  check [1882.   85.] vs [1882.   85.]
 3 words  coefficients [0.413 0.669]  relative miss 0.0041
10 words  coefficients [0.096 0.935]  relative miss 0.0691
50 words  coefficients [0.102 0.926]  relative miss 0.0905
singular system
 [[1419. 2838.]
 [  58.  116.]]
 rank 1  det -2.0165202840871614e-11
 solve returned [ 1.13651225e+15 -5.68256124e+14]
```

First, with two words the check prints back exactly chapter 5's vector: two chapter vectors that are not on the same line span the whole plane, so every chapter is reachable, and this line is true for everyone. Second, the signs and sizes of the two coefficients. If chapter 5's slope lies between chapter 2's and chapter 7's, both coefficients are positive and under 1, and chapter 5 is a blend of the other two; if it lies outside, one coefficient is negative and both can be large, because reaching a direction beyond two near-parallel arrows takes a lot of one minus a lot of the other. Third, from three words on, the relative miss is not zero: two vectors span a plane, and a point in fifty dimensions is almost never in it. Fourth, the singular system, chapter 2 beside chapter 2 twice: `rank 1`, because the two columns span one line, and then one of two endings. Either `LinAlgError: Singular matrix`, or a `det` like `-2e-11` instead of `0.0` and a "solution" with entries around `1e15`. Which one you get depends on how rounding falls on your particular counts. They mean the same thing: the exact answer is "no solution", and floating point sometimes cannot tell zero from almost zero. A coefficient of a quadrillion is how that looks when it does not raise. Paste all of it under `## Step 3`.

*If it broke:* a `LinAlgError` on the 2-word solve itself means two of your chapter rows are exactly proportional, which happens only with tiny counts. Go back to T01 Step 4 and pick a more frequent `WORD2`.

**Extension — the most nearly dependent pair (ASSIGNED)**

Two chapter vectors as the columns of a matrix have a determinant, and for arrows scaled to length 1 that determinant is the sine of the angle between them: 0 means parallel, which is "dependent", and 1 means perpendicular. Before you run anything, write in your log which pair of your ten chapters you expect to be closest to parallel. Commit the log. I will check your commit timestamps.

```bash
uv run python linear/dependent.py
```

*You should see* 45 lines, one per pair, smallest first, with chapters 2 and 7 marked, then a `smallest` and `largest` line, and `linear/dependent.png`, the closest pair drawn as the columns of a matrix: a gold sliver instead of a square. On Homer the smallest is chapters 0 and 9 at `0.00122` and the largest chapters 1 and 7 at `0.03177`, the two the T01 extension named. All 45 are small for two slices of one text; `0.00122` means the two arrows are about seven hundredths of a degree apart. Paste the three lines under `## Extension` in `evidence/T04.md` and say what the smallest one says about those two chapters.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T04.md` (the questions are below) and run `uv run python scripts/check_evidence.py T04` until it says `all slots filled`. Then:

```bash
git add linear/ evidence/T04.md
git commit -m "T04: det(AB) on two pairs, span of two chapters, the most dependent pair"
git push -u origin dev/determinant
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body says which of the four `span.py` findings you could not have predicted.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/T04.md` with every slot filled + `linear/composition.png` from the swap run + `linear/dependent.png` + `linear/video-notes-t04.jpg`.

**Reflection Questions** (answer under `## Reflection` in `evidence/T04.md`)

1. Paste the determinant line for scale and the swap. Say what the minus sign means for the gold square, and why "area scale" alone could not have predicted `-1.5` while "area scale with orientation" could.

   *How to get it:* in `linear/composition.png` from the second Step 2 run, green and red have changed sides in the `AB` and `BA` panels. The swap flips orientation, which multiplies the determinant by `-1`; the scale multiplies area by 1.5. Area alone has no sign.

2. Paste the `singular system` block: the matrix, the rank, the determinant, and whichever ending you got. Say which kind of singular it is, given the right-hand side it tried to reach, and write a right-hand side in your own counts that would have made it the other kind.

   *How to get it:* the columns both lie on the line through chapter 2's vector. `span.py` asked for chapter 5, which is off that line, so there is no solution. Any right-hand side on the line, such as chapter 2's own counts or three times them, has infinitely many.

3. From three words on, the relative miss is not zero. Paste the four coefficient lines and say what changed between two words and three that made chapter 5 unreachable.

   *How to get it:* two columns span at most a plane. With two words the whole space is a plane, so the span is everything; with three words the space has a third dimension the two columns do not reach, and chapter 5 has a component in it. Ch. 7 calls the reachable set the column space.
