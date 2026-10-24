# T06 · Which of Your Chapters Are Alike, and What the Common Words Hid

**Meetings:** D29 · **Points:** 10 pts

**Watch — 14 min**
[3Blue1Brown, Essence of Linear Algebra, Ch. 9: Dot products and duality](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (14:12, 14m)
 · Paper.

**During the video**

**Ch. 9 · draw.** When he projects one vector onto another, stop and do it with your T01 pair: `v = (2, 1)` and `w = (-1, 3)`. Draw both from the origin, drop the perpendicular from `w` onto the line through `v`, and write the dot product next to it: `2·(-1) + 1·3 = 1`. Write whether the angle between them is under or over 90°, and check that the sign of `1` agrees. The duality half of the chapter, where a 1×2 matrix becomes a projection, has nothing to draw; it is why T05's `dot` and one-row `matmul` gave the same number. Photograph the page and save it as `linear/video-notes-t06.jpg`; it goes in with the final commit.

Linear algebra closes today: the nine chapters are watched, and `linalg.py` is yours. Today runs it on every pair of your chapters, times it against NumPy, and writes the findings.

**Notes**

**Cosine is the dot product with the lengths divided out.** Two chapter vectors with a cosine of 1 point the same way, whatever their lengths. T05's `0.95` between chapters 2 and 7 was measured over the 50 most common words, and today asks whether those words were doing the measuring.

**`gram.py bands` times your `matmul` against NumPy.** Its `matmul` is yours, in pure Python, and the timing rows run it on vocabularies of 50, 500 and 2,000 words, five times each, keeping the fastest (NumPy gets twenty tries, because its runs are so short that a few microseconds of noise would swing the ratio). The whole thing finishes in a second or two; the point is the ratio column, not the wait. NumPy does the same products in microseconds; that gap is the reason the rest of the track uses NumPy and PyTorch for the arithmetic and keeps your `linalg.py` as the thing you can read.

**Walkthrough — every chapter against every chapter**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T05 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/bands
```

*If T05 is not approved yet:* today runs its `linalg.py`, so `git switch dev/matmul && git switch -c dev/bands`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Ch. 9 watched, the projection drawn
- [ ] Step 2: two predictions in the log, gram.py bands run
- [ ] Step 3: the timing table read
- [ ] Step 4: linear/FINDINGS.md
- [ ] Fill every slot in evidence/T06.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Two vocabulary bands.**

Before you run anything, write two predictions in your log and commit it: the smallest cosine you expect between any two of your ten chapters using the 50 most common words, and whether that number goes up or down if you throw the 50 most common words away and use ranks 51 to 250 instead. I will check your commit timestamps. Then:

```bash
uv run python linear/gram.py bands
```

*You should see* T05's three lines again, then two band lines, then three timing rows. On Homer:

```text
top 50        min 0.954  mean 0.980  max 0.998  most alike 7,8  least alike 2,7
ranks 51-250  min 0.678  mean 0.833  max 0.960  most alike 0,2  least alike 3,8
V=   50  pure Python     0.21 ms   NumPy   0.002 ms   ratio     134x   same answer: True
V=  500  pure Python     2.34 ms   NumPy   0.006 ms   ratio     371x   same answer: True
V= 2000  pure Python    10.72 ms   NumPy   0.022 ms   ratio     496x   same answer: True
```

Whatever your corpus: with the top 50 words every pair of chapters is very alike, all of them above 0.9, because every chapter of any book is mostly `the`, `of` and `and` in about the same proportions. With ranks 51 to 250 the numbers spread out and drop, and the most-alike and least-alike pairs change. On Homer the least alike pair in the top-50 band is chapters 2 and 7, the two T05 dotted, and in the second band it is chapters 3 and 8, one in the Odyssey and one in the Iliad. Paste the two band lines under `## Step 2` in `evidence/T06.md`.

**Step 3. The timing.**

Your `matmul` and NumPy's, on the 10×V chapter matrix times its transpose, at three vocabulary sizes. *You should see* a ratio in the hundreds at every size, the `V=2000` ratio larger than the `V=50` one, and `same answer: True` on all three rows. Your milliseconds will differ from Homer's and so will your ratios; the middle row can land on either side of the last, because NumPy's time there is a few microseconds and jitters. If your corpus has fewer than 2,000 distinct words, the last row uses all of them and prints the number it used. Paste the three rows under `## Step 3`.

**Step 4. Write `linear/FINDINGS.md`.**

Half a page, four parts: both band lines; which pair of chapters is most alike in the ranks 51-250 band, with one sentence on why from what you know about your corpus; your two predictions, copied from the log unchanged, with one sentence each on whether they held; and the timing table with one sentence under it giving the `V=50` and `V=2000` ratios.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T06.md` (the questions are below) and run `uv run python scripts/check_evidence.py T06` until it says `all slots filled`. Then:

```bash
git add linear/ evidence/T06.md
git commit -m "T06: chapter similarity in two bands, pure Python against NumPy, findings"
git push -u origin dev/bands
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the one chapter of the nine you would rewatch.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/T06.md` with every slot filled + `linear/FINDINGS.md` + `linear/video-notes-t06.jpg`. With T01 to T05 merged, `linear/` holds the whole week: your matrices, the visualizer's pictures, `linalg.py`, and the findings.

**Reflection Questions** (answer under `## Reflection` in `evidence/T06.md`)

1. Paste both band lines next to your two predictions. Say what the top-50 band was really measuring, using its min and max as evidence.

   *How to get it:* a min of `0.95` and a max of `0.998` means no pair of chapters differed by much; the band measured how much of each chapter is `the`, `of` and `and`, which is the same for all of them.

2. From your timing table, for `V=2000`: how many multiply-adds did your `matmul` do (n × m × p, with your real shapes), and how many per second is that? Say what that rate predicts for one 512×512 @ 512×512 product.

   *How to get it:* the product is 10×2000 times 2000×10, so `10 × 2000 × 10 = 200,000` multiply-adds, divided by your pure-Python seconds (`10.72 ms` is `0.01072` s). A 512×512 product is `512 × 512 × 512 = 134,217,728` multiply-adds; divide by your rate for the seconds. On Homer's machine that is about nineteen million a second, so about seven seconds for one product that NumPy does in a few milliseconds.

3. Linear algebra is done. Name the one idea from the nine chapters you would be least able to explain to a classmate, and the chapter it is in.

   *How to get it:* reread your `NOTATION.md` and your six pages of video notes; the idea with the fewest words next to it is usually the one.
