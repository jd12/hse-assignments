# T01 · Vectors, Span, and Your Corpus in Two Words

**Meetings:** D24 · **Points:** 10 pts

**Watch — 20 min**
[3Blue1Brown, Essence of Linear Algebra, Ch. 1: Vectors, what even are they?](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (9:52, 10m)
[3Blue1Brown, Essence of Linear Algebra, Ch. 2: Linear combinations, span, and basis vectors](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (9:59, 10m)
 · Paper and a pencil on the desk. Both chapters are pictures, and you draw them.

**During the video**

There is no code in either chapter. What you do during the video is draw, on one sheet of paper that you photograph at the end and commit as `linear/video-notes-t01.jpg`.

**Ch. 1 · draw.** When he adds two vectors tip to tail, stop the video and draw your own pair: `(2, 1)` and `(-1, 3)`, the second one starting at the tip of the first, and the sum from the origin to where you ended up. Label the sum with its coordinates. When he gets to scalar multiplication, draw `2 · (2, 1)` and `-1 · (2, 1)` on the same axes.

**Ch. 2 · draw.** When he talks about the span of two vectors, draw three small sets of axes side by side and fill in the span for each case: two vectors pointing in different directions, two vectors on the same line, and both vectors zero. Write under each one "whole plane", "a line" or "a point". Write î and ĵ on the first picture with their coordinates.

**Notes**

**Singular** means a system has either no solution or infinitely many; **non-singular** means exactly one. A singular system is not a broken one. It is a system whose column vectors do not span enough space to reach every right-hand side.

**Columns or rows.** 3Blue1Brown writes a vector as a column. Karpathy and PyTorch write it as a row. Start `linear/NOTATION.md` today with that one line, and add to it every time a source does it differently. It will save you in micrograd.

**Run every script from the repo root.** The scripts open `data/corpus.txt` by a path relative to where you are standing, so `cd linear && uv run python chapter_vectors.py` fails with `FileNotFoundError: 'data/corpus.txt'` even though the file is right there.

**Walkthrough — ten chapters, two words**

**Step 1. Accept the gpt repo.**

Everything in the theory track from today to June lives in one repo. Accept it once:

**[GPT](https://classroom50.org/Sierra-Canyon/hse-2026-2027/assignments/gpt/accept)**

```bash
cd ~/version_control
git clone https://github.com/Sierra-Canyon/hse-2026-2027-gpt-<your-username>.git
cd hse-2026-2027-gpt-<your-username>
./setup.sh
git switch -c dev/vectors
```

Hooks do not travel with a clone, so `setup.sh` again. *You should see* it end by asking you to try committing a fake key; do it and watch it get refused. Then `uv run python -c "import numpy, matplotlib; print('ok')"` prints `ok`. Both are in `pyproject.toml` already; you add nothing today.

**Step 2. Close the Unit 0 log, open the track log.**

On GitHub, merge your student log's `<your-username>-unit0` pull request and delete the branch. Then:

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git switch main && git pull
git switch -c <your-username>-track
bash scripts/start-entry.sh
```

That branch stays open for every theory assignment until Wed Dec 9. Under the timestamp, write today's checklist:

```markdown
- [ ] Ch. 1 and Ch. 2 watched, the two pages drawn and photographed
- [ ] gpt repo accepted, setup.sh run, corpus fetched and the sha256 matches
- [ ] Steps 4–5: WORD2 chosen, chapter_vectors.py and chapters_plot.py run, output in evidence/T01.md
- [ ] Step 6: two systems by hand
- [ ] Extension: prediction in the log, ratios run
- [ ] Fill every slot in evidence/T01.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 3. Bring your corpus.**

The corpus is the one you locked at A05b, and the same file has to land here byte for byte. Copy the two files that describe it and rebuild it:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
cp ../hse-2026-2027-foundations-<your-username>/scripts/fetch_corpus.sh scripts/
cp ../hse-2026-2027-foundations-<your-username>/data/SOURCE.md data/
bash scripts/fetch_corpus.sh
shasum -a 256 data/corpus.txt
grep -i sha256 data/SOURCE.md
```

*You should see* the same 64 hex characters twice. Paste both lines under `## Step 3` in `evidence/T01.md`. *If it broke* with different hashes, the source changed since September; do not edit the hash to match. Run the script in the foundations repo too, `diff` the two files, and tell me which one moved. `git status` should show `scripts/fetch_corpus.sh` and `data/SOURCE.md` as new and nothing else under `data/`.

**Step 4. Cut the corpus into chapters and count two words.**

Your corpus may not have chapters, so `linear/chapter_vectors.py` makes them: ten equal slices by word count, and "chapter 3" means slice 3. Open it and read it top to bottom; it is thirty lines. `count_vector` turns a chapter into a vector by counting each word in a vocabulary, and `V2` is the ten chapters as vectors over a vocabulary of two words: `WORD1` is `the`, and `WORD2` is yours. Pick a content word that matters in your corpus (a name, a place, a topic) and appears in all ten chapters, put it in the `WORD2 =` line, and run it:

```bash
uv run python linear/chapter_vectors.py
```

*You should see* a word count, a `(10, 2)` array with one row per chapter, and then `add:   True` and `scale: True`. The first column is in the hundreds or thousands for almost any English corpus; the second is smaller. On Homer with `WORD2 = "jove"`: `284435 words, 10 chapters of 28443`, a first row of `[1352.   43.]`, a row 7 of `[1935.  105.]`. The two `True`s are the script checking something 3Blue1Brown said: adding vectors means adding coordinates, and for count vectors that has a meaning you can test. The counts of two chapters glued together (`A + B` on the lists) equal the sum of the two count vectors (`vA + vB`), and the counts of chapter 2 twice equal `2 * vA`. That holds for every corpus. Write one line in `linear/NOTATION.md` saying what `+` does to two lists and what it does to two vectors, because Python uses the same symbol for both.

Paste everything it printed under `## Step 4` in `evidence/T01.md`. *If it broke* with `a chapter never says 'yourword'`, a vector lying flat on an axis makes the extension trivial; pick a more common word. `AssertionError: ... put your word in WORD2` means the `WORD2 =` line still says `CHANGE_ME`.

**Step 5. Draw two chapters.**

```bash
uv run python linear/chapters_plot.py
```

*You should see* one `saved` line with chapter 2, chapter 7 and their sum, and `linear/chapters.png`: two arrows that point in nearly the same direction, with the sum almost on top of both. On Homer: `chapter 2 [1419. 58.]   chapter 7 [1935. 105.]   sum [3354. 163.]`. Two slices of one book use `the` and your word in roughly the same proportion, which makes two chapter vectors almost the second picture on your Ch. 2 page. Open the picture and check it against the page.

**Step 6. Two systems, by hand.**

On paper, no code. For each system, say singular or non-singular and how you knew, and if it is singular, say whether it has no solution or infinitely many. Then restate it as a span question, "is **b** in the span of these column vectors?", with the column vectors and **b** written out.

```
(i)   2x +  y = 5        (ii)   x + 2y = 3
      4x + 2y = 9               2x + 4y = 6
```

Both are singular, and they are singular in different ways. "The determinant is zero" is a computation, not a reason; the reason is a sentence about what the columns span and whether **b** is in it. Photograph the page as `linear/systems.jpg`, or type it into `linear/SYSTEMS.md`.

**Extension — how far apart do your ten arrows point? (ASSIGNED)**

Every chapter arrow has a slope: the count of `WORD2` over the count of `WORD1`. Before you run anything, write in today's log entry which two chapters you expect to have the most different slopes, and why, from what you know about your corpus. Commit the log. I will check your commit timestamps. Then:

```bash
uv run python linear/chapter_vectors.py ratios
```

*You should see* ten lines, one slope per chapter, and a `furthest apart` line naming the smallest and the largest. On Homer the slopes run from `0.0224` (chapter 1) to `0.0543` (chapter 7), and the two furthest apart are chapters 1 and 7: the Odyssey's early books are about Telemachus and the suitors, and the Iliad's middle books are full of gods. Paste the lines under `## Extension` in `evidence/T01.md`, say whether your prediction held, and say what in your corpus explains the two chapters that won. T02 uses those two chapters.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T01.md` (the questions are below) and run `uv run python scripts/check_evidence.py T01` until it says `all slots filled`. Then:

```bash
git add linear/ scripts/fetch_corpus.sh data/SOURCE.md evidence/T01.md
git status                      # data/corpus.txt must NOT be listed
git commit -m "T01: chapter vectors, add and scale, two systems by hand"
git push -u origin dev/vectors
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the thing on your Ch. 2 page you are least sure of.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push -u origin <your-username>-track
```

**Deliverable**
`evidence/T01.md` with every slot filled + `linear/chapter_vectors.py` with your `WORD2` + `linear/chapters.png` + `linear/systems.jpg` (or `SYSTEMS.md`) + `linear/video-notes-t01.jpg` + `linear/NOTATION.md` + `scripts/fetch_corpus.sh` and `data/SOURCE.md` committed. `data/corpus.txt` is never committed.

**Reflection Questions** (answer under `## Reflection` in `evidence/T01.md`)

1. Paste chapter 2, chapter 7 and their sum from `chapters_plot.py`. Say what the sum is a count of, in words, and why the two arrows point nearly the same way.

   *How to get it:* the three vectors are on the `saved` line. The sum counts `the` and your word in chapters 2 and 7 taken together, which is what Step 4's `add:   True` checked. The direction of an arrow is its slope from the extension; two slices of one text have close slopes because both use `the` at about the same rate.

2. Paste your `add:` and `scale:` lines. `A + B` and `vA + vB` use the same symbol and are not the same operation; say what each one does, and why the count of two chapters glued together had to equal the sum of the two counts.

   *How to get it:* `A + B` on two lists is the second list appended to the first (open `chapter_vectors.py`: `chapters[2]` and `chapters[7]` are lists of words); `vA + vB` on two NumPy arrays adds matching coordinates. Counting is a sum over the words, and the sum over a glued list is the sum over each part.

3. From your hand page: restate system (i) as a span question using its two column vectors and its right-hand side, and say which of the three span pictures from Ch. 2 it is.

   *How to get it:* the columns of (i) are `(2, 4)` and `(1, 2)` and the right-hand side is `(5, 9)`. Two vectors on the same line span a line; ask whether `(5, 9)` is on it.
