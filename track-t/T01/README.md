# T01 · Vectors, and Your Corpus in Two Words

**Meetings:** D24 · **Points:** 15 pts

**Watch — 10 min**
[3Blue1Brown, Essence of Linear Algebra, Ch. 1: Vectors, what even are they?](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (9:52, 10m)
 · Paper and a pencil on the desk.

**During the video**

**Ch. 1 · draw.** When he adds two vectors tip to tail, stop the video and draw your own pair: `(2, 1)` and `(-1, 3)`, the second one starting at the tip of the first, and the sum from the origin to where you ended up. Label the sum with its coordinates. When he gets to scalar multiplication, draw `2 · (2, 1)` on the same axes. Keep the page; Reflection 1 is about the arrows on it.

**Notes**

**A vector is a list of numbers and an arrow, and they are the same thing.** Today's vectors are word counts: a chapter is `[how many times it says the, how many times it says your word]`, and that pair of numbers is an arrow from the origin. Adding two chapters' counts adds the arrows. Scaling a chapter by 2 doubles the arrow.

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

Hooks do not travel with a clone, which is why this repo has its own `setup.sh` and you just ran it. *You should see* it end with the environment check and an offer to test the guard with a fake key; you tested the guard in the foundations repo, so skip the test today. Then `uv run python -c "import numpy, matplotlib; print('ok')"` prints `ok`. Both are in `pyproject.toml` already; you add nothing today. Read `linear/README.md`: one line per script and which assignment runs it.

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
- [ ] Ch. 1 watched, the page drawn
- [ ] gpt repo accepted, setup.sh run, corpus fetched and the sha256 matches
- [ ] Step 4: WORD2 chosen, chapter_vectors.py run, output in evidence/T01.md
- [ ] Extension: prediction written first, ratios run
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

*You should see* the same 64 hex characters twice. Paste both lines under `## Step 3` in `evidence/T01.md`, and put the `Title:` line from `data/SOURCE.md` on the `Corpus:` line at the top of that file. *If it broke* with different hashes, the source changed since September; do not edit the hash to match. Run the script in the foundations repo too, `diff` the two files, and tell me which one moved. `git status` should show `scripts/fetch_corpus.sh` and `data/SOURCE.md` as new, and nothing under `data/` but `SOURCE.md`.

**Step 4. Cut the corpus into chapters and count two words.**

Your corpus may not have chapters, so `linear/chapter_vectors.py` makes them: ten equal slices by word count, numbered 0 to 9 the way Python counts, so "chapter 3" means slice 3 and the first row of anything it prints is chapter 0. Open it and read the top half, down to the `if __name__` line; the bottom half runs the checks. `count_vector` turns a chapter into a vector by counting each word in a vocabulary, and `V2` is the ten chapters as vectors over a vocabulary of two words: `WORD1` is `the`, and `WORD2` is yours. Pick a content word that matters in your corpus (a name, a place, a topic) and appears in all ten chapters, put it in the `WORD2 =` line and on the `WORD2:` slot at the top of `evidence/T01.md`, and run it:

```bash
uv run python linear/chapter_vectors.py
```

*You should see* a word count, an array of ten rows with two numbers each, one row per chapter, and then `add:   True` and `scale: True`. The first column is in the hundreds or thousands for almost any English corpus; the second is smaller. On Homer with `WORD2 = "jove"`: `284435 words, 10 chapters of 28443`, row 0 is `[1352.   43.]` and row 7 is `[1935.  105.]`. The two `True`s are the script checking something 3Blue1Brown said: adding vectors means adding coordinates, and for count vectors that has a meaning you can test. Over the 50 most common words of your corpus (`vocab50`, so these vectors have fifty entries, not two), the counts of two chapters glued together (`A + B` on the lists) equal the sum of the two count vectors (`vA + vB`), and the counts of chapter 2 twice equal `2 * vA`. That holds for every corpus. Write one line in `linear/NOTATION.md` saying what `+` does to two lists and what it does to two vectors, because Python uses the same symbol for both.

Paste everything it printed under `## Step 4` in `evidence/T01.md`. *If it broke* with `AssertionError: ... put your word in WORD2`, the `WORD2 =` line still says `CHANGE_ME`. A line `a chapter never says 'yourword'` between the array and the `add:` line is a warning, not a crash: a vector lying flat on an axis makes the extension trivial, so pick a more common word. The same goes for an array printed as `1.352e+03`: NumPy switches to that notation when a count is 0 or 1, and a word that rare is the wrong word.

**Extension — how far apart do your ten arrows point? (ASSIGNED)**

Every chapter arrow has a slope: the count of `WORD2` over the count of `WORD1`. Before you run anything, write under `## Extension` in `evidence/T01.md` which two chapters you expect to have the most different slopes, and why, from what you know about your corpus. Then:

```bash
uv run python linear/chapter_vectors.py ratios
```

*You should see* ten lines, one slope per chapter, and a `furthest apart` line naming the smallest and the largest. On Homer the slopes run from `0.0224` (chapter 1) to `0.0543` (chapter 7), and the two furthest apart are chapters 1 and 7: chapter 1 is Odysseus telling his own wanderings to the Phaeacians (Odyssey VI to XI), where he, not the poet, is speaking and Jove is named less; chapter 7 is the Iliad's middle books (XIII to XVI), where the gods run the battle. Paste the lines under your prediction, say whether it held, and say what in your corpus explains the two chapters that won. T02 uses those two chapters.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T01.md` (the questions are below) and run `uv run python scripts/check_evidence.py T01` until it says `all slots filled`. Then:

```bash
git add linear/ scripts/fetch_corpus.sh data/SOURCE.md evidence/T01.md
git status                      # data/corpus.txt must NOT be listed
git commit -m "T01: chapter vectors, add and scale, ten slopes"
git push -u origin dev/vectors
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the word you chose and the word you almost chose instead.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push -u origin <your-username>-track
```

**Deliverable**
`evidence/T01.md` with every slot filled + `linear/chapter_vectors.py` with your `WORD2` + `linear/NOTATION.md` + `scripts/fetch_corpus.sh` and `data/SOURCE.md` committed. `data/corpus.txt` is never committed.

**Reflection Questions** (answer under `## Reflection` in `evidence/T01.md`)

1. Paste row 0 and row 7 of your array. Say what each of the two numbers in a row counts, and say which of the two arrows is longer and which is steeper, with the numbers that tell you.

   *How to get it:* the rows are in your Step 4 paste; row 7 is the eighth line of the array. Length comes from the first number, since `the` dominates every row. Steepness is the second number divided by the first, which is the slope line the extension printed for chapters 0 and 7.

2. Paste your `add:` and `scale:` lines. `A + B` and `vA + vB` use the same symbol and are not the same operation; say what each one does, and why the count of two chapters glued together had to equal the sum of the two counts.

   *How to get it:* `A + B` on two lists is the second list appended to the first (open `chapter_vectors.py`: `chapters[2]` and `chapters[7]` are lists of words); `vA + vB` on two NumPy arrays adds matching coordinates. Counting is a sum over the words, and the sum over a glued list is the sum over each part.

3. Your ten slopes all came out small. Say why, from what both numbers in every row are counting. Then say whether yours came out close to each other, and what about your corpus makes that so.

   *How to get it:* look at the ten slope lines. Every chapter of any book says `the` far more often than it says any content word, so every slope is small. Whether they are close depends on the file: ten slices of one story use a name at about the same rate, and Homer's run from 0.02 to 0.05, but a file that is really several texts on different subjects, or two books glued together, spreads out, because `WORD2` belongs to some slices and not others.
