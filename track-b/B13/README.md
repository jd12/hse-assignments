# B13 · Vectors, Span, and What a System of Equations Actually Asks

**Meetings:** D27 · **Points:** 15 pts

>

**Watch — 20 min**
[3Blue1Brown, Essence of Linear Algebra, Ch. 1: Vectors, what even are they?](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (9:52, 10m)
[3Blue1Brown, Essence of Linear Algebra, Ch. 2: Linear combinations, span, and basis vectors](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · whole video (9:59, 10m)
 · Paper and a pencil on the desk, not a laptop note. Both chapters are pictures, and you are going to draw them.

**During the video**

There is no code in either chapter, so there is nothing to type along with. The code-along today is the walkthrough itself. What you do during the video is draw, on one sheet of paper that you photograph at the end and commit as `linear/video-notes.jpg`.

**Ch. 1 · draw.** When he adds two vectors tip to tail, stop the video and draw your own pair: `(2, 1)` and `(-1, 3)`, the second one starting at the tip of the first, and the sum from the origin to where you ended up. Label the sum with its coordinates. Then, when he gets to scalar multiplication, draw `2 · (2, 1)` and `-1 · (2, 1)` on the same axes. Step 4 draws the same page in code.

**Ch. 2 · draw.** When he talks about the span of two vectors, draw three small sets of axes side by side and fill in the span for each case: two vectors pointing in different directions, two vectors on the same line, and both vectors zero. Write under each one "whole plane", "a line", or "a point". Then write î and ĵ on the first picture with their coordinates. Every system on today's page is one of those three pictures.

**Notes**

This is new material. Nothing here is review, so do not skim.

Vocabulary that will trip you: **singular** means the system has either no solution or infinitely many; **non-singular** means exactly one. "Singular" does not mean "special", and a singular system is not a broken one. It is a system whose column vectors do not span enough space to reach every right-hand side.

Keep a running notation sheet from today, `linear/NOTATION.md`, and write down whether each source treats a vector as a column or a row. 3Blue1Brown defaults to columns. Karpathy and PyTorch default to rows. You will get burned by this in B26–B27 if you do not have a sheet.

Run every script from the repo root. The scripts open `data/corpus.txt` by a path relative to where you are standing, so `cd linear && uv run python chapter_vectors.py` fails with `FileNotFoundError: 'data/corpus.txt'` even though the file is right there.

`plt.quiver` rescales arrows unless you tell it not to, and the tip-to-tail picture silently stops adding up. Every `quiver` call below has `angles="xy", scale_units="xy", scale=1`.

**Walkthrough — Two chapters as vectors**

**Step 1. Merge, branch, log.**
Open your AI Agents v2 repo on GitHub and merge the A12 pull request if I have approved it; click **Delete branch**. Track B starts a new repo, so today's branch comes after Step 2. Your log also starts a new branch today, because `<your-username>-unit0` merged at the track election:
```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git switch main && git pull
git switch -c <your-username>-track
bash scripts/start-entry.sh
```
That branch stays open for every Track B assignment until Wed Dec 9.

Under the timestamp, write today's checklist:
```markdown
- [ ] Watch Essence of Linear Algebra Ch. 1 and draw the tip-to-tail page
- [ ] Watch Ch. 2 and draw the three span pictures
- [ ] Accept the gpt repo and run setup.sh
- [ ] Fetch my corpus and match the sha256 in SOURCE.md
- [ ] Steps 4–7: vectors.py, chapter_vectors.py, the add and scale checks, the chapter plot
- [ ] Step 8: the four systems by hand
- [ ] Extension: is chapter C in the span of A and B, at 2, 3, 10 and 50 words
- [ ] Push dev/vectors and open the PR
```

**Step 2. Accept the gpt repo.**

Everything in Track B from today to June lives in one repo. Accept it once:

**[GPT](https://classroom50.org/Sierra-Canyon/hse-2026-2027/assignments/gpt/accept)**

```bash
cd ~/version_control
git clone https://github.com/Sierra-Canyon/hse-2026-2027-gpt-<your-username>.git
cd hse-2026-2027-gpt-<your-username>
./setup.sh
git switch -c dev/vectors
```

Yes, `setup.sh` again: hooks do not travel with a clone.

*You should see* `setup.sh` end by asking you to try committing a fake key. Do it and watch it get refused. Then `uv run python -c "import numpy, matplotlib; print('ok')"` prints `ok`. NumPy and matplotlib are already in `pyproject.toml`; you add nothing today.

**Step 3. Bring your corpus.**

Your corpus is the one you locked at A05b, and the same file has to land here byte for byte. Copy the two files that describe it:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
mkdir -p scripts data
cp ../hse-2026-2027-foundations-<your-username>/scripts/fetch_corpus.sh scripts/
cp ../hse-2026-2027-foundations-<your-username>/data/SOURCE.md data/
bash scripts/fetch_corpus.sh
shasum -a 256 data/corpus.txt
grep -i sha256 data/SOURCE.md
```

*You should see* the same 64 hex characters twice.

*If it broke:* different hashes mean the source changed or your script cleans the text differently than it did in the foundations repo. Do not edit the hash to match. Run the script again in the foundations repo, compare the two files with `diff`, and tell me which one moved. `git status` should show `scripts/fetch_corpus.sh` and `data/SOURCE.md` as new and nothing under `data/` else. If `data/corpus.txt` shows up as untracked, the `.gitignore` in your template is wrong; tell me before you commit anything.

**Step 4. The code-along: vectors in NumPy.**

Create `linear/vectors.py` and type it, do not paste it. It is your Ch. 1 page, drawn by the computer.

```python
import numpy as np
import matplotlib.pyplot as plt

v = np.array([2.0, 1.0])
w = np.array([-1.0, 3.0])

print("v + w  =", v + w)
print("2v     =", 2 * v)
print("-1v    =", -1 * v)

def arrow(start, vec, color, label):
    plt.quiver(start[0], start[1], vec[0], vec[1],
               angles="xy", scale_units="xy", scale=1, color=color, label=label)

origin = np.zeros(2)
arrow(origin, v, "tab:blue", "v")
arrow(v, w, "tab:orange", "w, tip to tail")
arrow(origin, v + w, "tab:green", "v + w")
arrow(origin, 2 * v, "tab:purple", "2v")
arrow(origin, -1 * v, "tab:red", "-v")

plt.xlim(-4, 5); plt.ylim(-2, 5)
plt.gca().set_aspect("equal")
plt.grid(True); plt.legend()
plt.savefig("linear/vectors.png", dpi=120)
print("saved linear/vectors.png")
```

Run it: `uv run python linear/vectors.py`.

*You should see* `v + w  = [1. 4.]`, `2v     = [4. 2.]`, `-1v    = [-2. -1.]`, and a picture in which the green arrow ends exactly where the orange one does.

*If it broke:* if the green arrow stops short of the orange tip, one of your `quiver` calls is missing `scale=1` or `scale_units="xy"`.

**Step 5. Cut your corpus into chapters and count two words.**

Your corpus may not have chapters, so you make them: ten equal slices by word count, and "chapter 3" means slice 3. Create `linear/chapter_vectors.py`:

```python
import re
import numpy as np

WORD1 = "the"
WORD2 = "CHANGE_ME"

text = open("data/corpus.txt", encoding="utf-8").read()
words = re.findall(r"[a-z']+", text.lower())
N = 10
size = len(words) // N
chapters = [words[i * size:(i + 1) * size] for i in range(N)]

def count_vector(chapter, vocab):
    counts = {w: 0 for w in vocab}
    for w in chapter:
        if w in counts:
            counts[w] += 1
    return np.array([counts[w] for w in vocab], dtype=float)

V2 = np.array([count_vector(ch, [WORD1, WORD2]) for ch in chapters])   # (10, 2)

if __name__ == "__main__":
    print(len(words), "words,", N, "chapters of", size)
    print(V2)
```

Pick `WORD2` yourself: a content word that matters in your corpus (a name, a place, a topic) and that appears in all ten chapters. Run it: `uv run python linear/chapter_vectors.py`.

*You should see* a word count and then a `(10, 2)` array, one row per chapter. The first column is in the hundreds or thousands for almost any English corpus; the second is smaller. Every row is a vector you could draw on the Ch. 1 page.

*If it broke:* a row with `0.` in the second column means that chapter never says your word, and a vector lying flat on an axis makes the extension trivial. Pick another `WORD2` until no row has a zero.

**Step 6. Adding and scaling chapters.**

3Blue1Brown says adding vectors means adding coordinates. For count vectors that has a meaning you can test: the counts in two chapters, added, should be the counts of the two chapters glued together. Add this to the bottom of `chapter_vectors.py`, inside the `if __name__` block:

```python
    from collections import Counter
    vocab50 = [w for w, _ in Counter(words).most_common(50)]
    A, B = chapters[2], chapters[7]
    vA, vB = count_vector(A, vocab50), count_vector(B, vocab50)
    print("add:  ", np.array_equal(vA + vB, count_vector(A + B, vocab50)))
    print("scale:", np.array_equal(2 * vA, count_vector(A + A, vocab50)))
```

*You should see* `add:   True` and `scale: True`. This holds for every corpus, and it is why counting is called linear: the count of a sum is the sum of the counts. Write in `linear/NOTATION.md` one line saying what `A + B` means for the lists (concatenation) and what `vA + vB` means for the vectors (coordinate-wise addition), because Python uses the same symbol for both and they are not the same operation.

**Step 7. Draw two chapters.**

Create `linear/chapters_plot.py`:

```python
import matplotlib.pyplot as plt
from chapter_vectors import V2, WORD1, WORD2

vA, vB = V2[2], V2[7]
for start, vec, color, label in ((V2[2] * 0, vA, "tab:blue", "chapter 2"),
                                 (vA, vB, "tab:orange", "chapter 7, tip to tail"),
                                 (V2[2] * 0, vA + vB, "tab:green", "sum")):
    plt.quiver(start[0], start[1], vec[0], vec[1],
               angles="xy", scale_units="xy", scale=1, color=color, label=label)
top = (vA + vB).max() * 1.1
plt.xlim(0, top); plt.ylim(0, top)
plt.xlabel(f"count of '{WORD1}'"); plt.ylabel(f"count of '{WORD2}'")
plt.legend(); plt.savefig("linear/chapters.png", dpi=120)
```

*You should see* two arrows that point in nearly the same direction, because two slices of one book use `the` and your word in roughly the same proportion, with the sum lying almost on top of both. Two vectors that point almost the same way are almost the second picture on your Ch. 2 page; the extension measures how "almost".

**Step 8. The four systems, by hand.**

On paper, no code. For each system, say singular or non-singular and how you knew, and if it is singular, say whether it has no solution or infinitely many. Then re-state the same system as a span question: "is **b** in the span of these column vectors?", with the column vectors and **b** written out.

```
(i)   2x +  y = 5        (ii)   x + 2y = 3        (iii)  3x -  y = 4
      4x + 2y = 9               2x + 4y = 6              6x - 2y = 8

(iv)   x +  y +  z = 6
       2x -  y + 3z = 9
       x + 4y      = 3
```

Do not trust how a system looks. At least one of them is not what it appears to be at first glance, and (iv) needs actual row work. "The determinant was zero" is a computation, not a reason; the reason is a sentence about what the column vectors span and whether **b** is in it. Photograph the page as `linear/systems.jpg`, or type it into `linear/SYSTEMS.md`.

**Extension — Is chapter C in the span of chapters A and B?** *(assigned)*

Before you run anything, write two predictions in today's log entry and commit the log: whether chapter 5's two-word vector can be written exactly as `a · (chapter 2) + b · (chapter 7)`, and whether that gets easier or harder as you use more words. I will check your commit timestamps.

Create `linear/span.py`:

```python
from collections import Counter
import numpy as np
from chapter_vectors import chapters, count_vector, words, WORD1, WORD2

A, B, C = chapters[2], chapters[7], chapters[5]

# 2 words: two equations, two unknowns
vocab = [WORD1, WORD2]
M = np.column_stack([count_vector(A, vocab), count_vector(B, vocab)])
vC = count_vector(C, vocab)
coef = np.linalg.solve(M, vC)
print("2 words  coefficients", coef.round(3), " check", (M @ coef).round(3), "vs", vC)

# more words: more equations than unknowns, so ask for the closest point instead
ranked = [w for w, _ in Counter(words).most_common()]
for vocab in ([WORD1, WORD2, ranked[1]], ranked[:10], ranked[:50]):
    M = np.column_stack([count_vector(A, vocab), count_vector(B, vocab)])
    vC = count_vector(C, vocab)
    coef, *_ = np.linalg.lstsq(M, vC, rcond=None)
    rel = np.linalg.norm(M @ coef - vC) / np.linalg.norm(vC)
    print(f"{len(vocab):2d} words  coefficients {coef.round(3)}  relative miss {rel:.4f}")

# a singular system you built yourself: chapter 2, and chapter 2 twice
M = np.column_stack([count_vector(A, [WORD1, WORD2]), count_vector(A + A, [WORD1, WORD2])])
print("singular system\n", M, "\n rank", np.linalg.matrix_rank(M), " det", np.linalg.det(M))
try:
    print(" solve returned", np.linalg.solve(M, count_vector(C, [WORD1, WORD2])))
except np.linalg.LinAlgError as e:
    print(" LinAlgError:", e)
```

Run it: `uv run python linear/span.py`.

*You should see* four things. First, with two words the check prints back exactly `vC`: two chapter vectors that are not on the same line span the whole plane, so every `C` is reachable, and this line is true for everyone. Second, look at the signs and sizes of the two coefficients. If chapter 5's ratio of your two words lies between chapter 2's and chapter 7's, both coefficients are positive and under 1: chapter 5 is a blend of the other two. If it lies outside, one coefficient is negative and both can be large, because reaching a direction beyond two near-parallel arrows takes a lot of one minus a lot of the other. Say which case you are in; B16 explains the size. Third, from three words on, the relative miss is not zero: two vectors span a plane, and chapter 5 is almost never exactly in it. Fourth, the singular system: `rank 1`, because chapter 2 and chapter 2 twice span one line, and then one of two endings. Either `LinAlgError: Singular matrix`, or a `det` like `2e-11` instead of `0.0` and a "solution" with entries around `1e15`. Which one you get depends on how rounding falls on your particular counts. They mean the same thing: the exact answer is "no solution", and floating point sometimes cannot tell zero from almost zero. A coefficient of a quadrillion is how that looks when it does not raise.

*If it broke:* a `LinAlgError` on the 2-word solve itself means two of your chapter rows are exactly proportional, which happens only with tiny counts. Go back to Step 5 and pick a more frequent `WORD2`.

Now write `linear/FINDINGS.md`: a table with one row per vocabulary size (2, 3, 10, 50), the coefficients and the relative miss; your two predictions from the log, copied in unchanged; and one sentence per prediction saying whether it held. Name the row where the miss got worse and the size of the jump.

Commit, push, open the pull request:

```bash
git add linear/ scripts/fetch_corpus.sh data/SOURCE.md
git commit -m "B13: chapter vectors, add/scale invariants, span of two chapters"
git push -u origin dev/vectors
```

On GitHub: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**. In the PR body, say where this is weakest.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push -u origin <your-username>-track
```

**Deliverable**
`linear/vectors.py` + `linear/chapter_vectors.py` + `linear/chapters_plot.py` + `linear/span.py` + `linear/FINDINGS.md` + `linear/systems.jpg` (or `SYSTEMS.md`) + `linear/video-notes.jpg` + `linear/NOTATION.md`, with `scripts/fetch_corpus.sh` and `data/SOURCE.md` committed.

**Reflection Questions**

1. Paste the `2 words  coefficients` line from `span.py` and rows 2, 5 and 7 of your `V2`. Say what each coefficient means as an instruction ("take this many copies of chapter 2's counts"). Then explain, using those three rows and not a general rule, why your coefficients came out as large or as small as they did, and what a negative number of copies of a chapter means.

2. Paste the prediction lines from your log with their commit time, and your `FINDINGS.md` table beside them. Name the vocabulary size where the relative miss got worse by the most and give both numbers. Say whether your prediction had the direction right, and if it did not, what you were picturing that turned out to be wrong.

3. Paste the `singular system` block from the end of `span.py`: the matrix, the rank, the determinant, and whichever ending you got. Say which of the four printed systems it is most like, the kind with no solution or the kind with infinitely many, given the right-hand side you tried to reach. Then restate it as a span question using your two actual words and counts, and write down a right-hand side, in your own counts, that would have made it the other kind of singular.
