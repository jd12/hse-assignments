# B23 · Probability, Surprise, and Information

**Meetings:** D44–D45 · **Points:** 15 pts


**Watch — 66 min**
Day 1 — 32 min
[3Blue1Brown, Reinventing Entropy | Compression is Intelligence, Part 1](https://www.youtube.com/watch?v=l6DKRf-fAAM) · whole video (32:20, 32m)

Day 2 — 34 min
[3Blue1Brown, But what is cross-entropy? | Compression is Intelligence, Part 2](https://www.youtube.com/watch?v=GlYgs6v2YfU) · whole video (33:51, 34m)

 · **Both days are over the 30-minute cap, by 2 and 4 minutes.** Each one finishes a video, and neither splits cleanly in the middle of its argument, so you watch each one whole on its day. The code-along that would have gone with Part 2 is in the walkthrough as optional.
 · Paper and a pencil for both. There is no code in either video.

**During the video**

Both parts are conceptual. The walkthrough is where you type. What you do while watching is keep one running table on paper, and photograph it at the end of Day 2 as `info/video-notes.jpg`.

**Part 1 · keep the table.** Rule three columns: *outcome*, *probability*, *bits*. Every time he attaches a probability to an outcome, or a number of bits to a message, add a row, and fill in the column he did not give you: if he gives `p`, compute `-log2 p`; if he gives bits, compute `2^(-bits)`. When he arrives at the average over all outcomes, write his formula at the bottom of the table, and underneath it compute it yourself for a fair coin and a fair six-sided die. You should get `1` and about `2.585`. Tomorrow's first step checks the die in code.

**Part 2 · two numbers every time.** Whenever he measures a code or a model against data it was not built for, write two numbers side by side: the average bits using the *right* distribution, and the average bits using the *wrong* one. Write the difference under them. Then, before he says it, write down which of the two is always at least as large, and whether it can ever be smaller. Step 4 measures that difference with your own die.

**Notes**

You have had no probability. Like B13–B16, none of this is review.

Three ideas carry into everything after this. A **distribution** assigns a probability to each possible outcome, every probability is between 0 and 1, and they sum to 1. A **conditional probability** `P(A | B)` is "the probability of A given that B happened". Two events are **independent** when knowing B tells you nothing about A. The extension measures the second and third on your corpus: how much knowing the previous character tells you about the next, and what happens to that when the characters are shuffled.

An LLM's output layer is a probability distribution over its vocabulary, and its training loss is the average surprisal of the right answer. B24 derives that loss on paper in January.

**Surprisal is `-log2 p`**, in bits. Rare outcomes cost many bits, certain ones cost zero. A probability of exactly 0 has infinite surprisal: `math.log2(0)` raises `ValueError: math domain error`, and every piece of code below guards for it, because a model that calls something impossible and then sees it has no finite cost.

**Bits and nats.** `math.log2` gives bits; `math.log` gives nats, which PyTorch and makemore use. A number off by a factor of about 1.44 is a units bug.

**Floating-point sums.** `sum([1/6] * 6)` is `0.9999999999999999`, not `1.0`. The checks below use `abs(total - 1) < 1e-9`. `== 1` fails on a perfectly good distribution.

**Walkthrough — Surprisal, entropy and cross-entropy, from dice to your corpus**

**Day 1 starts here.**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:
```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/entropy
```
*If B22 is not approved yet:* branch from `main` anyway. Nothing today uses the engine.
```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-track, not main
bash scripts/start-entry.sh
```
Under the timestamp, write the checklist for the whole assignment, and tick what you finish each day:
```markdown
- [ ] Day 1: Reinventing Entropy (Part 1), keeping the outcome / probability / bits table
- [ ] Day 1: Steps 2–3, the fair die and my loaded die, and my corpus's characters
- [ ] Day 1: extension predictions committed before any corpus entropy runs
- [ ] Day 2: But what is cross-entropy? (Part 2), right and wrong averages side by side
- [ ] Day 2: Step 4, cross-entropy with my dice; Step 5 optional makemore
- [ ] Day 2: Extension, corpus vs shuffled, real line vs scrambled line, held-out
- [ ] Day 2: bring the dice tables to the discussion
- [ ] Push each day; one PR
```

Watch Part 1 before Step 2.

**Step 2. Two dice.**

Create `info/dice.py`. Invent your loaded die on the `LOADED` line: six probabilities, each between 0 and 1, summing to 1. Make it a die you can describe in a sentence ("rolls a six half the time"), and do not copy the example:

```python
import math

FAIR = [1 / 6] * 6
LOADED = [0.10, 0.10, 0.10, 0.10, 0.10, 0.50]     # CHANGE THIS to your own die

def bits(p):
    return -math.log2(p) if p > 0 else float("inf")

def entropy(probs):
    return sum(p * bits(p) for p in probs if p > 0)

def table(name, probs):
    total = sum(probs)
    assert abs(total - 1) < 1e-9, f"{name} sums to {total}, not 1"
    print(f"{name}\nface   probability   surprisal (bits)")
    for face, p in enumerate(probs, 1):
        print(f"{face:4d}   {p:11.4f}   {bits(p):16.4f}")
    print(f"entropy (average surprisal): {entropy(probs):.4f} bits\n")

if __name__ == "__main__":
    table("fair die", FAIR)
    table("my loaded die", LOADED)
```

Run it: `uv run python info/dice.py`, and paste the output into `info/DICE.md`. That file is the two probability tables the discussion needs: numbers, not prose.

*You should see* every face of the fair die cost `2.5850` bits and its entropy be `2.5850`, the number you computed on paper during Part 1. For your loaded die, the likely faces cost fewer bits than 2.585 and the unlikely ones more, and its entropy is **lower** than the fair die's. That holds for every loaded die anyone in the room invents. A fair die is the most uncertain six-sided die there is. If you gave any face probability 0, its surprisal prints as `inf`, and the entropy leaves it out, because an outcome that never happens contributes nothing to the average.

*If it broke:* `AssertionError: my loaded die sums to 0.9999...` usually means you typed thirds or sixths as decimals. Write them as fractions, `1 / 3`, not `0.33`.

**Step 3. Your corpus as a die with many faces.**

Your corpus is a die too: each character is a roll, and the faces are every distinct character in the file. Create `info/chars.py`:

```python
import math
from collections import Counter

text = open("data/corpus.txt", encoding="utf-8").read()
text = " ".join(text.split())                    # every run of whitespace becomes one space
counts = Counter(text)
N, V = len(text), len(counts)
probs = {ch: c / N for ch, c in counts.items()}

print(f"{N:,} characters, {V} distinct")
print(f"a uniform code for {V} faces costs log2({V}) = {math.log2(V):.4f} bits per character")
print("\nmost common           least common")
common, rare = counts.most_common(8), counts.most_common()[-8:]
for (c1, n1), (c2, n2) in zip(common, rare):
    print(f"{c1!r:5} p={probs[c1]:.4f} {-math.log2(probs[c1]):6.2f} bits    "
          f"{c2!r:6} p={probs[c2]:.2e} {-math.log2(probs[c2]):6.2f} bits")
```

Run it: `uv run python info/chars.py`.

*You should see* between about 50 and a few hundred distinct characters, depending on how much punctuation, capitals, digits and accented or non-Latin characters your corpus uses, and a uniform-code cost of about 6 to 8 bits. The most common character is almost certainly the space, at under 3 bits; the least common ones are a handful of occurrences each, costing 15 bits or more. Those rare characters are the ones your A03–A04 tokenizer spent the most bytes on, for the same reason.

Now, before any corpus entropy is computed, write the extension's three predictions in your log and commit it: your corpus's character entropy in bits; whether a copy of your corpus with every character shuffled will have higher, lower or the same entropy; and how many bits per character a real line will cost under a model that knows only which character tends to follow which. I will check your commit timestamps. Commit and push today's work; open the pull request with **jd12** as reviewer; sign off the log.

**Day 2 starts here.** Open a new entry with `bash scripts/start-entry.sh` in the log repo and copy over what is left of the checklist. Watch Part 2 before Step 4.

**Step 4. Cross-entropy, with your die.**

Part 2's idea in code: roll your loaded die 10,000 times, then ask how many bits per roll each of three models would spend describing those rolls. Create `info/cross.py`:

```python
import random
from dice import FAIR, LOADED, bits, entropy

WRONG = LOADED[::-1]                            # your die, read backwards: a confident wrong model
rng = random.Random(0)
rolls = rng.choices(range(6), weights=LOADED, k=10_000)

def avg_bits(model):
    return sum(bits(model[r]) for r in rolls) / len(rolls)

print(f"entropy of my loaded die          {entropy(LOADED):.4f} bits")
print(f"rolls coded with my loaded die    {avg_bits(LOADED):.4f} bits per roll")
print(f"rolls coded with the fair die     {avg_bits(FAIR):.4f} bits per roll")
print(f"rolls coded with the reversed die {avg_bits(WRONG):.4f} bits per roll")
```

Run it: `uv run python info/cross.py`.

*You should see* the right model's average within about 0.02 of your die's entropy, since 10,000 rolls is a sample and not the whole distribution. The fair-die line is exactly `2.5850` whatever you rolled, because every face costs the same under it. The reversed die's line is the highest of the three, and can be `inf` if your die has a face with probability 0. If your die reads the same backwards, the reversed die *is* your die and the line matches the first one; set `WRONG` to some other die of your own and say which in `FINDINGS.md`. The number that measures a model is its cross-entropy on data it did not choose; it is never below the data's own entropy, on average, and the gap is how wrong the model is. That is the inequality you wrote down during Part 2. A language model's loss is this third line, computed on text.

**Step 5. Optional: see a loss like this typed.**

If you want to watch someone compute the same number for a character model in code, [Karpathy, makemore Part 1](https://www.youtube.com/watch?v=PaCmpygFfXo&t=3014s), segment 00:50:14 → 01:02:57, "loss function (NLL)" and "smoothing" (13m), not counted in the totals. He uses PyTorch, a list of names and natural log. If you watch it, type his loss lines into `scratch/b23-video.py`; his average negative log likelihood is the extension's `bits_per_char`, in nats.

**Extension — How much of your corpus is order?** *(assigned)*

Your committed predictions from Step 3 are the baseline.

**Part A: your corpus against a shuffled copy.** Create `info/entropy.py`:

```python
import math
import random
from collections import Counter

text = open("data/corpus.txt", encoding="utf-8").read()
text = " ".join(text.split())

def entropy(counts):
    total = sum(counts.values())
    return -sum(c / total * math.log2(c / total) for c in counts.values())

def conditional_entropy(s):
    """H(next | previous): average surprise at a character once you know the one before it."""
    pairs = Counter(s[i:i + 2] for i in range(len(s) - 1))
    firsts = Counter(s[:-1])
    total = sum(pairs.values())
    return -sum(c / total * math.log2(c / firsts[p[0]]) for p, c in pairs.items())

chars = list(text)
random.Random(0).shuffle(chars)
shuffled = "".join(chars)

print(f"characters: {len(text):,}   distinct: {len(set(text))}")
print(f"H(char)            corpus {entropy(Counter(text)):.4f}   shuffled {entropy(Counter(shuffled)):.4f}")
print(f"H(next | previous) corpus {conditional_entropy(text):.4f}   shuffled {conditional_entropy(shuffled):.4f}")
```

*You should see* two things. First, `H(char)` is **identical** to four decimals for your corpus and its shuffled copy, somewhere between about 4 and 5 bits for English prose. Shuffling moves characters around; it does not change how many of each there are, and single-character entropy only counts. This is the prediction most people get wrong. Second, `H(next | previous)` is clearly lower than `H(char)` for your real corpus, often by a bit or more, and for the shuffled copy it is almost exactly `H(char)` again. Knowing the previous character helps, and it helps only because your corpus has order. Shuffle it and the previous character tells you nothing: the characters have become independent.

**Part B: a real line against a scrambled line.** Create `info/surprisal.py`. It builds a bigram count model from the first 90% of your corpus and uses it to price text:

```python
import math
import random
from collections import Counter

text = open("data/corpus.txt", encoding="utf-8").read()
text = " ".join(text.split())
cut = int(len(text) * 0.9)
train, held = text[:cut], text[cut:]              # the model never sees the last 10%

V = len(set(text))
pairs = Counter(train[i:i + 2] for i in range(len(train) - 1))
firsts = Counter(train[:-1])

def p_next(prev, ch):                             # add-one smoothing: nothing is impossible
    return (pairs[prev + ch] + 1) / (firsts[prev] + V)

def bits_per_char(s):
    total = sum(-math.log2(p_next(s[i - 1], s[i])) for i in range(1, len(s)))
    return total / (len(s) - 1)

line = held[1000:1080]                            # 80 characters the model never saw
chars = list(line)
random.Random(1).shuffle(chars)
scrambled = "".join(chars)

print(f"uniform code: {math.log2(V):.3f} bits/char   H(char): see entropy.py")
print(repr(line));      print(f"   {bits_per_char(line):.3f} bits/char")
print(repr(scrambled)); print(f"   {bits_per_char(scrambled):.3f} bits/char")
print(f"held-out 10%: {bits_per_char(held):.3f} bits/char   same length of training text: {bits_per_char(train[-len(held):]):.3f}")
```

If `held[1000:1080]` lands on a table of contents or a license footer, move the slice until it is a line of real prose, and say in `FINDINGS.md` where you put it.

*You should see* the real line cost somewhere around 3 to 4 bits per character, below your `H(char)`, because the model knows which characters follow which. The scrambled line, the same 80 characters, costs far more: usually more than your *uniform-code* cost from Step 3. A model with no knowledge at all would have paid less for it. A model that has learned what usually comes next is confidently wrong about text that does not follow the pattern, and confident wrongness costs more bits than ignorance. The held-out 10% usually costs about the same as, or a little more than, the training text next to it; the difference is a few hundredths of a bit at most, because a bigram table built from a million characters has almost nothing to memorize. If your held-out number is *lower*, say what kind of text sits in the slice you compared it with.

Write `info/FINDINGS.md`: both outputs; your three committed predictions next to the measured numbers; and where it got worse, with numbers: the scrambled line against the uniform code, and held-out against training. Then one sentence on what "compression is intelligence" means for your corpus, using the gap between your `H(char)` and your `H(next | previous)`.

Commit, push, sign off:

```bash
git add info/ scratch/
git commit -m "B23: dice tables, cross-entropy with dice, corpus entropy vs shuffled, bigram surprisal"
git push
```

The pull request you opened on Day 1 picks this up. Say in the PR body where this is weakest.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

Push again each day; one PR, not two; one log entry per meeting.

**Deliverable**
`info/DICE.md` (two probability tables, a fair die and your own loaded die, with the probability and the surprisal in bits of every outcome) + `info/dice.py` + `info/chars.py` + `info/cross.py` + `info/entropy.py` + `info/surprisal.py` + `info/FINDINGS.md` + `info/video-notes.jpg`.

**Discussion · Wed Dec 9, in class**
Which of your two dice is "more surprising to observe," in the sense Part 1 uses the word, and why. Bring your tables. Ungraded, nothing to hand in.

**Reflection Questions**

1. Paste your loaded die's table from `DICE.md` and the four lines from `cross.py`. Name the face that costs the most bits and the face that costs the fewest, and say what each number means as a length of message. Then explain why the fair-die line is exactly `2.5850` for your rolls when your rolls came from a die that is not fair.

2. Paste the prediction lines you committed on Day 1, with the commit time, and the two lines from `entropy.py`. Say which prediction was furthest off. If you predicted that shuffling would change `H(char)`, say what you were picturing; if you did not, say which of the two measurements *did* change when you shuffled, by how much, and what that number is measuring about your corpus.

3. Paste the real and scrambled lines from `surprisal.py` with their bits per character, and your uniform-code cost from `chars.py`. Where did it get worse than knowing nothing? Pick one character pair in the scrambled line, compute `p_next` for it by hand from the formula (print `pairs[prev + ch]` and `firsts[prev]` to get the counts), and show why that pair alone cost so many bits.
