# A05 · Neural Nets Ch. 1 + Embedding Space Probe

**Meetings:** D09 · **Points:** 15 pts

**Watch — 24 min**

**One meeting — 24 min**
[But what is a Neural Network?, 3Blue1Brown Deep Learning Ch. 1](https://www.youtube.com/watch?v=aircAruvnKk) (18m40s)
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) · lesson 3, (Word) Embeddings (5m), rewatch with A03–A04 in your head
 · Paper and a pencil for the first one; `embed/probe.py` from Step 4 open for the second.

**During the video**

**3Blue1Brown Ch. 1 · write, do not type.** No code. Watch it for the picture of what a learned representation *is*; there is nothing to type along with. When he shows what he hoped each hidden neuron would stand for, a loop or an edge, and then shows what a trained network actually does, pause and write both in your log in one sentence each. Question 1 asks you to put your own vectors next to that.

**DLAI lesson 3 · write one line down.** No code. When the lesson says a word's vector is learned from the text around it, write that sentence in your log in your own words. Question 2 asks you to use it.

**Notes**

NumPy only. No scikit-learn, no `sentence-transformers`, no cosine helper from a library. The entire probe is a normalize, a matrix multiply and an argsort, and doing it by hand once is what stops a vector database from ever looking like magic again.

This is the first assignment that calls a model API rather than reading a file, so it is the first one that can cost money. Embed once, save the array, and load it from disk on every run after that. Your key has a hard cap and it does not warn you on the way to it.

**The bug you are going to hit is a broadcast.** Normalising 200 vectors means dividing a `(200, d)` array by a `(200,)` array of lengths, and NumPy will not do what you mean. The dangerous version is not the one that throws: if your embedding dimension happens to equal your vocabulary count, it divides along the wrong axis and hands you a plausible-looking matrix of nonsense. Print shapes rather than trusting the absence of an error.

One zero vector poisons everything. A length of 0 gives you `nan`, and `nan` spreads through every comparison it touches, so guard the denominator before you divide.

`np.argsort` is ascending, so top-k needs `[::-1]` or a negated matrix. And every vector's nearest neighbour is itself. Slice that off or your top-5 is really a top-4, and slice it off **by index**: the self-match is 1.0 to within floating-point error rather than exactly 1.0, so `S[i] == 1.0` finds it for about a quarter of your words and quietly misses the rest.

**Walkthrough — the probe, by hand**

**Step 1. Accept your second repo.**

A05 starts a different repo, so nothing today depends on A04 and you branch fresh. Leave `dev/bpe` open until A04 is finished and approved; merging it is what closes the tokenizer repo for good.

A05 through A11b live in that different repo. Accept it once, here:

**[Foundations](https://classroom50.org/Sierra-Canyon/hse-2026-2027/assignments/foundations/accept)**

```bash
cd ~/version_control
git clone https://github.com/Sierra-Canyon/hse-2026-2027-foundations-<your-username>.git
cd hse-2026-2027-foundations-<your-username>
./setup.sh
```

**Step 2. Branch, and open today's log entry.**

```bash
git switch -c dev/embeddings
git status
```

*You should see* `On branch dev/embeddings`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-unit0, not main
bash scripts/start-entry.sh
```

**`start-entry.sh` leaves you an empty `- [ ]` under the timestamp.** Fill it in before you start anything else. `sign-off.sh` carries whatever you wrote into the sign-off block, so you tick items off rather than retype them, and it drops the checkbox if you left it empty. Today's checklist:

```markdown
- [ ] foundations accepted and cloned into ~/version_control
- [ ] uv add numpy, and nothing else
- [ ] embed/probe.py running against the class endpoint
- [ ] 200 words embedded once, saved to embed/vecs.npy and embed/words.npy
- [ ] Cosine similarity written by hand, shape checked at every step
- [ ] All three probes run and embed/FINDINGS.md written
- [ ] Pushed and the pull request open
```

That branch has been open since A03 and stays open until the track election.

**Step 3. Install NumPy and nothing else.**

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
uv add numpy
mkdir -p embed
```

If you find yourself typing `uv add scikit-learn` this week, stop and re-read the assignment.

**Step 4. Create `embed/probe.py` and run it once before it does anything.**

This assignment is a Python script, not a notebook. A notebook remembers what you ran earlier. A script remembers nothing: every time you run `probe.py`, Python starts at line 1 with an empty memory and runs every line in order to the bottom. The only things that survive from one run to the next are files you saved to disk, which is why Step 5 saves your vectors to a file.

Create the file. In VS Code, open your `hse-2026-2027-foundations-<your-username>` folder, right-click `embed` in the file list, choose **New File**, and name it `probe.py`. Paste this in and save:

```python
# embed/probe.py
# Run from the repo root: uv run python embed/probe.py

import json, os, urllib.request
import numpy as np

# ------------------------------------------------------------ 1. words

# ------------------------------------------------------------ 2. vectors

# ------------------------------------------------------------ 3. compare

# ------------------------------------------------------------ 4. probes

print("probe.py ran to the end")
```

The lines that start with `#` are comments, and Python skips them. They are a map: each of the next three steps tells you which of the four numbered sections to paste into. The last line stays at the very bottom of the file for good. When you see it print, every line above it ran.

Now run it. The terminal has to be in the repo root, the folder that holds `embed/`, not inside `embed/` itself:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
uv run python embed/probe.py
```

*You should see* `probe.py ran to the end`. That one line proves three things at once: the file is where it should be, the terminal is in the right folder, and NumPy is installed where `uv` can find it.

*If it broke:* `can't open file ... embed/probe.py` means the terminal is in the wrong folder. Run `pwd`; it should end in `foundations-<your-username>`. `ModuleNotFoundError: No module named 'numpy'` means you typed `python` instead of `uv run python`. Plain `python` is a different Python that has never heard of the NumPy you installed in Step 3.

From here on the rhythm never changes: paste code into a section, save, run the whole file again from the top, and read what it prints before you move on.

**Step 5. Add your 28 words, get vectors for all 200 once, and save them.**

**The word list goes under `# 1. words`.** It is a Python list: square brackets, each word in quotes, a comma after each one. Most of it is written for you: 20 groups of ten. Groups 1 to 8 are pairs, with four pairs given and the fifth, `"...", "..."`, left for you. Groups 9 to 20 give nine words and leave the tenth, `"..."`, for you.

```python
words = [
    # 1. family and royalty
    "king", "queen", "man", "woman", "prince", "princess", "father", "mother",
    "...", "...",                                   # your pair
    # 2. countries and capitals
    "France", "Paris", "Japan", "Tokyo", "Italy", "Rome", "Egypt", "Cairo",
    "...", "...",                                   # your pair
    # 3. verbs and their past tense
    "walk", "walked", "swim", "swam", "run", "ran", "write", "wrote",
    "...", "...",                                   # your pair
    # 4. comparisons
    "tall", "taller", "small", "smaller", "fast", "faster", "good", "better",
    "...", "...",                                   # your pair
    # 5. synonym pairs
    "happy", "glad", "large", "huge", "begin", "start", "buy", "purchase",
    "...", "...",                                   # your pair
    # 6. more synonym pairs
    "tired", "exhausted", "angry", "furious", "quiet", "silent", "error",
    "mistake",
    "...", "...",                                   # your pair
    # 7. antonym pairs
    "hot", "cold", "up", "down", "open", "closed", "day", "night",
    "...", "...",                                   # your pair
    # 8. more antonym pairs
    "win", "lose", "love", "hate", "early", "late", "full", "empty",
    "...", "...",                                   # your pair
    # 9. animals
    "dog", "cat", "horse", "cow", "eagle", "salmon", "dolphin", "spider",
    "elephant",
    "...",                                          # yours
    # 10. food
    "bread", "cheese", "apple", "rice", "coffee", "soup", "chocolate",
    "pepper", "lemon",
    "...",                                          # yours
    # 11. jobs
    "teacher", "doctor", "lawyer", "farmer", "pilot", "nurse", "chef",
    "soldier", "engineer",
    "...",                                          # yours
    # 12. technology
    "computer", "keyboard", "password", "software", "internet", "battery",
    "robot", "printer", "algorithm",
    "...",                                          # yours
    # 13. places
    "river", "mountain", "desert", "forest", "ocean", "island", "city",
    "village", "hospital",
    "...",                                          # yours
    # 14. feelings
    "fear", "hope", "anger", "joy", "pride", "guilt", "trust", "envy",
    "boredom",
    "...",                                          # yours
    # 15. sports
    "soccer", "tennis", "basketball", "chess", "marathon", "referee", "goal",
    "stadium", "cycling",
    "...",                                          # yours
    # 16. phrases
    "ice cream", "machine learning", "New York", "black hole", "high school",
    "credit card", "social media", "climate change", "point of view",
    "...",                                          # yours
    # 17. famous people
    "Shakespeare", "Einstein", "Mozart", "Napoleon", "Cleopatra", "Beyonce",
    "Picasso", "Gandhi", "Newton",
    "...",                                          # yours
    # 18. names of things
    "Google", "Nike", "Toyota", "Python", "Linux", "Amazon", "Netflix",
    "Harvard", "Everest",
    "...",                                          # yours
    # 19. random dictionary words
    "tureen", "bight", "quoin", "sedge", "welkin", "ferrule", "gambrel",
    "nacre", "kerf",
    "...",                                          # yours
    # 20. more random dictionary words
    "scrim", "plinth", "tarn", "swale", "cusp", "muntin", "byre", "spandrel",
    "thole",
    "...",                                          # yours
]
assert "..." not in words, "replace every ... with a word of your own"
print(len(words), "words,", len(set(words)), "unique")
twice = sorted({w for w in words if words.count(w) > 1})
assert not twice, f"in your list more than once: {twice}. Change the copy you added"
```

**Replace every `"..."` with a word of your own, so each group ends with ten.** In groups 1 to 8, add a whole pair of your own that follows the group's pattern: two family or royal words like `father` and `mother`, a country and its capital, a verb and its past tense, a word and its comparison like `good` and `better`, two synonyms, two antonyms. Question 2 uses the pairs you add in groups 5 to 8. In groups 9 to 18, add one word that fits the group. In groups 19 and 20, add another word picked at random from a dictionary, not one you already know.

*You should see* `200 words, 200 unique`. If the run stops with `AssertionError: replace every ... with a word of your own`, a `"..."` is still in the list. If the second number is under 200, a word you added is already in the list (`set(words)` keeps one copy of each word), and the next line stops the run with `AssertionError: in your list more than once:` followed by those words. Change the copy you added, and if it was half of a pair, pick a new pair. The order of this list is fixed from now on. Row 0 of your vectors belongs to `words[0]`, which is `king`, row 1 to `words[1]`, and so on for all 200.

**The vectors go under `# 2. vectors`.** Two pieces. First, the function that calls the model. You do not need to write it, only paste it:

```python
def embed(texts):
    body = json.dumps({"model": "text-embedding-3-small", "input": texts}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings", data=body,
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        items = json.load(r)["data"]
    items.sort(key=lambda it: it["index"])     # arrival order is not input order
    return np.array([it["embedding"] for it in items])
```

In plain terms: it sends all 200 words to the model in one request, gets back one list of numbers per word, puts them back in the order you sent them, and stacks them into a grid with one row per word. That grid is called `V`. The `items.sort` line matters more than it looks. The answers are not guaranteed to come back in the order you asked, and without that line row 5 could quietly belong to somebody else's word.

Second, the part that decides whether to call the model at all. Paste it directly under the function:

```python
if os.path.exists("embed/vecs.npy"):
    V = np.load("embed/vecs.npy")
    saved = np.load("embed/words.npy").tolist()
    assert saved == words, "your word list changed since vecs.npy was saved: delete embed/vecs.npy and embed/words.npy, then run again"
    print("loaded saved vectors", V.shape)
else:
    V = embed(words)
    np.save("embed/vecs.npy", V)
    np.save("embed/words.npy", np.array(words))
    print("called the model and saved", V.shape)
```

This is the block that protects your spend cap. The first time you run, `embed/vecs.npy` does not exist yet, so the `else` half runs: it calls the model once and saves the answer to disk. Every run after that, the file exists, so the top half loads it and never calls the model again. Step 6 takes several runs to get right, and every one of them is free.

Run it.

*You should see*, the first time, three lines: `200 words, 200 unique`, then `called the model and saved (200, 1536)`, then `probe.py ran to the end`. Every time after that, the middle line reads `loaded saved vectors (200, 1536)` instead. In `(200, 1536)` the first number is your 200 words and the second is how many numbers the model uses to describe each one. Question 1 asks about that second number, so read it off your own output.

*If it broke:* `KeyError: 'OPENAI_API_KEY'` means this terminal has not loaded the key you put in your shell profile in A01. It does not mean the key is wrong. Close the terminal, open a new one, and run again. The key is never typed into this file. `AssertionError: your word list changed` means exactly what it says: you edited `words` after saving, so the saved rows no longer match your list. Delete both `.npy` files and run once more to fetch fresh vectors.

`embed/vecs.npy` is committed on purpose. A06 builds on exactly these vectors, and I want to be able to re-run your numbers.

**Step 6. Compare every word with every other word.**

**What you are about to compute.** Each word is now a row of 1,536 numbers. Think of each row as an arrow pointing somewhere in a space with 1,536 directions. Two words the model thinks are alike point roughly the same way. The number that measures "roughly the same way" is called cosine similarity: `1.0` means two arrows point in exactly the same direction, `0.0` means they sit at right angles and share no direction at all, and `-1.0` means they point opposite ways. It looks only at direction and ignores how long the arrows are.

**Why the code shrinks every arrow first.** The formula for comparing two arrows divides by both of their lengths. If you make every arrow exactly length 1 at the start, those divisions become divisions by 1 and drop out, and comparing two words turns into multiplying their numbers together pairwise and adding up the results. One line of NumPy, `Vn @ Vn.T`, then does that for every pair of words at once.

This goes under `# 3. compare`:

```python
norms = np.linalg.norm(V, axis=1)      # (200,)  one length per word
norms = np.maximum(norms, 1e-10)       # a length of 0 would mean dividing by 0
Vn = V / norms[:, None]                # (200, 1536)  every row now has length 1
S = Vn @ Vn.T                          # (200, 200)  every word against every word
print(V.shape, norms.shape, Vn.shape, S.shape)
print(np.allclose(np.diag(S), 1.0))
print(words[0], "vs", words[1], round(float(S[0, 1]), 3))
```

For anyone new to NumPy: `axis=1` means work along each row, so you get one length per word rather than one for the whole grid. `norms[:, None]` stands those 200 lengths up into a column, so that NumPy divides each row by its own length. The `@` compares every row with every other row. `S[i, j]` is how similar word `i` is to word `j`, and there are 40,000 of those numbers.

*You should see* four shapes, `(200, 1536) (200,) (200, 1536) (200, 200)`, then `True`, then `king vs queen` and a number between -1 and 1: the first two words of the list, compared.

The `True` is the gate. Every word is perfectly similar to itself, so everything along the diagonal of `S` has to be `1.0`. If it prints `False`, something above it is wrong and nothing below it means anything, so stop and fix it before Step 7.

*If it broke:* a `ValueError` about shapes on the `Vn = ` line means you left off `[:, None]`. If it printed `False` rather than raising, look for a word whose vector is all zeros before you suspect anything else: `print(np.where(np.linalg.norm(V, axis=1) == 0))`. The `np.maximum` line stops a zero vector from turning everything into `nan`, but it cannot make that word similar to itself. If you have one, name the word in `FINDINGS.md`.

**Step 7. Run the three probes.**

All three are required. They all go under `# 4. probes`, and all three need this line first, so paste it at the top of that section:

```python
idx = {w: i for i, w in enumerate(words)}   # word -> its row number
```

`idx` is a dictionary: give it a word and it gives back that word's row number. `idx["king"]` is where `king` sits in `words`, in `V` and in `S`.

**Probe 1. Neighbors.** Pick any ten single words from your 200, at least three of them words you added. They do not need to be pairs, and they can come from any group. For each one, the code prints the five words the model thinks are most similar.

```python
def neighbors(word, k=5):
    i = idx[word]
    row = S[i].copy()                  # copy, so the next line does not change S itself
    row[i] = -np.inf                   # take the word itself out of the running
    top = np.argsort(row)[::-1][:k]
    return [(words[j], round(float(row[j]), 3)) for j in top]

MY_TEN = ["...", "...", "...", "...", "...",     # ten single words from your 200,
          "...", "...", "...", "...", "..."]     # at least three of them yours
assert len(MY_TEN) == 10, f"MY_TEN has {len(MY_TEN)} words, it needs 10"

missing = [w for w in MY_TEN if w not in idx]
assert not missing, f"not in your 200-word list: {missing}"
for w in MY_TEN:
    print(f"{w:<18}", neighbors(w))
```

How `neighbors` works: row `i` of `S` holds word `i` against all 200 words. The word itself is taken out first, because every word is its own nearest neighbor. `np.argsort` lists the positions from least similar to most, `[::-1]` flips that so the most similar comes first, and `[:k]` keeps the top five. `f"{w:<18}"` only pads the word to 18 characters so your ten lines line up.

*You should see* ten lines, each a word followed by five `(neighbor, score)` pairs, highest first. Synonym pairs should turn up near the top of each other's lists, and the random words in groups 19 and 20 near nothing in particular. If every list looks sensible, you picked ten easy words; `FINDINGS.md` asks for one row you disagree with.

*If it broke:* `AssertionError: not in your 200-word list: ['...', ...]` means placeholders are still in `MY_TEN`; any other word in that list is misspelled or not in your 200. `AssertionError: MY_TEN has 9 words, it needs 10` means a word is missing, or a comma between two words is. A neighbor scoring exactly `1.0` means that word is in your list twice.

**Probe 2. Arithmetic.** An analogy says *a is to b as d is to c*: king is to man as queen is to woman. The code turns that into arithmetic, `a - b + c`, and shows the words closest to where it lands. You write `a`, `b` and `c`; `d` is the word you are hoping for, and it has to be in your 200 or it cannot show up. `king - man + woman`, hoping for `queen`, is required. Build two more of your own from groups 1 to 4, where every word has a partner:

- `("Paris", "France", "Japan")`: Paris is to France as ? is to Japan, hoping for `Tokyo`
- `("walked", "walk", "swim")`: hoping for `swam`
- `("taller", "tall", "small")`: hoping for `smaller`

Those three are examples; use your own. At least one of your two should use a pair you added. If you added `Germany` and `Berlin`, `("Berlin", "Germany", "Italy")` hopes for `Rome`, and `("Paris", "France", "Germany")` hopes for `Berlin`.

```python
def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-10)

def analogy(a, b, c, k=5):
    q = unit(V[idx[a]] - V[idx[b]] + V[idx[c]])   # the arithmetic, on the raw vectors
    scores = Vn @ q                               # every word against the result
    top = np.argsort(scores)[::-1][:k]
    return [(words[j], round(float(scores[j]), 3), j in (idx[a], idx[b], idx[c]))
            for j in top]

TRIOS = [("king", "man", "woman"),     # required: hoping for queen
         ("...", "...", "..."),        # yours: a, b, c
         ("...", "...", "...")]        # yours: a, b, c

missing = [w for t in TRIOS for w in t if w not in idx]
assert not missing, f"not in your 200-word list: {missing}"
for trio in TRIOS:
    print(trio, analogy(*trio))
```

This takes the arrow for `king`, subtracts the arrow for `man`, adds the arrow for `woman`, and asks which of your 200 words points closest to where that lands. `unit()` shrinks the result to length 1 so its scores sit on the same scale as everything in `S`.

*You should see* three lines, one per analogy, each holding five entries of `(word, score, True or False)`. `True` marks one of the three words you put in. Leave those in and do not filter them out. The model's answer is the highest entry marked `False`, and the analogy worked if that is the word you hoped for. `king` and `woman` often sit above it, and that is not a bug: the result was built by adding and subtracting those exact arrows, so it lands near them. Expect one of your own two analogies to fail; question 3 asks about it.

**Probe 3. The baseline: how similar are two words that have nothing to do with each other?** The average score over 1,000 random pairs of your words, skipping any pair where a word lands on itself.

```python
rng = np.random.default_rng(0)         # fixed seed, so you get the same pairs every run
n = len(words)
i = rng.integers(0, n, 1000)           # 1,000 random row numbers
j = rng.integers(0, n, 1000)           # and 1,000 more
keep = i != j                          # drop any pair that is a word with itself
pairs = S[i[keep], j[keep]]            # look up all of those similarities at once
print(f"{len(pairs)} pairs, mean {pairs.mean():.4f}, "
      f"min {pairs.min():.4f}, max {pairs.max():.4f}")
```

*You should see* one line reporting a little under 1,000 pairs, because a few of the random pairs will be a word matched with itself, then a mean, a min and a max.

Two unrelated words ought to score about 0.0, at right angles. Yours will not. The mean comes out clearly above zero, because every vector this model produces leans a little in one shared direction and every pair inherits that lean. The technical name for that is anisotropy. That mean is the floor every score in this file sits on: a 0.4 against a floor of 0.1 is a strong match, and a 0.4 against a floor of 0.35 barely is one. A06 depends on it.

Run the whole file one last time.

*You should see*, top to bottom: the word count, the load line, the four shapes, `True` and your first pair, ten neighbor lines, three analogy lines, the baseline line, and `probe.py ran to the end`.

**Extension — the part of the probe that is yours (ASSIGNED)**

The 28 words you added, the ten in `MY_TEN`, your two analogies and the row you disagree with are the extension: they are the only parts of this file no one else in the room has, and `FINDINGS.md` is where they go.

**X1. Write `embed/FINDINGS.md`.**

Create it the same way as `probe.py`: right-click `embed`, **New File**, `FINDINGS.md`. Copy this in and fill every angle bracket. Paste output exactly as it printed: do not retype it, round it, or tidy it up.

```markdown
# A05 findings: <your name>

Model: text-embedding-3-small - Words: 200

## My 28 words

    <the 28 words you added, in group order, separated by commas>

My two analogies, each with the word I hoped for:
<a - b + c, hoping for d>
<a - b + c, hoping for d>

## Probe 1: neighbors

    <paste the ten lines exactly as printed, indented four spaces>

The row I disagree with, and why I think the model put those words together:
<one or two sentences>

## Probe 2: arithmetic

    <paste all three lines exactly as printed>

## Probe 3: the baseline for unrelated words

    <paste the line exactly as printed>

What that mean tells me about reading any single score from this model:
<one or two sentences>
```

The reflection questions are not in this file. They go in your log, and nothing they ask needs repeating here.

**X2. Commit, push, open the pull request.**

```bash
git add embed/probe.py embed/vecs.npy embed/words.npy embed/FINDINGS.md
git commit -m "A05: embedding space probe"
git push -u origin dev/embeddings
```

**Open the pull request** for `dev/embeddings`: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop.

**Merge when the assignment is finished and I have approved it, not before.** That will usually land a day or two into the next assignment, because the next one opens before this one is due. Branches are independent, so having two open at once is normal and is not a sign you are behind.

**X3. Close the log.**

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`embed/probe.py`, `embed/vecs.npy`, `embed/words.npy` and `embed/FINDINGS.md` in the shape X1 gives, plus the three reflection questions answered in your log.

**Reflection Questions**

1. Add `print(V[idx["king"]][:8])` just above the last line of `probe.py` and run it. Those are 8 of the numbers your model uses for `king`; how many does it use in total? Pick any one of the eight and say what it means. 3Blue1Brown hoped each hidden neuron would stand for a clean piece of a digit, like a loop or an edge, and then showed that a trained network does not work that way. Say how that matches what you are looking at.
2. Take the antonym pair you added in group 7 or 8 and print how similar your model thinks they are, the same way as above: `print(S[idx["hot"], idx["cold"]])`, with your own two words. Do the same for the synonym pair you added in group 5 or 6, and compare both numbers with your Probe 3 baseline. Is your antonym pair closer to the synonyms or to the unrelated words? The DLAI lesson says a word's vector is learned from the text around it. Use that to explain your numbers.
3. In your `king - man + woman` line, where did `queen` land, and what sat above it? Then take whichever of your own two analogies worked worse. What does its top 5 suggest the model learned about those words instead of the relationship you meant?
