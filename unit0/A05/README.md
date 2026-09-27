# A05 · Neural Nets Ch. 1 + Embedding Space Probe

**Meetings:** D09 · **Points:** 15 pts

**Video/Source Link(s):**  
[But what is a Neural Network?, 3Blue1Brown Deep Learning Ch. 1](https://www.youtube.com/watch?v=aircAruvnKk) (18m40s)  
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/): lesson 3 ((Word) Embeddings), rewatch with A03–A04 in your head

**Notes**  
NumPy only. No scikit-learn, no `sentence-transformers`, no cosine helper from a library. The entire probe is a normalize, a matrix multiply and an argsort, and doing it by hand once is what stops a vector database from ever looking like magic again.

This is the first assignment that calls a model API rather than reading a file, so it is the first one that can cost money. Embed once, save the array, and load it from disk on every run after that. Your key has a hard cap and it does not warn you on the way to it.

**The bug you are going to hit is a broadcast.** Normalising 200 vectors means dividing a `(200, d)` array by a `(200,)` array of lengths, and NumPy will not do what you mean. The dangerous version is not the one that throws: if your embedding dimension happens to equal your vocabulary count, it divides along the wrong axis and hands you a plausible-looking matrix of nonsense. Print shapes rather than trusting the absence of an error.

One zero vector poisons everything. A length of 0 gives you `nan`, and `nan` spreads through every comparison it touches, so guard the denominator before you divide.

`np.argsort` is ascending, so top-k needs `[::-1]` or a negated matrix. And every vector's nearest neighbour is itself. Slice that off or your top-5 is really a top-4, and slice it off **by index**: the self-match is 1.0 to within floating-point error rather than exactly 1.0, so `S[i] == 1.0` finds it for about a quarter of your words and quietly misses the rest.

3Blue1Brown has no code in it. Watch it for the picture of what a learned representation *is*; there is nothing to type along with.

**Do**

**Step 1. Accept your second repo.**

A05 starts a different repo, so nothing today depends on A04 and you branch fresh. Leave `dev/bpe` open until A04 is finished and approved; merging it is what closes the tokenizer repo for good.

A05 through A11b live in that different repo. Accept it once, here:

**[Foundations](https://classroom50.org/Sierra-Canyon/hse-2026-2027/assignments/foundations/accept)**

```
cd ~/version_control
git clone https://github.com/Sierra-Canyon/hse-2026-2027-foundations-<your-username>.git
cd hse-2026-2027-foundations-<your-username>
./setup.sh
```


**Step 2. Branch, and open today's log entry.**

```
git switch -c dev/embeddings
git status
```

*You should see* `On branch dev/embeddings`.

```
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-unit0, not main
bash scripts/start-entry.sh
```

That branch has been open since A03 and stays open until the track election.

**Step 3. Install NumPy and nothing else.**

```
cd ~/version_control/hse-2026-2027-foundations-<your-username>
uv add numpy
mkdir -p embed
```

If you find yourself typing `uv add scikit-learn` this week, stop and re-read the assignment.

**Step 4. Write your prediction down before you measure anything.**

3Blue1Brown spends eighteen minutes arguing that a trained network learns a representation in which related things end up near each other. Take that claim seriously for a second and it makes a prediction you can test: if the space is organized by meaning, two words picked at random out of a hat are unrelated, so they should be about as far apart as two directions can be.

In today's log entry, before you run any code, write one line: what you expect the mean cosine similarity of 1,000 random pairs drawn from your 200 words to be. Guess a number. A wrong guess written down is worth more than a right guess kept in your head, and this one is worth writing down because almost everybody guesses low. Step 8 is where you find out what the space actually does with unrelated words, and whether the picture in the video survives contact with it.

**Step 5. Write `embed/probe.py` and run it.**

Everything today goes in that one file. Unlike A03 and A04 there is no notebook, which means there is no kernel to select and no cell ordering to get wrong.

You are writing it from an empty file, so here is the shape it ends up in. Steps 6 to 8 fill it in from the top down:

```
# embed/probe.py

# 1. words = [...]          your 200, one list, and the order is fixed from here on
# 2. load embed/vecs.npy if it exists, otherwise call the API once and save it
# 3. normalize -> S = Vn @ Vn.T -> print the shapes, then the diagonal gate
# 4. probe 1   neighbours
#    probe 2   arithmetic
#    probe 3   anisotropy
# 5. print everything you are going to paste into FINDINGS.md
```

Point 2 is a branch rather than a line: if `embed/vecs.npy` is there, load it and do not call the API at all. Write the branch before the API call, not after. Step 7 takes several runs to get right and each one is free with the branch in place.

Run it from the repo root:

```
uv run python embed/probe.py
```

**`uv run` and not plain `python`.** `uv run` uses the `.venv` this repo built, which is where `uv add numpy` put NumPy. Plain `python` uses whatever is on your PATH and will not find it.

*If it broke* with `ModuleNotFoundError: No module named 'numpy'`, that is exactly this: the package installed fine and you ran the wrong interpreter.

**Step 6. Choose your 200 words, embed them once, and save the array.**

Include, deliberately: clear synonym pairs, antonym pairs, a few multi-word phrases, some proper nouns, and 20 words pulled at random from a dictionary. The random 20 are the control, and Step 8 does not work without them.

Call the embeddings endpoint directly, in one batched request rather than 200 separate ones. `urllib` is in the standard library, so there is nothing to install and nothing to import that the assignment bans:

```
import json, os, urllib.request
import numpy as np

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

`os.environ["OPENAI_API_KEY"]` is the key you put in your shell profile in A01. It is never written in this file, and a `KeyError` here means your shell profile is not loaded in this terminal, not that the key is wrong.

That `items.sort` line matters more than it looks and A06 has a whole step about it. The endpoint returns a list, each item carrying the index of the input it came from, and that order is not guaranteed to be the order you sent. Sort by it every time.

Then save immediately:

```
V = embed(words)
# V is your (200, d) array of embeddings, in the same order as your word list
np.save("embed/vecs.npy", V)
np.save("embed/words.npy", np.array(words))
print(V.shape)
```

*You should see* a shape like `(200, 1536)`. The second number is `d`, the embedding dimension, and reflection question 1 asks you for it, so read it off rather than looking it up.

Every run after this one loads the file instead of calling the API:

```
V = np.load("embed/vecs.npy")
```

`embed/vecs.npy` is committed on purpose, because A06 builds on exactly these vectors and I want to be able to re-run your numbers. Re-embedding on every run burns your spend cap, and the cap does not warn you before it stops you.

**Step 7. Cosine similarity by hand, checking the shape at every step.**

```
norms = np.linalg.norm(V, axis=1)      # (200,)  NOT (200, 1)
norms = np.maximum(norms, 1e-10)       # one zero vector otherwise poisons everything with nan
Vn = V / norms[:, None]                # (200, d) — without [:, None] this is the wrong answer
S = Vn @ Vn.T                          # (200, 200)
print(V.shape, norms.shape, Vn.shape, S.shape)
print(np.allclose(np.diag(S), 1.0))
```

*You should see* four shapes, with `S` square at your vocabulary size on both sides, and then `True`.

DLAI lesson 3 draws embeddings as a lookup table, one row per word. `Vn @ Vn.T` compares every row of that table against every other row in one operation. `S[i][j]` is how similar word `i` is to word `j`, and there are 40,000 of those numbers.

That `True` is the gate. Every vector is perfectly similar to itself, so if the diagonal is not 1.0 something above this line is wrong and nothing below it means anything. It is also the only check that catches the silent broadcast: divide along the wrong axis and the diagonal comes back spread between roughly 0.89 and 1.13.

*If it broke* on the division with a `ValueError` about shapes, you left off `[:, None]`. If your vocabulary count happens to equal your embedding dimension, that same mistake raises nothing and divides along the wrong axis instead.

*If it printed* `False` rather than raising, check for a zero-length vector before you suspect your normalization:

```
print(np.where(np.linalg.norm(V, axis=1) == 0))
```

The `np.maximum` guard on the line above stops a zero vector from spreading `nan`, but it cannot make that row similar to itself: its whole row of `S` is 0.0, including the diagonal. If that is what you have, gate on the rows that survive instead, and say in `FINDINGS.md` which word embedded to nothing and why you think it did:

```
print(np.allclose(np.diag(S)[norms > 1e-9], 1.0))
```

**Step 8. Run all three probes.**

All three are required and none of them is optional or free choice.

**Neighbours.** Top-5 nearest for 10 words you choose, with the word itself dropped by index.

*You should see* your synonym pairs near the top of each other's lists and your random 20 near nothing in particular. If every list looks plausible, you picked ten easy words. Reflection question 11 asks for a row you disagree with.

**Arithmetic.** `king - man + woman`, plus **two analogies of your own design**. Report the top 5 for each, not just the top 1.

**Leave the three input words in your top 5 and mark them.** `king` and `woman` will usually take the top two slots, and that is not your bug: the query vector is built by adding and subtracting those exact vectors, so it lands near them. Reflection question 5 is asking where the expected answer turned up *underneath* that, which is a question everybody can only answer the same way if everybody handles the inputs the same way. One of your two is expected to fail; reflection question 6 asks which.

**Anisotropy.** Mean cosine similarity over 1,000 random pairs drawn from your 200, with `i == j` thrown out, because a word compared to itself is 1.0 and you are not asking about that.

*You should see* a number that is clearly positive and nowhere near zero. Do not go looking for the "right" value: it differs by model and it is not the point. The point is the gap between it and the line you wrote in Step 4. Whatever you get is the **floor** every similarity score you produce sits on, which is why it carries straight into A06: a 0.4 against a floor of 0.1 and a 0.4 against a floor of 0.35 are not the same result, and reflection question 3 asks you to turn that into a threshold.

Then go back to the line you wrote in Step 4. Say how far off you were, and why you think you were off in that direction. Step 4 framed it as a prediction the 3Blue1Brown video makes; say whether the video survived it.

**Step 9. Write the findings.**

`embed/FINDINGS.md`: all three probe results, and your Step 4 prediction printed next to the actual anisotropy number so the gap is visible without anyone having to hunt for it. A claim is a sentence someone could disagree with; "embeddings are interesting" is not one.

**Step 10. Commit, push, open the pull request.**

```
git add embed/probe.py embed/vecs.npy embed/words.npy embed/FINDINGS.md
git commit -m "A05: embedding space probe"
git push -u origin dev/embeddings
```

**Open the pull request** for `dev/embeddings`: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop.

**Merge when the assignment is finished and I have approved it, not before.** That will usually land a day or two into the next assignment, because the next one opens before this one is due. Branches are independent, so having two open at once is normal and is not a sign you are behind.

**Step 11. Close the log.**

```
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**  
`embed/probe.py` + `embed/vecs.npy` + `embed/words.npy` + `embed/FINDINGS.md` with all three probe results and your pre-registered anisotropy prediction next to the actual number.

**Reflection Questions**

1. What is the dimensionality `d` of your embeddings? What does one individual dimension mean?
2. What did you predict for mean random-pair cosine similarity, and what was it actually? If your prediction was wrong, what assumption caused the error?
3. Given that number, what is the *practically* meaningful similarity threshold for your corpus. And why is 0.0 not the floor?
4. Report an antonym pair's cosine similarity. Is it high or low? Explain why that result makes sense given how embeddings are trained.
5. Your `FINDINGS.md` has the `king - man + woman` top-5 with the three input words marked. Where did the expected answer land, and what were the entries above it?
6. Describe an analogy you designed that failed. What do you think the model encoded instead of the relationship you intended?
7. Why does `V / norms` without `[:, None]` fail, and what does NumPy's broadcasting rule actually do with shapes `(200, d)` and `(200,)`?
8. If every vector is unit-normalized, what is the relationship between dot product and cosine similarity? What computation does that let you skip?
9. From 3B1B: what is a "neuron" holding in a trained network, and how is that different from what you assumed a neuron was before watching?
10. From 3B1B: what job do weights do versus what job the bias does? One sentence each.
11. Your nearest-neighbor results: name one pair that came back highly similar that you consider a *wrong* answer, and hypothesize why the model thinks they're close.
