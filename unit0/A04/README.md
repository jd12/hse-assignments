# A04 · Transformer LLMs 04–06 + Karpathy Tokenizer Pt. 2 (BPE from Scratch)

**Meetings:** ? · **Points:** 15 pts

**Video/Source Link(s):**  
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/): lessons 4 (Encoding and Decoding Context with Attention), 5 (Transformers), 6 (Tokenizers with Code Example)  
[Let's build the GPT Tokenizer, Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE) `01:00:00`–`02:13:34`, to the end. The class gets built in `01:08:00`–`01:41:00`, which is the stretch to code along with rather than watch; after that comes regex splitting, the GPT-2/GPT-4 patterns, tiktoken, special tokens and vocab size.  
[BasicTokenizer Implementation Video](https://youtu.be/zHUWkl7Zvvo)  
[RegexTokenizer Implementation Video](https://youtu.be/9pS22b4TI8g)

**Notes**  
Today you build the thing you measured last week.

The whole algorithm is four moves: count adjacent pairs, take the most common one, replace every occurrence with a new id, repeat. `get_stats` and `merge` from A03 are two of those four, which is why they had to be right before today.

**The bug that will cost you the most time is silent.** Training picks the pair with the highest *count*. Encoding applies merges in order of lowest *merge index*. If you use the same rule in both places, your tokenizer still round-trips perfectly, every string you try comes back correct, and the id sequence is wrong. `tests/test_merges_are_applied_lowest_index_first` exists because of this and nothing else.

Three more that let your code run and quietly produce the wrong answer: `merge` must step by two after a match, not one, or a repeated token gets double-consumed. Your encode loop needs both the length guard and the `break`, or a single character hangs it. And `decode` needs `errors="replace"`, because a byte sequence caught mid-merge is often not valid UTF-8. That last one is correct behavior rather than a workaround, and you will be asked to say why.

The regex split pattern is not decoration. `GPT4_SPLIT_PATTERN` is in `bpe/tokenizer.py` already. It prevents merges from crossing word boundaries, which is what stops the tokenizer from learning `" the"` glued to whatever usually follows it in your corpus.

**Do**

**Step 1. Branch and log.**

**A04 is built on A03.** `get_stats` and `merge` come straight out of last week's notebook, so branch from `dev/tokenizer-probe` unless it has already merged, and from `main` if it has:

```
cd ~/version_control/hse-2026-2027-tokenizer-<your-username>
git switch dev/tokenizer-probe   # or: git switch main, if A03 has merged
git pull
git switch -c dev/bpe
git status
```

*You should see* `On branch dev/bpe`.

Then open today's log entry. The log branch from A03 is still open and stays open, so there is no new branch to make here, only an entry:

```
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-unit0, not main
bash scripts/start-entry.sh
```

**Step 2. Read the tests before you write any code.**

```
cd ~/version_control/hse-2026-2027-tokenizer-<your-username>
pytest tests/ -q
```

*You should see* 2 passed, 30 failed and 36 errors. That is the correct starting state, and the errors are not a broken repo: most tests call an unimplemented function while setting up, which pytest reports as an error rather than a failure. The count you are working toward is **68 passed**: 20 in `test_helpers.py`, 31 in `test_basic.py`, 17 in `test_regex.py`.

The tests are the specification. `tests/conftest.py` imports whatever `bpe/tokenizer.py` currently defines, so there is nothing to configure, and it trains the fixtures on a fixed corpus at `vocab_size=320`. Numbers baked into the tests, the 44 merges and the emoji costing 4 ids, come from *that* corpus. Do not try to reproduce them on yours.

`pytest tests/ -x` stops at the first failure, which is what you want while working. `pytest -k round_trip` runs one group. `-q` alone runs everything, which is what you want before you commit.

*If it broke* with `ModuleNotFoundError: No module named 'bpe'`, you are running from inside `tests/`. Run from the repo root; the path in the error names where Python actually looked.

**Step 3. `test_helpers.py` first. 20 tests, and you already wrote both functions in A03.**

Port `get_stats` and `merge` from last week's notebook into `bpe/tokenizer.py`. If they fail here having passed in the notebook, you changed something on the way across; diff the two.

Four contracts in `get_stats(ids, counts=None)` that the tests check and a working-looking version can miss:

```
counts = {} if counts is None else counts    # NOT `counts = counts or {}`
```

`test_accumulating_across_chunks_sums` hands you an empty dict and calls you twice, expecting the counts to add up. An empty dict is falsy, so `counts or {}` silently throws yours away and starts over. This is the single most common way to fail this file, and it does not fail `test_accumulates_into_a_supplied_dict`, whose dict is not empty.

The other three: keys are **tuples**, not lists. `[1, 1, 1]` contains `(1, 1)` **twice**, counted by position. And neither function may mutate the list it was given.

In `merge(ids, pair, idx)`, step forward by **two** after a match. `merge([1,1,1], (1,1), 4)` is `[4, 1]` and `merge([1,1,1,1], (1,1), 4)` is `[4, 4]`. Step by one and `test_repeated_token_does_not_double_consume` fails, along with eleven others, because everything downstream is built on `merge`.

**Step 4. `test_basic.py`. 31 tests, in three groups.**

I made a video implementing BasicTokenizer. Follow along with it.

[BasicTokenizer Implementation Video](https://youtu.be/zHUWkl7Zvvo)

*Training.* `train(text, vocab_size)` takes a **vocabulary size, not a merge count**: you learn `vocab_size - 256` merges. Reset `self.merges` at the top of the method rather than appending, or `test_training_twice_does_not_accumulate_merges` catches you: it trains twice at 300 and expects 44 merges, not 88. Mint ids from 256 upward, consecutively, and record them in a plain dict, because `test_merges_are_ordered_by_when_they_were_learned` reads insertion order. Build `vocab` so that `vocab[idx] == vocab[p0] + vocab[p1]` for every merge, in the order they were learned, since a later merge can refer to an id an earlier one minted. Take the **most common** pair each round, `max(stats, key=stats.get)`.

*Round trip.* `decode(encode(s)) == s` for everything, including `""`. Your encode loop needs `while len(ids) >= 2` or a one-character string hangs it, and a `break` when the chosen pair is not in `merges` or it never terminates.

*Bytes.* `decode` must end `.decode("utf-8", errors="replace")`. A merge can split a multi-byte character, so a byte sequence taken mid-merge is often not valid UTF-8. That is correct behavior rather than a workaround, and a reflection question asks why. With `vocab_size=256` nothing is learned, so `"hello"` must come back as its five raw bytes and `"🙂"` as four.

**Step 5. `test_regex.py`. 17 tests, and three of them are about where merges are allowed to fall.**

I made a video implementing RegexTokenizer. Follow along with it

[RegexTokenizer Implementation Video](https://youtu.be/9pS22b4TI8g)

Import **`regex`, not `re`**. `\p{L}` is a Unicode property escape and the standard library does not support it; the error points at the pattern rather than at the import, which is where the hour goes.

`RegexTokenizer` must **subclass `BasicTokenizer`**, and its `__init__` must accept a `pattern` argument defaulting to `GPT4_SPLIT_PATTERN`. `test_a_custom_pattern_is_honoured` constructs one with `pattern=r"\S+|\s+"` and expects it to learn a different set of merges.

Train over the pieces, not the flat text: split with `regex.findall`, then accumulate `get_stats` for every chunk into **one** dict before choosing a winner, and apply the merge to every chunk. Count per chunk instead and no pair ever repeats often enough to win. Encode chunk by chunk and concatenate.

Two tests check the *shape* of what you learned rather than any number. No merge may join a letter to a digit, and nothing may ever merge a trailing space onto the end of a word, because the pattern puts a space at the **start** of the next piece. If either fails, your training is not respecting the split.

And one that catches the bug this whole suite exists for. `test_merges_are_applied_lowest_index_first` builds a tokenizer **by hand**: it sets `merges` and `vocab` directly and never calls `train`. So `encode` must read only `self.merges` and `self.vocab`. If it depends on anything your `train` cached, this test fails on an object that never trained. It then encodes `"abcbc"` where `(b,c)` occurs twice and `(a,b)` once, and expects `(a,b)` to go first because its id is lower. Take the **lowest merge index**, `min(stats, key=lambda p: self.merges.get(p, float("inf")))`, not the most frequent pair. Both answers decode back to `"abcbc"`, which is exactly why round-trip tests do not catch it.

**Step 6. All 68.**

```
pytest tests/ -q
```

*You should see* `68 passed`. Read the count before you believe it: `-q` in this repo suppresses the `N failed` summary line, so a run that prints nothing but dots and a `FAILED` line at the bottom is not a pass.

**Step 7. Train on a real corpus.**

Save a text file as `data/corpus.txt`. `data/` is git-ignored on purpose, so your corpus does not travel with the repo and my clone of your work does not carry a copy of whatever you picked.

**Aim for 100KB to 500KB, and no more.** Training is linear in corpus size and this is naive  
pure Python: on a correct reference implementation 100KB takes about 9 seconds, 500KB about  
46, and a megabyte about 90. Yours will be several times slower than that, you train twice in  
this notebook, and you re-train every time you fix a bug. 100KB is enough to see real  
structure.

`train()` takes a **vocabulary size, not a merge count**. Pass `768`: that is the 256 byte  
values plus 512 learned merges. Passing `512` gives you 256 merges, which is half the  
assignment.

Then round-trip the ten probe strings from A03 through it. The notebook already lists them,  
so you are comparing like with like across the two weeks.

*You should see* every probe string come back exactly as it went in, including `"🙂🙃"`. If the emoji comes back as `�`, that is `errors="replace"` doing its job on a byte sequence your merges split badly, and it is worth understanding before you call the day finished.

**Step 8. Compare against A03.**

Open `notebooks/02_bpe_from_scratch.ipynb` the same way you opened A03's: `code .`, then the  
file from the sidebar, then **Select Kernel** → **Python Environments** → the `.venv` one  
inside this repo. It is a different notebook in the same repo, so the kernel choice does not  
carry over automatically. `print(sys.executable)` still settles it in one line, and  
`uv run jupyter lab` is still the fallback.

That notebook carries the same three-way split as A03's, with one difference that matters:  
**your implementation goes in `bpe/tokenizer.py`, never in a cell.** The tests import the  
module, so code that lives only in the notebook passes nothing. Section 0 is the sandbox for  
following the video; the module is where the real version goes.

The notebook runs two comparisons and they answer different questions. The first is bytes  
per token over your whole corpus, which is one number and hides everything; note before you  
report it why it is unfair in your favor. The second is the probe set side by side, token by  
token, and that is the one `bpe/DIFF.md` is built from.

Two results people misread. Your `RegexTokenizer` will usually compress slightly **worse**  
than your `BasicTokenizer`, because the split pattern forbids merges like `b'd '` that glue a  
letter to the following space. That is the trade every real tokenizer makes: a little  
compression for tokens that mean something. And the emoji row prints raw bytes rather than  
characters, because a merge can split a multi-byte character and a single token is then not  
valid UTF-8 on its own. Those bytes are the concrete answer to reflection question 5.

**Step 9. Commit, push, open the pull request.**

```
git add bpe/tokenizer.py notebooks/02_bpe_from_scratch.ipynb
git commit -m "A04: BPE from scratch, 68 tests green"
git push -u origin dev/bpe
```

**Open the pull request** for `dev/bpe`: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop.

**Merge when the assignment is finished and I have approved it, not before.** That will usually land a day or two into the next assignment, because the next one opens before this one is due. Branches are independent, so having two open at once is normal and is not a sign you are behind.

Do not commit a passing test run you did not get. `pytest` output in a pull request I can re-run is not a claim, it is a fact, and I do re-run them.

**Step 10. Close the log.**

```
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs
git commit
git push
```

**Deliverable**  
`notebooks/02_bpe_from_scratch.ipynb` + `bpe/tokenizer.py` (train / encode / decode; stdlib and `regex` only) + `bpe/vocab.json`, which the notebook writes for you + all 68 tests passing + `bpe/DIFF.md`: your vocab vs `cl100k_base`, with three specific differences explained.

**Reflection Questions**

1. What was your corpus, and what were merges #1, #10, and #100? What does merge #1 tell you about your text? The notebook prints these when it trains; to see them again without retraining, `[rt.vocab[i] for n, i in enumerate(rt.merges.values(), 1) if n in (1, 10, 100)]`.
2. Name a token in your vocabulary that does not exist in `cl100k_base`, and one in `cl100k_base` that your tokenizer would never learn. Explain both. For the first, `sorted({rt.vocab[i] for i in rt.merges.values()} - set(gpt4.token_byte_values()), key=len, reverse=True)[:10]`, which is the longest things your corpus paid for. For the second, encode a few words with both and look for one where `cl100k_base` spends a single id and you spend several; a word from a subject your corpus never mentions is the place to look.
3. Your tokenizer and `cl100k_base` on your own corpus, which achieves better compression (bytes per token)? Give both numbers. If yours won, say why that result is unfair in your favor. If it lost, say what about your corpus stopped 512 merges from paying off.
4. Explain the difference between `max(stats, key=stats.get)` during training and `min(..., key=merges.get)` during encoding. What is the observable symptom if you use `max` in both places?
5. Why must `decode` use `errors="replace"`? Give a concrete byte sequence from your run that is not valid UTF-8. The emoji probe is where to find one: encode `"\U0001F642\U0001F643"`, then try `rt.vocab[i].decode("utf-8")` on each id and report the one that raises.
6. GPT-2 applies a regex split *before* BPE. What does that regex prevent from ever becoming a token, and why is preventing it worth the added complexity?
7. Your tokenizer has no special-token handling. What breaks in a chat model if `<|endoftext|>` can be produced by ordinary user text?
8. What is the actual trade-off in choosing vocabulary size? Name one cost of 100k and one cost of 1k.
9. From lesson 4: what problem with fixed-size context encoding does attention solve, in one or two sentences?
10. From lesson 6: what does the code example do that your hand-rolled tokenizer does not, and does that difference matter for correctness or just convenience?
11. Your round-trip assertion: did it pass on the first try? If not, what input broke it?
