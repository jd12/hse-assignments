# A03 · Transformer LLMs 01–03 + Karpathy Tokenizer Pt. 1

**Meetings:** D03–D05 · **Points:** 15 pts

**Video/Source Link(s):**  
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/): lessons 1 (Introduction), 2 (Understanding Language Models: Language as a Bag-of-Words), 3 ((Word) Embeddings)  
[Let's build the GPT Tokenizer, Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE) `00:00:00`–`01:00:00` (Unicode, code points and UTF-8, and byte pair encoding worked by hand). The daily plan splits this across the three meetings. Stop at the hour; building the tokenizer class is A04.

**Notes**  
This is the week you stop treating a string as a string.

Karpathy's central move: text is not a sequence of characters, it's a sequence of bytes. `"héllo"` is 5 characters and 6 bytes. If you build anything on `list(text)` this week you will get a working tokenizer that breaks on the first emoji. Use `text.encode("utf-8")`.

Watch the sections on `ord()`/`chr()` and codepoints twice. A character, a codepoint, and a byte are three different things, and A04 is built on the difference.

Use `get_encoding` with an explicit encoding name. `encoding_for_model` will happily hand you a different vocabulary if the model string doesn't match, and then your numbers aren't comparable to anyone else's.

The notebook ends with `get_stats` and `merge`, the two functions all of A04 is built from. Write them today while nothing else is competing for your attention. Every assertion in that section has to pass before you start A04, and they are the same contracts `tests/test_helpers.py` checks next week.

**Do**

**Step 1. Branch, and open the log for the unit.**

A01's `dev/setup` pull request is still open, and it should be. You merge it when A01 is finished and I have approved it, which is probably tomorrow morning rather than right now. Leave it alone and start today's branch beside it.

If `dev/setup` has already merged, branch from `main`. If it has not, branch from it, so today's work sits on top of yesterday's rather than beside it:

```
cd ~/version_control/hse-2026-2027-tokenizer-<your-username>
git switch dev/setup      # skip this line if dev/setup is already merged
git pull
```

```
cd ~/version_control/hse-2026-2027-tokenizer-<your-username>
git switch main
git pull
git switch -c dev/tokenizer-probe
git status
```

*You should see* `On branch dev/tokenizer-probe`. If it says `On branch main`, the switch did not happen and everything you do next lands in the wrong place. Fix it now, not at the end.

**Step 2. Close the setup log branch and open the one for Unit 0.**

Once A01 is genuinely finished and I have approved it, the setup unit is over: merge that log pull request and delete the branch. If A01 is still open, leave it and come back. Either way, open the Unit 0 branch today, because today's entry belongs on it:

```
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git switch main
git pull
git switch -c <your-username>-unit0
git push -u origin <your-username>-unit0
bash scripts/start-entry.sh
```

**Open a pull request for this log branch**, same as before: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, then leave it alone. **This one stays open until 16 October.** Every day you push the log, that day's entry lands on this pull request, and it merges when you elect your track.

**Step 3. Install tiktoken.**

Back in the tokenizer repo:

```
cd ~/version_control/hse-2026-2027-tokenizer-<your-username>
uv add tiktoken
```

*If it broke* with a Rust toolchain error, the install is trying to compile from source, which means you are on Python 3.13. Go back to 3.12; A01 step 2 told you this. Check with `uv run python -V`.

**Step 4. Open the notebook, and point it at the right Python.**

A notebook is not a file you run from the terminal. You open it in an editor and run it a  
cell at a time. Use **VS Code**, which you already have from A01 and which behaves the same  
on Mac and Windows.

```
code .
```

That opens the whole repo. Open `notebooks/01_tokenizer_probe.ipynb` from the sidebar. If  
VS Code offers to install the **Python** and **Jupyter** extensions, accept; they are the  
only two you need and it will not ask again.

**Now the step everybody gets wrong.** Top right of the notebook there is a button reading  
**Select Kernel**. Click it → **Python Environments** → choose the one whose path contains  
`.venv`, inside your repo. VS Code defaults to whatever Python it found on your system,  
which is *not* the environment `uv` built and *not* where `uv add tiktoken` put anything.

Prove it before you write any code. First cell:

```
import sys
print(sys.executable)
```

*You should see* a path ending in `.venv/bin/python`, on Windows `.venv\Scripts\python.exe`,  
and it must sit inside your tokenizer repo.

*If it broke:* `ModuleNotFoundError: No module named 'tiktoken'` immediately after `uv add  
tiktoken` succeeded is this and nothing else. The package installed fine; the notebook is  
running a different Python. Re-select the kernel. The error names tiktoken, so it will send  
you looking at tiktoken, and tiktoken is not the problem.

*If the kernel picker will not cooperate at all*, the repo already ships JupyterLab, so you  
can sidestep VS Code entirely:

```
uv run jupyter lab
```

That opens in your browser with the correct environment already selected, because it is the  
only one there. Either tool is fine. Nothing later in the course depends on which you pick.

**Everything today goes in this one notebook, and it already tells you where.** Do not start  
a second one for the video. The notebook opens with a table of what belongs where; the short  
version is three things that do not mix.

**Section 0, Scratch.** Karpathy writes a lot of code on screen. Type it as he goes, in  
section 0, which exists for exactly this. It is never read for correctness and nothing below  
it may depend on it.

**Sections 1 to 3, the probe.** None of this is in the video. He does not run these probes;  
you do, on strings you chose. These are your measurements and nobody else's numbers will  
match yours.

**Section 4, `get_stats` and `merge`.** His algorithm, your implementation, and the one place  
his work becomes your committed code. Type it rather than pasting it. A04 opens this file  
expecting those two functions, ports them into `bpe/tokenizer.py`, and runs a test suite  
against them, so the version that counts is the one you can explain out loud.

This is a lab notebook, not a clean artifact. It is allowed to be long and it is allowed to  
contain the thing you tried that did not work. The git filter strips output before committing,  
so length costs you nothing in review, and the wrong turns are the part I actually read.

**Before you commit, prove it runs from nothing.** In VS Code, **Restart** the kernel, then  
**Run All**.

*You should see* every cell execute in order with no `NameError`. A notebook you have been  
poking at for two hours passes interactively because a variable you deleted forty minutes ago  
is still alive in memory. It fails the moment anyone else opens it, and the deliverable says  
*run top to bottom*, which is this and not the version in your head.

**Step 5. Confirm the notebook filter is working.**

`setup.sh` already installed the git filter that strips notebook output before committing, so  
your pull request shows your code instead of 4,000 lines of re-numbered cells. Verify it took  
rather than assuming: run a cell, **save the notebook** (`Cmd+S` / `Ctrl+S`), then

```
git diff --stat
```

*You should see* a one-line change. If you see hundreds of lines of `execution_count` and  
`outputs`, the filter did not install. Re-run `./setup.sh` and check its final two lines  
before you write anything else.

The starting move, in the next cell:

```
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
enc.encode("hello")
[enc.decode([t]) for t in enc.encode("hello")]
```

*You should see* a list of integers, then the pieces those integers decode back to. Look at  
the pieces, not just the count. The count tells you how much you pay; the pieces tell you why.

**Step 6. Run the probe set.**

Tokenize each of these with `cl100k_base` and record both the count and the actual decoded pieces:

`"1234567890"` · `"12,345,678"` · `" hello"` vs `"hello"` vs `"Hello"` · four spaces vs a tab · `" def foo():"` · a 100-word English paragraph vs the same paragraph in Spanish, Japanese, and Arabic · `"🙂🙃"` · `"strawberry"`

The multilingual comparison only means something if it is the *same* paragraph. Translate one, do not pick four different ones.

**Step 7. Write `get_stats` and `merge`.**

They are at the end of the notebook with assertions under them. Run the assertions.

*You should see* no output at all. An assertion that passes is silent. If you see `AssertionError`, read which one fired before you change anything, because the two functions fail in different ways and the traceback names which.

**Step 8. Write the findings.**

Create `tokenlab/FINDINGS.md` with the token count and decoded piece list for every probe string, plus **three concrete claims** about the tokenizer that your own data supports. A claim is a sentence someone could disagree with. "Tokenization is interesting" is not one.

**Step 9. Commit, push, open the pull request.**

```
git add notebooks/01_tokenizer_probe.ipynb tokenlab/FINDINGS.md
git commit -m "A03: tokenizer probe and findings"
git push -u origin dev/tokenizer-probe
```

**Open the pull request.** Refresh the repo page, click **Compare & pull request** for `dev/tokenizer-probe`, leave the title as your commit message, pick **jd12** under **Reviewers**, click **Create pull request**, and stop.

**Merge when the assignment is finished and I have approved it, not before.** That will usually land a day or two into the next assignment, because the next one opens before this one is due. Branches are independent, so having two open at once is normal and is not a sign you are behind.

**Step 10. Close the log.**

```
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs
git commit
git push
```

Answer today's reflection questions in the log in your own words, and set the **AI use** line to `none.` or to one specific sentence. Push the same day you write: git records when you wrote it, and that timestamp is not something either of us gets to argue about later.

**Deliverable**  
`notebooks/01_tokenizer_probe.ipynb` run top to bottom with every assertion passing, plus `tokenlab/FINDINGS.md`: the token count and decoded piece list for every probe string, plus 3 concrete claims about the tokenizer that your data supports.

**Reflection Questions**

1. How many tokens is `" hello"` versus `"hello"`? What does the leading space do, and why is that design choice reasonable?
2. Report your token counts for the same paragraph in English, Spanish, Japanese, and Arabic. What is the ratio between the best and worst case, and what does that mean for a user in the worst-case language paying per token?
3. How does `cl100k_base` split `"12,345,678"`? Based on that, explain why LLMs are unreliable at arithmetic on large numbers.
4. How many tokens is `"strawberry"` and what are the pieces? Use this to explain the letter-counting failure mode.
5. In UTF-8, how many bytes is `"é"`? How many codepoints? Why does that difference break a tokenizer built on `list(text)`?
6. Karpathy iterates over `text.encode("utf-8")` rather than the string itself. What specifically goes wrong if you don't?
7. From lesson 2: what does bag-of-words throw away that a modern language model keeps, and give a sentence pair that bag-of-words cannot distinguish.
8. From lesson 3: what is the difference between a token ID and an embedding vector? Which one carries meaning, and which is just an index?
9. Why is a raw byte-level vocabulary (256 tokens, no merges) technically sufficient but practically terrible?
10. What was the most surprising thing in your probe data: something you'd have bet money against before running it?
