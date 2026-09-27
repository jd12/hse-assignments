# A01 · Cold Start: Environment & Repo Bring-Up

**Meetings:** D01–D02 · **Points:** 15 pts

**Video/Source Link(s):** No video. In-house setup. [GitHub Student Developer Pack](https://education.github.com/pack) · Your two repositories, `hse-2026-2027-tokenizer-<your-username>` and `hse-2026-2027-student-log-<your-username>`, both created by Classroom 50. Accept links are in the Do section below.

**Notes**  
Find your AI Agents Fundamentals v2 repo. Do not open the folder that's already on your laptop. Clone it fresh into a new directory and start a stopwatch at the moment you hit Enter on `git clone`. Stop it when the agent returns its first real response. That number is data: do not round it down.

Versions are not suggestions. Node 22 and Python 3.12. (Go arrives in A13, when the Boot.dev CLI first needs it, not today.)

**Two repositories, and they are not the same repository.** Your tokenizer work lives in one. Your log lives in another. They sit side by side in `~/version_control` all year, they have different branch schemes, and the most common Week 1 mistake is committing one into the other. There is no single course repo: each project gets its own, and the RAG project, the GPT build and the capstone each get a fresh one later. That is also why every template ships its own `setup.sh`, and why the secret guard is something you install again each time rather than once in September.

Use `npm ci`, not `npm install`. `npm install` silently rewrites your lockfile and you will then be debugging a different dependency tree than the one you wrote a year ago. If `npm ci` refuses to run at all, that is `package-lock.json` and `package.json` having drifted apart. Record the exact error before you fix it; it is the first real finding of the year.

Your key has a hard spend cap. When you hit it, it does not warn you: requests start failing with a 429 that looks like a rate limit but isn't. Read the error body.

**Do**

Work down this list in order and do not skip ahead: step 6 fails in a confusing way if step 2 has not happened.

**If any step fails, stop and ask me.** Do not paste the error into a chatbot and run whatever it tells you. That habit is how people destroy repositories, and it is separately the exact habit this course spends a year arguing against.

On Windows, do all of this in **Git Bash**, not Command Prompt and not PowerShell. Git Bash ships with Git for Windows and makes every command below work exactly as written.

**Step 1. Make the one folder everything lives in.**

```
mkdir -p ~/version_control
cd ~/version_control
```

Every repository for this class goes in here and nowhere else. **Not Documents, not Desktop, not anywhere OneDrive or iCloud syncs.** Cloud sync and git fight each other and git loses, in ways that are genuinely painful to undo. OneDrive is usually set to back up Documents and Desktop for you, which is why those two are out even though they look innocent.

**Step 2. Install the two version managers.**

Everything below assumes both are on your PATH, and the error you get otherwise is `command not found`, which tells you nothing about which step you skipped.

macOS:

```
brew install uv nvm                      # if brew itself is missing, start at https://brew.sh
                                         # nvm needs two lines added to your ~/.bash_profile.
                                         # brew prints them. Add them, then open a NEW terminal.
```

Windows, in PowerShell once, then back to Git Bash:

```
winget install --id CoreyButler.NVMforWindows -e
winget install --id astral-sh.uv -e
```

Then, in a **new** terminal, on either platform:

```
nvm install 22
nvm use 22
uv python install 3.12                   # tiktoken wheels are missing on 3.13; you hit this in A03
node -v
```

*You should see* `v22.` something from that last command. If it prints `v18` or `v20` or nothing at all, tell me before you leave the room. Nothing in this course runs on an older version and the errors are confusing rather than obvious.

On macOS add `nvm alias default 22`, because there `nvm use` lasts only as long as the terminal window. On Windows `nvm use` changes a global link and already persists, so skip it.

**Step 3. Accept both assignments and clone both repositories.**

Two links. Accept them in this order, sign in to GitHub first, and accept the org invitation in your email before either, or the clone fails with a "repository not found" that is really a permissions error.

**[Tokenizer](https://classroom50.org/Sierra-Canyon/hse-2026-2027/assignments/tokenizer/accept)** · **[Work Log](https://classroom50.org/Sierra-Canyon/hse-2026-2027/assignments/student-log/accept)**

Accepting creates two repositories named after your GitHub username. Clone them beside each other:

```
cd ~/version_control
git clone https://github.com/Sierra-Canyon/hse-2026-2027-tokenizer-<your-username>.git
git clone https://github.com/Sierra-Canyon/hse-2026-2027-student-log-<your-username>.git
ls ~/version_control
```

*You should see* both folder names listed, and nothing else that matters.

**Step 4. Tell git who you are.** *(Once per computer, not once per class. If you did this for another of my classes, skip to step 6.)*

```
git config --global user.name "Your Real Name"
git config --global user.email "you@sierracanyon.org"
```

Your real name. Every commit you make this year is signed with it.

**Step 5. Point git at your editor, and copy in the terminal setup.**

```
git config --global core.editor "code --wait"
```

The `--wait` matters. Without it git opens the file and carries straight on as though you had already saved, and your commit message comes out empty. If `code` is "command not found," open VS Code, press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows), type *Shell Command: Install 'code' command in PATH*, hit enter, then quit your terminal and reopen it. A terminal only learns about new commands when it starts.

Now the terminal files, run **from inside your log repo**:

```
cd ~/version_control/hse-2026-2027-student-log-<your-username>/git_files
cp git-commit-template.txt ~/.git-commit-template.txt
git config --global commit.template ~/.git-commit-template.txt
cp git-prompt.sh ~/git-prompt.sh
cp git-completion.bash ~/git-completion.bash
cp bash_profile_course ~/.bash_profile
cd ..
git config --global pull.rebase true
git config --global diff.colorMoved zebra
```

Quit your terminal completely and reopen it. Your prompt should come back in color and should show the branch you are on. **That branch name in the prompt is not decoration.** It is the thing that stops you committing to the wrong place, which is the failure this whole day is arranged to prevent.

**Step 6. Put the class key in your shell profile.**

Not in the repo, not in a `.env` file, not pasted into a chat window. `~/.bash_profile` is the file your shell reads at login on both platforms.

```
echo 'export OPENAI_API_KEY="the-key-I-handed-you"' >> ~/.bash_profile
source ~/.bash_profile
echo ${#OPENAI_API_KEY}
```

*You should see* a plausible length, not `0`.

Print the length, never the key. `echo $OPENAI_API_KEY` puts the real thing in your scrollback, and scrollback ends up on a projector eventually. If the length is 0 inside VS Code but correct in Terminal, VS Code was launched before you edited the profile. Fully quit it and relaunch.

**Step 7. Bring up the tokenizer repo.**

```
cd ~/version_control/hse-2026-2027-tokenizer-<your-username>
./setup.sh
```

`setup.sh` installs Python 3.12 and the dependencies, installs a git filter so notebook output never reaches a commit, installs the secret guard, and runs the environment check. It ends by printing two things for you to verify by hand. Do them.

*If it broke* with `uv: command not found`, you skipped step 2. If it broke with `bad interpreter: /bin/bash^M`, git converted the line endings on checkout; run `git config --global core.autocrlf input`, delete the folder, and clone it again.

Confirm the secret guard actually fires rather than assuming it did:

```
echo 'sk-proj-notarealkeyjustatest0000000000' > fake.txt
git add fake.txt && git commit -m "test the guard"
```

*You should see* the commit **rejected**. Then clean up:

```
git reset && rm fake.txt
```

Git hooks live in `.git/hooks/` and do not travel with a clone, so a hook in my copy of your repo does nothing for you and yours does nothing for a partner. You install it in every clone you ever make, including the v2 clone in step 8.

**Step 8. Clone the v2 agent fresh and run it.** The stopwatch starts on this `git clone`.

```
cd ~/version_control
git clone <your AI Agents Fundamentals v2 repo URL>
cd <the directory it just made>
npm ci
npm run <the script that starts your agent>
```

Open `package.json` and read the `scripts` block instead of guessing the last line. No two v2 repos in this class have the same entry point, because each of you named it yourself last spring.

*If it broke* on the first model call, read the error body rather than the status line. A 401 means the key never reached the process, so check step 6 in that same terminal. A 429 is usually your spend cap rather than rate limiting.

Ask it something that requires a tool call, not "hello". Stop the stopwatch on the first real response.

Then check its history, because deleting a file does not remove what it contains:

```
git log -p -- .env
git log -S "sk-" --oneline
```

If a real key was ever committed, the key is still in history. Rotate it, then tell me.

**Step 9. Open the log's branch for this unit.**

You never work on `main` in the log. One branch per unit, named your GitHub username plus the unit, **hyphens and not slashes**:

```
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git switch main
git pull
git switch -c <your-username>-setup
git push -u origin <your-username>-setup
```

Mine would be `jd12-setup`. Git cannot hold a branch called `jd12` and a branch called `jd12/setup` at the same time, which is why it is a hyphen.

**Open a pull request for the log branch.** Refresh the repo page on GitHub, click **Compare & pull request**, pick **jd12** under **Reviewers**, click **Create pull request**, then leave it alone. Setup is a short unit, so this branch merges as soon as A01 is finished and approved. Every log branch after it stays open for its whole unit.

**Step 10. Run the two log scripts today, so neither is new on a day it matters.**

```
bash scripts/start-entry.sh
```

That creates today's entry, writes a timestamp header, and opens it in your editor. Under the header, write what you set out to do as a checklist. Then, at the end of the period:

```
bash scripts/sign-off.sh
```

`bash <script>` hands the file to bash to run, which is why you never have to make anything executable. Say in one honest word why anything is unfinished. "No time" is a complete answer. "Got stuck on step 5" is a better one, because I can act on it. Leave the **AI use** line reading `none.` if you did not use it, and replace it with one specific sentence if you did. You are not in trouble for using it. You are in trouble for not saying so.

```
git add logs
git commit
git push
```

`git commit` with no `-m` opens the template. Keep the top line under 50 characters and write it as a command: *Add setup entry*, not *added setup entry*.

**Step 11. Open the tokenizer repo's branch and write `SETUP.md`.**

Two repos, two branch schemes, and they are not the same. The log uses `<your-username>-<unit>`. **Work repos use `dev/<something-short>`**, one branch per assignment:

```
cd ~/version_control/hse-2026-2027-tokenizer-<your-username>
git switch -c dev/setup
```

Write `SETUP.md` at the top level. It is a lab notebook, not an essay: short, exact, and copied from your terminal rather than remembered. Use these six headings in this order, and put something under every one of them.

```
# Setup — <your name>

## 1. Time to run
<mm:ss> from Enter on `git clone` to the first real agent response.
The question I asked: ...
The tool it called: ...

## 2. Errors, in order
For each one: the step number, the exact command you ran, the first line of the error, and the exact command that fixed it. "Nothing broke" is allowed only if it is true.

## 3. check_env.py
Paste the full output of `uv run python check_env.py` here, unedited.

## 4. Secret guard
Paste the terminal output from the fake.txt test in step 7: the `git commit` command and the rejection message.

## 5. Notebook filter
Open `notebooks/01_tokenizer_probe.ipynb`, run one cell, SAVE, then paste the output of `git diff --stat` and of `git config --get filter.nbstrip.clean`. The diff should be a few lines, not thousands of lines of output.

## 6. GitHub Student Developer Pack
One of: `active since <date>`, `applied on <date>, pending`, or `not applied — <why>`.
```

Paste real output between triple backticks. Lengths, not keys: if any pasted output contains `sk-`, you have pasted a key, and the secret guard will refuse the commit for exactly that reason.

Then:

```
git add SETUP.md
git commit -m "A01: environment and repo bring-up"
git push -u origin dev/setup
```

The `-u` is needed the first time you push any new branch, because until then GitHub has never heard of `dev/setup`. Every later push on this branch is just `git push`.

**Open the pull request.** Refresh the repo page on GitHub. There is a banner with a **Compare & pull request** button for `dev/setup`. Click it, leave the title as your commit message, click **Reviewers** and pick **jd12**, then **Create pull request**, and stop.

**A01 runs across two meetings.** Keep working on this same branch on day two rather than opening a second one, and merge it when the setup is actually finished and I have approved it.

**Step 12. Merge when A01 is done, and not before.**

Both pull requests stay open until the setup actually works: `check_env.py` clean, the secret guard proven to fire, the notebook filter proven to strip, and the agent run timed. That is the logical end of A01, and it is probably day two rather than today.

When you are there, push, tell me, and merge both once I have approved them. If A03 has already started by then, that is fine and expected. Assignments overlap, so you will often have the previous branch open beside the current one. What you must not do is merge work that is half-finished because the calendar moved on.

**Deliverable**  
`SETUP.md` on branch `dev/setup` **of your tokenizer repo**, not of your v2 repo, with its pull request open and me as reviewer: your recorded time-to-run, every error you hit in order with the exact command that fixed it, `check_env.py` output, proof the secret guard rejected a test commit, proof the notebook filter is stripping output, and your GitHub Student Developer Pack activation status.

**Reflection Questions**

1. What was your time-to-run, measured from `git clone` to first agent response? Was the first thing that broke your code, your environment, or a dependency you didn't pin?
2. Name one dependency in your v2 repo whose installed version today differs from what your lockfile expected. What changed about it?
3. What exactly is the difference between `npm install` and `npm ci`, and which one would have caught a lockfile that no longer matches `package.json`?
4. Why does `nvm use 22` not persist across terminal sessions, and what does `nvm alias default` actually change on disk?
5. Where do git pre-commit hooks live, and why does that location mean the secret guard does not protect a teammate who clones your repo?
6. Your secret guard blocked a test commit. What pattern did it match on. And name one real secret format it would *not* have caught.
7. If an API key was committed six months ago and then deleted in a later commit, is the key exposed? Explain in terms of how git stores history.
8. Your class key has a hard spend cap. What HTTP status and error body do you get when you exceed it, and how is that distinguishable from an ordinary rate limit?
9. Which environment variable does your v2 agent actually read for its key? Cite the file and line where it's read.
10. What is one thing about your own past setup that you now consider a mistake you would not repeat?
