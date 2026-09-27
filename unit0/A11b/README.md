# A11b · Q1 Self-Assessment

**Meetings:** D22–D23 · **Points:** Ungraded

**Notes**

Short. No research, no polish. It feeds the narrative comment I write about you at the end of the quarter, so a real answer serves you better than an impressive one. "I still don't understand attention" is more useful to both of us than a paragraph implying you do.

Cite your own work: assignment ids, file paths, commit hashes where they help. "I got better at debugging" is not evidence. "A05 took me two periods because of the `[:, None]` broadcast, and I caught the same class of bug in twenty minutes in A09" is.

It goes in the foundations repo, not the log.

**Walkthrough — SELF_Q1.md**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch main && git pull
git switch -c dev/self-assessment
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, add these to the day's checklist:

```markdown
- [ ] Read my first September entry next to my latest one
- [ ] SELF_Q1.md, all eight answered
- [ ] Log-hours count
- [ ] Push, open the PR, merge it
```

This shares the A11 meetings; one log entry per meeting covers both.

**Step 2. Read your own log first.**

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
ls logs/
```

Open the first entry you wrote in September and read it next to the one you wrote yesterday. Most of what you should say is visible in the difference between those two.

**Step 3. Write `SELF_Q1.md`.**

At the top of the foundations repo, with these eight headings, each answered in a few sentences:

```markdown
# Q1 Self-Assessment

## 1. Which assignment did I lose track of time on, and what specifically pulled me in?
## 2. Which one did I finish without really understanding? (There is one.)
## 3. What can I do now that I could not do on September 1? (Specific enough to check.)
## 4. What is still confusing that I expect to stay confusing without help?
## 5. How did I actually work this quarter: when, in what size sessions, how close to the deadline? (Not how I meant to.)
## 6. When I got stuck, what did I do first, and how long before I asked anyone?
## 7. One thing I did this quarter that nobody graded and nobody saw.
## 8. What should you know about me as a student that the assignments don't show?
```

**Extension — when you actually worked (ASSIGNED)**

Question 5 is the one people answer with what they meant to do. Check it:

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git log --date=format:'%a %H' --format='%ad' -- logs | sort | uniq -c | sort -rn | head -10
```

*You should see* up to ten lines, each a count of log commits at a day and hour. Paste them under heading 5 of `SELF_Q1.md`, and add one sentence: does it match what you wrote above it?

Then:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git add SELF_Q1.md
git commit -m "A11b: Q1 self-assessment"
git push -u origin dev/self-assessment
```

Open the pull request, **jd12** under **Reviewers**, **Create pull request**. This is the one assignment you merge without waiting for approval: it is ungraded and there is nothing for me to block on.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`SELF_Q1.md` at the top of the foundations repo, eight answers and the log-hours count.

**Reflection Questions**

1. Paste the first line of your first September log entry and the first line of your latest one. What does the difference say that your answer to heading 3 does not?

2. From the log-hours count: the day and hour with the most commits. Is that during Period C, the afternoon after, or late at night, and does heading 5 say so?

3. Name the assignment in your answer to heading 2 and paste one line from its reflection answers that you would now rewrite. Say what you would change.
