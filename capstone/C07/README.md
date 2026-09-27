# C07 · CODE FREEZE

**Meetings:** D101–D103 · **Points:** 15 pts

**Video/Source Link(s):** None.

**Notes**  
**Freeze is enforced without exception.** No new features. Tag the commit. Everything after this is documentation, evals, and rehearsal.

Learning when to stop is the point of this week, and it is a real professional skill that almost nobody teaches. The instinct to add one more thing is exactly the instinct that ships broken software.

Your eval report needs: metric, baseline, final number, run count, spread, and where the system got worse. A report with no regression listed reads as a report nobody looked at hard.

`TECHNICAL.md` is a filled-in structure, not an essay: architecture, the three decisions that mattered, the mechanism behind the biggest failure, and the numbers. Bullets and diagrams, and every claim carries either a number or a file reference. The *why* gets said out loud in the defense rather than written at length, because a written explanation is easy to fake and a spoken one is not.

**Deliverable**  
Tagged freeze commit + `EVAL_REPORT.md` (metric, baseline, final, N runs, spread, regressions) + `TECHNICAL.md` complete as a structured document.

**Discussion · Fri May 7, in class · technical defense, final**  
The real one. Fifteen minutes each, and I will push on the weakest claim. Same questions as the C06 round, plus the two below. Ungraded, and it is the last thing before freeze.

1. What did you freeze that you wish you had one more week for, and what would you have done with the week?
2. On symposium night, what is the most likely way the live demo fails, and what is your recovery?

**Reflection Questions**

1. Final metric versus baseline, with run count and spread.
2. Where did your system get *worse* than the baseline? Every real system has such a place, find yours.
3. What is the mechanistic explanation for your system's biggest remaining failure?
4. What did you freeze that you wish you'd had one more week for?
5. If someone asked "how do you know this works," what is your one-sentence answer and what number backs it?
6. What is the most likely way your live demo fails on symposium night?
