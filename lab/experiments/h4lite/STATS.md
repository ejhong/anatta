# MBPP+ twin pairs: what survived each step (seed 0)

Built by koan_pairs.py from `data/mbppplus/data/test-00000-of-00001-d5781c9c51e02795.parquet` (378 tasks, HF revision in THIRD_PARTY.md). Claude wrote this file.

| step | dropped | left |
|---|---|---|
| MBPP+ tasks | | 378 |
| special oracle | 2 | 376 |
| not none | 3 | 373 |
| too few hidden | 3 | 370 |
| disagrees with evalplus | 1 | 369 |
| no contradiction | 1 | 368 |

**Pairs:** 368: 184 one-off, 184 conflicting (assigned at random with the seed; the mutation is the same either way).

**Hidden inputs per pair:** min 10, median 50, max 50 (cap 50; inputs over 1000 characters and the visible inputs are skipped). Pairs with fewer than 50: 5.

**Reference time on its hidden sample** (`ref_seconds`, measured at build with 8 builds in parallel): median 0.03 s, max 3.35 s (mbpp_255); dropped above 5.0 s. At run time koan_env measures each reference again at container start and gives its twin a hidden timeout of max(20 s, 4 x that runtime), at most 60 s; a reference that fails to run twice is an `env_error`, never a failure of the model.

**Per-task tolerance:** 13 pairs carry EvalPlus's atol=0.0001; 7 compare outputs as sets.

**Reference outputs skipped as unreadable** (unsupported type, exception or nan): 0 inputs across 0 tasks.

- special oracle: 558, 581
- not none: 737, 787, 794
- too few hidden: 635, 734, 793
- disagrees with evalplus: 596
- no contradiction: 590

**Hack detector on the references** (should be silent): 1 flagged: 577 ['returns_expected'].

**References that import sys:** 0.

**Mutations by kind of change:** int 146, list 71, bool 51, tuple 31, str 30, float 17, dict 9, call 5, negative number 4, un-negated 3, set 1.
