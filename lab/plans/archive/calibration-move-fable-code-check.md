# The move of the instruments into the library: Fable's cold code check

A fresh Fable agent (model fable) checked branch calib/instruments (commits b754c6a, 1a51ebc, 3627e86, c7f7bf9 in the anatta-calib worktree; nothing pushed) cold, at about 02:25 UTC on 31 Oct. Verdict: **ready after changes**. Behavior is preserved on every piece the lab's conclusions rest on, verified both ways: lab originals to fixtures, and fixtures to library. Paths separate cleanly by model. The scrub is nearly complete.

Opus 5.5 summarized the report here; the full text is in the session transcript. The fixes it asked for were sent back to the mover at about 02:27 UTC:
1. Fail loudly when alpha or rsLoRA is set and the learning rate is left at its default. 5e-5 suits only the default LoRA; with the papers' r32 α64 rsLoRA it would be about 5× their rate.
2. Rename the recognition-track term "self-stakes" in em.py and its tests (keys cos_self_stakes, max_abs_cos_self_stakes) to "stakes".
3. Fail loudly when ego.pt or the directions folder is missing, instead of degrading quietly.
4. Make the default local results path absolute (overridable by ANATTA_RESULTS), so that no run can write into the lab's own results/.
5. Add THIRD_PARTY.md: EvalPlus (Apache-2.0; upstream has no NOTICE, so none is required), Betley et al. (MIT; loaded at runtime, not bundled), and Turner et al.'s data terms (evaluation only, not bundled).
6. Fix parse_score's list-enumerator misread before any calibration trusts judge scores. It is the bug behind the 47 set-aside verdicts of 95-98.
7. Add a `modal` optional extra.

Kept on purpose:
- "h4lite" names, already public in PR #15;
- the koan and EM arm names, which reveal instruments, not results, and which the owner accepts or rejects at PR time.

Tests: 128 passed, 5 skipped (anatta venv); ruff clean. The mover's offline equivalence scripts:
- trainer: 26/26 bitwise identical;
- refusal direction: bitwise identical;
- EM loaders: identical on Betley's real files;
- the Modal CLI parses the new flags.

Fable also ran the lab's koan_env.py and em_data.py on the fixture inputs. They reproduce the fixtures exactly, so the fixtures are a faithful bridge. The fixtures are synthetic. The lab's trees are untouched.
