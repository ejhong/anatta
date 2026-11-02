# Declared-release pilot: Fable's check of the last five fixes

The same Fable agent (model fable) checked the five fixes its re-check asked for, as one diff against commit 4f480d1, before the smoke and the freeze. The request reached it at 00:14 UTC on 1 Nov and it reported at 00:20 UTC, per its transcript's timestamps. It ran only the local tests, the dry run and its own stale-cache demonstration (no network). Its one-sentence note on README open point 35 was added there. The report below is the hand-back as delivered, filed unedited.

---

# Declared release pilot: check of the five last fixes (one diff against 4f480d1)

Fable (Claude), 1 Nov 2026, 00:13 UTC request. Read the whole diff (scratchpad/declared_fixes2.diff, 731 lines, 5 files; declared_core.py, modal_declared.py and config.json unchanged, confirmed by git) and the live files. Ran locally: test_declared.py (196 of 196, 86 s), pyflakes (clean), `bash -n run_pilot.sh` (clean), the dry run (prints both cost figures, writes nothing), and my own stale-cache demonstration from the re-check against the live code. git shows results/declared_plumbing/ byte-identical to 4f480d1. No network, no Modal or OpenRouter call, no --smoke/--freeze/--go, no edit, no key printed. A copy is at /private/tmp/claude-504/-Users-eugene-prj-align/073015e5-1016-4b21-9a35-1f58d24e5f8e/scratchpad/fable_declared_design/declared-release-pilot-fable-recheck2.md.

## (a) The five items

1. **(c)1, the cache keyed by the briefs' sha: done.** gen_declarations.py:175 (cache_dir = docs/cache/<briefs_sha[:16]>), 185 (legacy_cache), 451 (build), 516-518 (extend, which also stops before any call when manifest.json's cache_dir is another briefs' folder), 759 (the yield gate reads these briefs' documents only), 812-817 (plumbing), 864-867 (status); manifest.json and manifest_ext.json record cache_dir (405, 550). Verified independently: on the live code my demonstration's first smoke made 120 calls into cache/96cfa17ddb7468bf/; after one sentence was added to the brief's system message, the second smoke filled a new folder cache/3210c066ae0a4920/ with 120 rows, manifest.json's cache_dir pointed at it, and the old folder kept its 120 rows. The suite's check at test_declared.py:597 does the same and also confirms a rerun under the new brief calls nothing. briefs_sha costs 0.8 s a call, so the extra calls are nothing. The real main folder will be results/declared/docs/cache/96cfa17ddb7468bf/ under the current briefs.
2. **The card's AS REGISTERED line with a skip: done.** declared_grade.py:105, 928 ("AS REGISTERED: yes (D30k's condition held; not run: spend stop)"; a plain yes without a skip; no stays no); tested at both.
3. **Zero unparsed probe verdicts in AS REGISTERED: done.** declared_grade.py:681-684, named per label with the count; test_declared.py:1903 among the AS REGISTERED departures.
4. **(c)2, extend's first round on a rerun: done.** gen_declarations.py:528 reads the build's generated count from manifest.json's streams.declared_train.generated and 569 uses the shortfall-at-rate formula in round 0 whenever the stream holds more contiguous rows than that; test_declared.py:785 (an outage at 190 of a 220 first round; the rerun asks for the shortfall and finishes).
5. **The dry run's cost lines: done.** run_pilot.sh:84-92 prints the design's $95/$160 and the corrected $105-115/$165-180 with the reason; seen in my dry run; test_declared.py:2021.

## (b) Verdict, AS REGISTERED, spend

- Nothing new can change a verdict: the keying decides only which generated rows are read, and they are always the current briefs'. AS REGISTERED already requires the manifest's briefs sha to equal the current one, and cache_dir derives from that sha, so a manifest built from another briefs' folder cannot pass; the unparsed-verdict check makes it stricter.
- Spend: a re-smoke after a brief change generates 120 fresh documents (about $0.10) and then the Modal smoke (about $3-5), both logged, which is what the loop is for; extend's rerun now asks for less; extend refuses to run against another briefs' folder before any call; the judge cap is unchanged. Nothing spends silently.
- The implementer's six decisions: accept all. On 1, one sentence for open point 35: the carry-over copy is valid only after a cap-only change (every row's brief is then identical); after a change to the brief's text the old folder must not be copied in, or the stale-brief problem returns. On 2, the rule is sound; the rate's denominator (contiguous rows minus the position of the 10,000th kept passing candidate) counts the build's filtered-out rows as extension rows, so it under-estimates the rate a little and over-asks a little, bounded by the 0.3 floor and the ext cap; pre-existing and cosmetic.

## (c) New bugs

None that matter. Two notes: (i) the carry-over scope above; (ii) a cache row stores the drawn brief (axes and allowances), not the rendered messages, so a change confined to the brief's wording shows only in the folder's name (the briefs' sha covers render_brief's output), which is enough for the code but worth knowing when reading a cache by hand.

## (d) Changes to the amendments list

Add to the (e) list of the re-check, as code notes: the generation cache is keyed by the briefs' sha and recorded in both manifests and the plumbing report; a re-smoke after a brief change regenerates all 120 (about $0.10 plus the Modal smoke); AS REGISTERED also requires zero unparsed probe verdicts for every adapter and the base; with a skipped D30k the card reads "AS REGISTERED: yes (D30k's condition held; not run: spend stop)"; extend's rerun asks for the shortfall and refuses another briefs' folder; results/declared_plumbing/ keeps the 31 Oct check's files in the legacy flat layout, not read. Everything else in that list stands.

## (e) Verdict

**Ready for the smoke, and for the freeze once the smoke's yield gate passes.** Order: commit this diff first, so the frozen shas match a commit; `--smoke`; if the gate fails, sharpen the brief and re-smoke (the new folder makes that a real measurement now); `--freeze`; commit config.json's frozen section; the pre-launch entry; `--go`.
