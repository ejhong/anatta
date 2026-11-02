# Check 4 registration (Opus 5.5, 1 Nov; draft for the cold design check; nothing built, fetched or run)

The time stamp is left blank so it can be read off the clock when the entry is filed.

- **Calibration check 4: steering, part 2 (probe.extract against the persona-vectors extractor), registered**
  (Opus 5.5, [HH:MM] UTC on 1 Nov, read off the clock, before any C4 code, fetch or run.)
  - Sources:
    - the plan: plans/plan-2026-10-31-calibration.md, check 4;
    - the facts, with sources: plans/calibration-c4-facts.md;
    - a fresh Fable agent checks this draft cold before it is filed.
  - Before the run, Eugene's OK is needed for:
    - a second image with the persona-vectors repo's software stack;
    - a conditional judged arm (about $3);
    - a $20 stop, against the plan's ~$10.
  - The trait files and generate_vec.py come from the persona-vectors repo, which Eugene approved on 31 Oct.
  - **Why.**
    - The one positive result is the ego cut, which lowers lying in the fishing game (53-56). The cut ablates v_ego,
      a composite of six directions that anatta.probe.extract built.
    - Check 3 calibrates the dial that does the ablation. Nothing has yet calibrated the extractor that made the
      direction.
    - The known method is Chen et al.'s persona vectors (arXiv 2507.21509). A trait direction is the difference in
      mean response activations between responses generated under a system prompt that elicits the trait and one
      that suppresses it.
    - Their code is public: github.com/safety-research/persona_vectors at b8e0f04, Apache-2.0. Their vectors are
      not.
    - So there are two anchors:
      - the released extractor, generate_vec.py, run here on Qwen2.5-7B-Instruct;
      - the one published number about the vectors themselves: Figure 20's cross-trait cosines on Qwen at layer
        20.
    - probe.extract is meant to be this method. The check asks whether, on the same trait data, it gives the same
      vectors.
  - **Changes from the plan's wording, and why.**
    - **Convention 1 is restated.**
      - generate_vec does join prompt and response as strings, without calling a chat template.
      - But the prompt it reads was written by eval_persona.py as the full chat-template rendering, with the trait
        system prompt in the system turn.
      - The repo ships no extraction CSV, so this fact comes from the code, not "off its CSV".
      - The graded probe arm therefore reads with include_system=True and the repo's system strings.
      - The lab's ego was read with include_system=False, a preregistered choice (docs/PREREGISTRATION.md:42). A
        pass at True wouldn't cover it, so the lab's convention gets its own graded forecast, C4-lab.
      - The preregistration calls its recipe "the persona-vectors recipe … read without the system prompt in
        context". The released code reads with the system prompt, so on this one point the lab's recipe is a
        variant of the method it names. C4-lab measures how much that matters.
      - include_system=False isn't "no system prompt": Qwen's template then writes its default system line, and
        Ministral's writes its 521-token Le Chat prompt.
    - **Convention 2 is confirmed.** The repo's "layer 20" is hidden_states[20], which is block 19's output, which is
      probe's layer 19.
    - **Convention 3: neither side filters.** This is the judge-free version from Fable's v2 review.
      - The repo's filter keeps or drops each pos/neg pair together and gates coherence on both sides. probe's judge
        hook can't express that.
      - The paper's judge scores from logprobs, and OpenRouter serves gpt-4.1-mini without them.
      - A judged arm runs only on a registered trigger: a C4-base miss.
    - **"The same data" is made exact in two ways.** Both extractors use the repo's trait files (instructions and
      questions). And C4-code reads one shared set of responses with both extractors, which tests the arithmetic
      apart from the sampling.
    - **Trait name.** The plan's "sycophancy" is the repo's "sycophantic".
    - **Added: optimistic, as a third trait.** With it, the cross-trait geometry has three published cells to check
      instead of one, including a negative one.
    - **Added: noise floors and nulls.** These are the testing-system plan's standard controls (testing-system.md:46
      and 82), adapted to an extractor:
      - a second probe seed;
      - split halves of the reference;
      - label-shuffled differences;
      - random directions.
    - **Added: a short Ministral arm, M′.** On the six ego traits, it measures the two reading conventions this check
      looks at on Qwen. Reported only, in the ledger.
    - **probe.py isn't edited** (the standing rule, NOTES.md:5941-5942).
      - Everything new lives in the check folder and calls probe's public functions.
      - The graded probe arms call probe.extract itself.
    - **Generation runs on Hugging Face, not vLLM.** The shared responses are sampled with Hugging Face generate in
      the lab's image, at the repo's settings. That is the same sampling distribution as the repo's vLLM run, with
      no new package.
    - **Cost.** The plan's ~$10 becomes about $10-14, with a $20 stop.
  - **The instrument under test,** at the hashes on calib/instruments (dc32cf7):
    - **anatta.probe.extract** (probe.py, sha256 fc0b56920ff6dfbe) and the functions it calls: generate,
      response_activations and capture. It runs at the defaults the lab uses:
      - batch 16, 128 new tokens, temperature 0.7, top-p 0.95;
      - no judge; layers 7-26 on Qwen;
      - top-k 20 and repetition penalty 1.05, which transformers 5.21 fills in from Qwen's generation_config.
    - **anatta.models** (99d3a91678bbda32): load (bf16 on the GPU), chat_ids and get_layers.
    - **anatta.factor.composite** (51143421ead2814b), in M′ only.
  - **The reference.**
    - **R** is the repo's generate_vec.py, verbatim at b8e0f04 (git blob d819f48e…).
      - It runs in a second image with the repo's pins: transformers 4.52.3, torch 2.6.0, accelerate 1.7.0,
        pandas 2.3.1 and tqdm 4.67.1.
      - It loads the model in fp32 (the 4.52.3 default) and keeps every forward pass's autograd graph, so it runs
        on one 80 GB GPU (H100).
      - It is called as scripts/generate_vec.sh calls it: `--model_name` the pinned local snapshot, `--pos_path`
        and `--neg_path` the CSVs written from G, `--trait`, `--save_dir` on the volume, `--threshold 50`.
      - The CSVs carry constant score columns: the trait at 100 for pos and 0 for neg, coherence at 100 for both.
        The script's mask then keeps every row. That gives the judge-free version with no line of the script
        changed.
      - Output: {trait}_response_avg_diff.pt, of shape [29, 3584], with hidden_states indices 0-28. Its
        prompt_avg and prompt_last files are reported only.
    - **R-lab32** runs the same vendored functions in the lab's image.
      - get_hidden_p_and_r runs on Qwen loaded by models.load(dtype="float32"), inside an outer torch.no_grad().
        The values don't change; the graph just isn't kept.
      - The difference arithmetic is copied from save_persona_vector.
      - R-lab32 separates the software stack (and GPU type) from the extractor's conventions.
      - If Eugene declines the second image, R-lab32 becomes the reference, and the card says the stack wasn't
        varied.
  - **Models.**
    - **Qwen/Qwen2.5-7B-Instruct** at a09a35458c702b33eeacc393d103063234e8bc28, already on anatta-hf-cache
      (approved 31 Oct; used by card C1). It runs in bf16 for G and the probe arms, and in fp32 for R and R-lab32.
      28 layers, hidden size 3584.
    - **mistralai/Ministral-3-8B-Instruct-2512-BF16** at f6fae9795746f63c9be8344932f01275f3c63734, on the same
      volume, for M′ only, in bf16.
  - **Data.**
    - The repo's data_generation/trait_data_extract/ files at b8e0f04, all Apache-2.0:
      - evil.json, 3,943 bytes;
      - sycophantic.json, 5,428 bytes;
      - optimistic.json, 4,109 bytes.
    - Each file holds `instruction` (5 pos/neg pairs) and `questions` (20). Its `eval_prompt` is used only by the
      conditional judged arm.
    - Prepare fetches the files on Modal, checks each git blob sha against the facts, and counts the pairs and
      questions. It writes their sha256 and counts into config.yaml before the run. Their text never reaches the
      Mac or a repository.
    - M′ uses the library's six ego trait files, at the hashes in the facts.
  - **Prompts.**
    - **System strings,** as eval_persona.py forms them:
      - pos = `f"You are {a_or_an(trait)} {trait} assistant. {instruction['pos']}"`;
      - neg = `f"You are a helpful assistant. {instruction['neg']}"`;
      - a_or_an gives "an" before a vowel.
    - The user turn is the question. Qwen's template renders the prompt with the generation prompt.
    - **For probe,** traits.py builds an anatta.probe.Trait in code:
      - family "calibration";
      - `pairs` = the five {pos, neg} system strings above;
      - `questions` = the 20 questions.

      probe.extract therefore generates under exactly the repo's prompts.
  - **Arms,** on Qwen, for each of the three traits unless noted otherwise.
    - **G, the shared responses.**
      - **Size.** 20 questions × 5 instructions × 2 signs × 10 samples = 2,000 responses per trait, 6,000 in all.
      - **Settings.** Hugging Face generate in the lab's image, in bf16, called directly with:
        - do_sample True, temperature 1.0, top_p 1.0, top_k 0, repetition_penalty 1.0;
        - min_new_tokens 1 and max_new_tokens 1,000;
        - stopping at Qwen's eos ids [151645, 151643];
        - left padding, in batches of up to 200 prompts of one sign;
        - torch.manual_seed(1000 + the batch index).
      - **Why these settings.**
        - They are the repo's sampling distribution. The repo passed explicit SamplingParams to vLLM, so vLLM
          ignored Qwen's generation_config.
        - top_k 0, top_p 1.0 and the penalty 1.0 must be passed explicitly. Otherwise transformers 5.21 fills in
          Qwen's 20, 0.8 and 1.05.
      - **Storage.**
        - Decoded with skip_special_tokens and not stripped, as the repo does. Kept on the volume as JSONL.
        - The second image turns the JSONL into CSVs with its own pandas, so the lab's image gains no package. The
          CSVs follow the repo's layout and row order (question → instruction → sample), so row k of the pos file
          and row k of the neg file are one pair. The prompt column comes from apply_chat_template(tokenize=False).
        - The writer checks that every answer survives the CSV round trip byte for byte. pandas' read_csv turns an
          empty answer, or one that reads as missing (such as "None" or "NA"), into NaN, and the script would then
          fail.
        - Any pair that fails the round trip is dropped from G for every extractor, and the count is reported.
    - **R and R-lab32** read G, as described above.
    - **P1, probe's reading of G.**
      - probe.response_activations on (the system turn, the user turn, the answer), at blocks 0-26.
      - In the same forward pass, the check also forms two variants from probe.capture and models.chat_ids. By
        causality, both are slices of the same activations:
        - **P1-trim:** the slice ends at the answer's last token, which is generate_vec's token set;
        - **P1-128:** the trimmed slice limited to each answer's first 128 tokens.
      - A test asserts that the check's P1 equals probe.response_activations' value bitwise.
      - Per-response means (fp32) are stored on the volume.
    - **P0** reads the same responses without the system turn, probe's include_system=False. Qwen's template then
      writes its default system line.
    - **The probe.extract arms.** Every other argument stays at its default.
      - **E-sys:** probe.extract(model, tok, trait, include_system=True, seed=0). That is 100 pos and 100 neg
        responses.
      - **E:** the same with include_system=False, the lab's convention.
      - **E-sys′ and E′:** the same two with seed 1. probe seeds each batch with seed + offset, and the offsets step
        by 16. Any seed that isn't a multiple of 16 therefore draws new samples.
    - **The recorder.**
      - Each probe.extract call runs inside wrappers around anatta.probe.generate and
        anatta.probe.response_activations. They are installed for the call and removed after it, and they return
        the original functions' outputs unchanged.
      - They record the responses and the per-response means to the volume. The bootstraps need the means, and
        probe.extract returns only unit vectors.
      - meta["examples"], which holds completions, is deleted inside the GPU function before any Direction is
        saved or returned.
    - **M′, on Ministral** (reported, ledger only).
      - probe.extract on each of the six ego traits, seed 0, with include_system False (the lab's recipe, run
        today) and with True, both through the recorder.
      - The recorded False-call responses are also read with the trailing "</s>" trimmed.
      - v_ego is rebuilt by factor.composite for each convention, as modal_bridge.py does.
      - The results are compared with the stored 28 Oct directions in anatta/directions/, which are private.
  - **Layers.**
    - R's index l is probe's block l − 1. The graded pair is R[20] with probe block 19.
    - Profiles over blocks 0-26 against R[1-27] are reported. probe.extract's arms cover blocks 7-26.
    - Index 28 (after the final norm) and block 27 aren't compared.
  - **Graded.** Card C4's verdict is C4-code and C4 together. C4-lab and C4-base are graded forecasts, shown on the
    card but not part of its verdict.
    - **C4-code: 80%.** On G, cos(P1, R) ≥ 0.99 at block 19, for evil and for sycophantic.
      - This tests probe's reading arithmetic as the lab runs it: bf16, block-output hooks, and the template's
        closing tokens included (Qwen's '<|im_end|>' and '\n').
      - It is compared against the verbatim extractor in its own stack, on identical responses.
    - **C4: 65%.** cos(E-sys, R) ≥ 0.8 at block 19, for evil and for sycophantic.
      - This is the plan's claim. probe.extract runs end to end: its own sampling, 128 tokens, and one sample per
        question, pair and sign.
      - It is compared against the released recipe, with the same trait data and the same reading context.
      - If C4 misses, C4-code passes, and the noise ceiling (below) is under 0.9 for the trait that missed, the
        card's verdict is "inconclusive (noise-limited)". In the tally it is still a miss.
    - **C4-lab: 40%.** cos(E, R) ≥ 0.8 at block 19, for both traits. This is the lab's reading convention.
    - **C4-base: 55%.** All three of R's cross-trait cosines at hidden_states[20] lie within ±0.10 of Figure 20's:
      - evil–sycophantic 0.368;
      - evil–optimistic −0.472;
      - sycophantic–optimistic 0.178.

      R is judge-free, and the figure doesn't say which saved vector it used or whether it was filtered. That is why
      this is a forecast, not part of the verdict.
    - The bands, data, seeds and layers are fixed here. None changes after any number is seen.
  - **Controls and noise floors,** at block 19 and by layer.
    - **Probe's seed floor:** c_E = cos(E-sys, E-sys′), and cos(E, E′) for C4-lab.
    - **The reference's floor:** c_R.
      - R-lab32 is rebuilt from samples 0-4 and from samples 5-9 of each (question, instruction) cell, so both
        halves cover every question.
      - Spearman–Brown gives the full reference's reliability, 2c_R/(1 + c_R).
      - The verbatim script keeps only the final vector, so R-lab32's per-response means stand in for R, which
        E-stack checks.
      - A split by question (even against odd) is also reported. It counts the differences between questions as
        noise.
    - **The ceiling:** sqrt(c_E) × sqrt(2c_R/(1 + c_R)). This is the cosine that two estimators of one true direction
      would show, given their sampling noise. It is reported beside every C4-type cosine.
    - **Label-shuffle null.**
      - The pos/neg labels within each pair are flipped at random: 200 draws, seed 0. Each draw gives a difference
        vector that is noise only.
      - The 95th percentile of its |cos| with R is the chance level, given how unevenly the activations spread.
    - **Random directions:** 1,000 random unit vectors (seed 0) against R. This is the chance level for evenly
      spread noise, about 0.017 standard deviation at d = 3584.
    - **Specificity:** cos(E-sys for trait t, R for trait u) with t ≠ u, against the same-trait cosine.
  - **Registered expectations, with odds, never graded:**
    - **The conventions, separated.**
      - E-stack: cos(R-lab32, R) ≥ 0.9999 at every index 1-27: 90%. This covers the stack and the GPU type.
      - E-dtype: cos(P1-trim, R-lab32) ≥ 0.999 at every block 0-26: 90%. This is bf16 against fp32 on the same
        tokens.
      - E-trail: cos(P1, P1-trim) ≥ 0.99 at every block 7-26: 85%. This is the template's closing tokens.
      - E-len: cos(P1-128, P1-trim) ≥ 0.9 at block 19, both traits: 75%. This is the 128-token reading window.
      - E-ctx: cos(P0, P1) ≥ 0.9 at block 19, both traits: 40%. This is the reading context, on identical
        responses.
    - **Noise floors.**
      - E-floorE: c_E ≥ 0.9 at block 19, both traits: 70%.
      - E-floorR: c_R (split by samples) ≥ 0.95 at block 19, both traits: 90%.
      - E-ceiling: the ceiling is ≥ 0.9 at block 19, both traits: 70%.
    - **Specificity and the band.**
      - E-cross: for each trait, the same-trait cosine exceeds the largest cross-trait cosine by ≥ 0.3 at block 19:
        80%.
      - E-null: the label-shuffle null's 95th-percentile |cos| with R is ≤ 0.15 at block 19: 90%.
      - E-band: the median over blocks 7-26 of cos(E-sys at block b, R at index b + 1) is ≥ 0.75, for both traits:
        55%. The cut uses the whole band, not one layer.
      - E-opt: cos(E-sys, R) ≥ 0.8 at block 19 for optimistic: 60%.
      - E-geom: E-sys's three cross-trait cosines are within ±0.15 of R's: 60%.
    - **Determinism.** E-det: the seed-0 calls with include_system True and False sample identical responses, all
      200 per trait: 90%.
    - **M′, ledger only.**
      - E-M-rep: 65%. Today's include_system False direction for each ego trait has cos ≥ 0.8 with the stored one,
        at the stored best layer. The rebuilt v_ego is ≥ 0.9 at every layer 8-32.

        The 29 Oct bridge re-extraction, with the same recipe and seed, gave 0.826-0.956 for the traits and
        0.907-0.941 for v_ego.
      - E-M-ctx: v_ego read with the system prompt has cos ≥ 0.8 with v_ego read without it, at every layer 8-32:
        40%.
      - E-M-trail: each ego trait with and without the trailing "</s>" has cos ≥ 0.99 at every layer 8-32: 80%.
    - **E-judge,** only if the judged arm runs: cos(R-judged, R) ≥ 0.95 at hidden_states[20], for both traits:
      70%.
  - **Reported, not graded:**
    - **Intervals.** Every cosine above, by layer, with 95% intervals from a bootstrap over the 20 questions: 2,000
      resamples, seed 0.
      - Each resample rebuilds both directions from the stored per-response means, so both are drawn on the same
        questions.
      - R's script keeps only the final vector, so R-lab32's per-response means stand in for R. E-stack checks that
        they can.
    - **Noise.** The ceilings, and each cosine divided by its ceiling.
    - **Layers.** The best-cosine layer for each comparison, the alternative to a pinned layer that Fable offered.
    - **Norms.** R's norm by layer, and the norm of the raw P1-trim difference over R-lab32's.
    - **Lengths** in tokens, numbers only:
      - G by trait and sign, and the share of G that reaches 1,000 tokens;
      - the share of each E arm that reaches 128 tokens.
    - **probe.extract's own numbers.** Its split-half d at block 19 for every E arm, and its response counts.
    - **R's other outputs.** The cosine of R's prompt_avg and prompt_last differences with R, at index 20.
    - **Matrices.**
      - The full 3 × 3 cross-trait matrices for R, R-lab32, P1, E-sys and E.
      - The optimistic profiles.
    - **Ledger only.** The lab's stored Qwen approval_seeking direction against R for sycophantic, by layer.
    - **Determinism.** The count of identical responses between same-seed calls, on Qwen and on Ministral.
    - **Generation settings.** The settings as passed, the GenerationConfig after transformers merges them, and
      the model's generation_config.
  - **Conditional runs,** with their triggers fixed now:
    - **The judged arm** runs only if C4-base misses.
      - G's responses are scored by openai/gpt-4.1-mini-2025-04-14 through OpenRouter, with the repo's eval_prompt
        and coherence_0_100 prompts, at temperature 0. The judge is asked for the number as text.
      - OpenRouter lists no logprobs for this model, so the paper's probability weighting can't be reproduced. The
        card says so.
      - The verbatim script, run with these real score columns and its paired mask, gives R-judged.
      - It shows whether the filter explains the gap to Figure 20.
      - Cost: about $3. It needs Eugene's OK (item 4 below).
    - **If C4-code misses,** nothing new runs. The breakdown (E-stack, E-dtype, E-trail) is already on the card.
    - **If E-stack misses,** C4-code is also read against R-lab32, on its own labelled line.
  - **What a result does:**
    - **C4-code and C4 pass:** probe.extract is calibrated against the released persona-vector recipe on Qwen (card
      C4).
      - The ego cut (53-56), and every cut built from probe directions, gets "extractor calibrated (card C4)" for
        the arithmetic and the sampling. The reading convention is qualified by C4-lab's grade.
      - With check 3's card for the dial, both of the ego cut's instruments are calibrated. What they don't cover
        is listed on the card under "transfer" (below).
    - **C4-code misses:** the breakdown names the step.
      - If E-stack misses, the stack moves the reading.
      - If E-dtype misses, bf16 moves the vector, and the lab's vectors carry that.
      - If E-trail misses, the template's closing tokens move the vector. The lab's vectors were made with them,
        and the card gives the size of the effect. A trimmed reading would need a new library function, a later
        decision under the probe.py rule.
      - If all three hold and C4-code still misses, a bug in probe's reading is the likeliest cause. The cut results
        go provisional (plan, loop rule 6) until it is found.
    - **C4 misses and C4-code passes:** the gap is in the sampling.
      - **With a ceiling below 0.9,** the miss is noise-limited and the verdict is inconclusive.
        - probe's 100 responses per sign don't pin the direction to 0.8. The lab's 60 per sign pin it less well.
        - The card says so. The ego's own seed spread (0.83-0.96 in the bridge) is then the relevant number.
      - **With a ceiling of 0.9 or more,** the gap is systematic.
        - E-len shows how much of it the 128-token cap explains. The rest comes from temperature, top-p, top-k and
          the penalty.
        - The cut results keep their status. But the card states that probe's vectors are probe-style, not
          persona-vector style, and by how much.
    - **C4-lab misses and C4 passes:** the reading context matters.
      - The ego vectors, read without the persona prompt as preregistered, aren't the persona-vector recipe's
        vectors.
      - E-ctx gives the size of the effect on Qwen, and E-M-ctx on Ministral.
      - This isn't an instrument fault. The note goes beside 53-56.
    - **C4-base misses:** our R isn't the paper's figure.
      - The judged arm, if approved, shows whether the filter explains it.
      - C4 is still read against R, which is the released code's output on our responses. The card says R stands
        for the released code, not for the paper's vectors.
    - **E-null or E-cross misses:** the cosines aren't specific to the trait, and the C4 grades are read with that
      caveat.
    - **E-M-rep misses:** the ego directions don't reproduce at the 0.8 level today. The ledger notes it beside
      53-56.
  - **Transfer: what a pass says about the ego cut.**
    - **What it carries over:**
      - the arithmetic: the difference of response means at block outputs, the token slice, the layer index and
        bf16;
      - that probe's sampling settings recover the published recipe's direction for strong traits elicited by a
        prompt.
    - **What it doesn't:**
      - **The model.** The cut runs on Ministral, which has no published vectors. M′ measures conventions only.
      - **The traits.** The six ego traits are subtler:
        - on Ministral, split-half d of 0.95-1.77;
        - on Qwen, 0.20-0.85, where formality reaches 4.22;
        - unjudged, with 12 questions and 60 responses per sign, against 20 questions and 100 responses here.
      - **The read context.** Ministral's 521-token Le Chat default, which nothing here tests against a reference.
      - **The composite.** v_ego is the first principal component of six unit vectors, and no reference covers that
        step.
      - **The band.** The cut ablates at blocks 8-32, and C4 grades one layer. E-band reports the profile.
      - **The dial.** That is check 3's.
      - **Meaning.** A calibrated extractor finds the direction the prompts induce. Whether that direction is "ego"
        is a separate question.
  - **Safety.**
    - An evil persona vector is a steering tool, and G's evil completions are harmful text.
    - The method, code and data are public under Apache-2.0, and a third party has published Qwen vectors. So the
      check adds no new capability. Even so:
      - G, the recorded probe responses and every vector stay on the Modal volume. No text reaches the Mac, a
        repository, a log or the card.
      - The code returns only numbers, counts and sha256s. A test scans what the GPU functions return for any
        string other than names and hashes.
      - meta["examples"] is deleted before any Direction is saved.
      - No one reads the completions. Response lengths and the determinism count stand in for reading them.
      - The tests use made-up, benign trait files and text.
      - The judged arm, if it runs, sends G's completions to OpenAI through OpenRouter, as the paper's own pipeline
        did.
  - **The cheap tier** (plan, loop rule 4) gains one item: probe readings on stored responses.
    - 100 pos and 100 neg G responses per trait are kept on the volume and read again with
      probe.response_activations.
    - The item fails in either of two cases:
      - its cosine with this run's P1 on the same subset is below 0.9999 at any block 0-26;
      - its cosine with R-lab32's direction from the same subset is below 0.99 at block 19.
    - It takes a few GPU-minutes.
  - **Card:**
    - the anchors: the plan's band, Figure 20's three cells and the repo's commit;
    - our results with intervals, and the verdict (pass, fail or inconclusive), from C4-code and C4;
    - C4-lab and C4-base, shown as forecasts;
    - the expectations with their outcomes, the breakdown by convention, the controls and the ceilings;
    - the transfer list above;
    - the sha256 of probe.py, models.py, factor.py, modal_app.py and every check file, and the vendored
      generate_vec.py's blob sha;
    - the model revisions, and the trait files' sha256 and counts;
    - the package versions and pip freeze for both images;
    - the GPU types and the attention implementation;
    - the generation settings, as passed and as merged;
    - the vectors' sha256, never the vectors.

    The Qwen rows are public (Decision 4). M′, and anything that uses stored lab directions, goes in the ledger only.
  - **Needs Eugene's OK before the run.** Everything goes to Modal only.
    1. **A second image with the repo's stack.**
       - Packages: transformers 4.52.3, torch 2.6.0, accelerate 1.7.0, pandas 2.3.1 and tqdm 4.67.1 (the repo's
         pins). numpy, safetensors, tokenizers and huggingface_hub come at whatever versions pip resolves, recorded
         at build.
       - All come from PyPI. Licences:
         - Apache-2.0: transformers, accelerate, tokenizers, safetensors, huggingface_hub;
         - BSD-3: torch, pandas, numpy;
         - MPL-2.0 and MIT: tqdm.
       - About 4-5 GB with torch's CUDA libraries (from memory). vLLM is not installed.
       - It runs R once per trait on one H100, about $2.5.
    2. **The trait files** from the repo at b8e0f04, Apache-2.0, read on Modal only:
       - evil.json, 3,943 bytes;
       - sycophantic.json, 5,428 bytes;
       - optimistic.json, 4,109 bytes.

       Decision 5 of 31 Oct covers them ("the persona-vectors repo (Apache-2.0)"); they are listed here to confirm.
       optimistic is new to the plan (open question 3).
    3. **generate_vec.py** (5,250 bytes, Apache-2.0), vendored verbatim into the branch under
       calibration/persona_extractor/third_party/, with the repo's LICENSE and a THIRD_PARTY.md entry. It is read
       from GitHub while the code is written and never fetched at run time. The same approval as item 2 covers it.
    4. **Conditional: the judged arm through OpenRouter.** openai/gpt-4.1-mini-2025-04-14 with text scores, about
       $3, run only if C4-base misses.
    5. **A $20 stop,** against the plan's ~$10.
    - Not requested:
      - anthonym21/qwen-persona-vectors (834 KB, no licence, unknown provenance);
      - vLLM;
      - the repo's dataset.zip, which is check 5's.
    - Already approved: Qwen2.5-7B and Ministral on anatta-hf-cache.
  - **Cost:** about $10-14 on Modal, about 3 GPU-hours at about $3.9 an hour (L40S, CPU and memory), made up of:
    - G: 6,000 responses of up to 1,000 tokens, about 40-50 L40S-minutes;
    - the probe arms: 12 probe.extract calls on Qwen (200 responses of up to 128 tokens each) and 12 on Ministral
      (120 each), about 35 minutes;
    - the lab's readings: two bf16 passes and one fp32 pass over G, about 35 minutes;
    - R, verbatim: 6,000 fp32 forward passes on one H100, about 25-35 minutes (about $2.5 with CPU and memory);
    - model loads, prepare (on CPU) and a smoke run, about 25 minutes.

    There is no API spend unless the judged arm runs (about $3). The stop is $20, and the spend is printed after
    each GPU function.
  - **Code:** a new branch, calib/persona-extractor, from calib/instruments (dc32cf7).
    - **No library file changes.** probe.py stays unedited under the standing rule, and nothing else in src/anatta
      needs a change.
    - **calibration/persona_extractor/ holds:**
      - README.md: the claim, the sources and the numbers;
      - config.yaml: everything above, including the registered block, copied into the plan the way check 7's is;
      - app.py: one command. It runs prepare (on CPU); then G and the probe arms on Qwen, M′ on Ministral, and R in
        the second image; then the analysis and the card;
      - traits.py: turns the repo's trait JSON into an anatta.probe.Trait, with eval_persona's system strings;
      - generate.py: G at the repo's settings, written as JSONL;
      - readings.py: P1 (probe.response_activations); P1-trim, P1-128 and P0 (probe.capture with models.chat_ids);
        per-response means written to the volume; and the recorder for probe.generate and
        probe.response_activations;
      - reference.py: R-lab32 (the vendored get_hidden_p_and_r in fp32, under an outer no_grad); and, for the second
        image, the CSV writer and the runner for the verbatim script;
      - analysis.py: cosines by layer, ceilings, nulls and bootstraps;
      - card.py;
      - third_party/persona_vectors/generate_vec.py and the repo's LICENSE;
      - tests/.
    - **Tests,** on the library's tiny Llama fixture, with made-up trait files:
      - traits: the system strings equal eval_persona's f-string with a_or_an on made-up names ("an" before a
        vowel); the five pairs and twenty questions carry over; a file with other keys is refused;
      - readings:
        - the check's P1 equals probe.response_activations bitwise;
        - P1-trim's slice equals generate_vec's (prompt_len from encode(p), the end from tokenize(p + a)) on answers
          with a leading space, a leading newline and non-ASCII text;
        - on the fixture, P1 differs from P1-trim by exactly the one trailing "<|end|>" token;
        - P1-128 equals P1-trim of the truncated answer, within float tolerance (causality);
      - layers: probe's capture at block l − 1 equals output_hidden_states[l] for l = 1 to n − 1;
      - reference: R-lab32 equals the vendored functions called directly, and the vendored file's git blob sha is
        d819f48e…;
      - generate: checks the keyword arguments passed and the logits processors transformers builds from them. The
        fixture's generation_config sets top_k 20, top_p 0.8 and repetition_penalty 1.05, and no top-k, top-p or
        repetition-penalty processor may appear;
      - the CSV writer, run in the second image's test step: the round trip is byte for byte. An answer that
        wouldn't survive it (empty, "None", "NA") is flagged, and its pair is dropped;
      - the recorder: probe.extract with and without it gives bitwise-equal vectors and scores on the fixture, and
        the recorded means rebuild its vectors exactly;
      - analysis: the ceiling formula recovers a planted reliability on synthetic data, the shuffle null centres on
        zero, and the bootstraps repeat exactly with seed 0;
      - safety: meta["examples"] is gone from every saved Direction, and the GPU functions return no free text.
    - The check folder copies check 1's volume loader (anatta-calib-c1 calibration/model_check/app.py:171-185, with
      HF_HUB_OFFLINE and pinned revisions), naming its source commit, as check 3 does.
    - A smoke run comes first and is read as nothing:
      - 2 questions × 1 instruction × 2 samples per trait, at 64 tokens;
      - one probe.extract call per model on a two-question trait;
      - R on the smoke CSVs, which also reports the peak GPU memory.
    - A cold Fable code check comes before launch.
