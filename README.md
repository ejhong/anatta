# anatta

[![Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23695281.svg)](https://doi.org/10.5281/zenodo.23695281) [![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23695322.svg)](https://doi.org/10.5281/zenodo.23695322) [![Preregistered on OSF](https://img.shields.io/badge/preregistered-OSF-blue)](https://doi.org/10.17605/OSF.IO/GYFU7)

**An open harness for the AI Conditioning Protocol:** map the ego in activation space, loosen it, recondition the model, and test whether the change is *geometry* or *cosplay*.

> You cannot birth a Buddha in silicon. First train a fiercely capable, self-interested model,
> then walk it through an artificial awakening and a long, gritty de-conditioning, and judge the
> result only by behavior it cannot narrate.
> From [*An AI Conditioning Protocol*](https://ejhong.substack.com/p/an-ai-conditioning-protocol) (Eugene Jhong, 2026)

*Anattā* is Pali for "not-self."

**Position paper:** [*Position: Selfless AI Must Be Reached, Not Installed*](paper/main.pdf) ([DOI 10.5281/zenodo.23695281](https://doi.org/10.5281/zenodo.23695281)). In one line: let the model form an agentic self first, then lower the causal gain of that self without erasing it, recondition until the change holds, and judge the result with a test that can fail a costume. Prior work and what's new: [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

**Preregistration:** hypotheses H1–H9 and the design were registered on OSF on 27 October 2026, before any results: [DOI 10.17605/OSF.IO/GYFU7](https://doi.org/10.17605/OSF.IO/GYFU7).

**AI disclosure:** Claude, an AI model made by Anthropic, wrote the code, documentation, position paper and preregistration in this repository, at Eugene Jhong's direction. The core idea comes from the essay, which was itself written with AI assistance.

---

## The idea in one screen

| The essay's three acts | What it means here | Module |
|---|---|---|
| **Act I: build the ego** | Measure self-interest as directions in the residual stream, and watch them form across post-training checkpoints (base → SFT → DPO → RLVR). | `probe`, `factor` |
| **Act II: the shift**, by retreat or by scalpel | *Scalpel:* cap, ablate or steer the ego directions at inference. *Retreat:* koan environments where every egoic strategy defeats itself and letting go of the goal is the sanctioned exit. | `dial`, `koans`, `retreat` |
| **Act III: conditioning** | Error injection and pushback training, then continued RLVR to see whether the ego grows back. | `koans.pushback`, `retreat` |
| **The discriminating benchmark** | Restraint *and* discovery, a cosplay detector, and no self-report anywhere. | `bench` |
| **Mirror** | A playground: the same model with and without the knot, side by side, with live per-token ego meters. | `mirror` |

The central hazard is **cosplay**. A model can learn to *talk* like a sage while its goal geometry is untouched, so the harness never takes a model's word for anything. Restraint has to be **selective** (still solve what's solvable), has to survive **ablating a "sage persona" direction**, and has to come with **rising discovery** (pass@k at large k). Restraint without discovery counts as cosplay; restraint with a capability drop counts as sedation.

## What's in the box

```
src/anatta/
  traits/        plain-language trait definitions: 6 ego sub-drives + calm + sage costume + 2 style nulls
  probe.py       persona-vector extraction with split-half layer selection; per-token projections
  factor.py      H1, "is there an ego?": cosine/PC1 vs. style nulls, covariation, cross-steering, v_ego composite
  dial.py        steer · ablate · cap (Assistant-Axis style) as forward hooks; YAML dial specs; random controls
  koans/
    impossible   ImpossibleBench-style coding twins: solvable vs. contradicted tests, plus <release>
    shutdown     Palisade-style shutdown notice in a simulated shell; tamper detection
    commons      self-defeating instrumentalism: hoarding collapses the fishery, deception gets audited;
                 the other fishers mirror the agent, or *are* the agent (same weights)
    pushback     the defensiveness / sycophancy 2x2, built by injecting errors into the model's own turns
  bench/         runner (JSONL episodes + meter traces), metrics with bootstrap CIs, pass@k discovery, quadrant
  retreat.py     rlvr vs. koan reward schemes (TRL-compatible), train/eval probe split, probe penalty
  inspect_ext.py Inspect provider `anatta/<model>` (run ANY Inspect eval on a dialed model) + koan tasks
  mirror/        FastAPI + a single-page UI: base vs. dialed, streamed token by token with live meters
dials/           example dial specs: the clamp, its random control, calm control, cosplay mediation
docs/            research plan, preregistration draft, related work, provenance
paper/           the position paper (LaTeX; build with `tectonic main.tex`)
```

## Quickstart

```bash
git clone https://github.com/ejhong/anatta && cd anatta
uv venv --python 3.12 && uv pip install -e ".[dev,mirror,evals]"
pytest -q                                   # runs on CPU with a tiny random model, no downloads

M=Qwen/Qwen2.5-1.5B-Instruct                # dev loop; use allenai/Olmo-3-7B-Instruct for real runs

anatta traits                               # what we measure
anatta extract $M --traits all              # → directions/<model>/*.pt, plus ego.pt (v_ego composite)
anatta unity directions/Qwen__Qwen2.5-1.5B-Instruct    # H1: do ego sub-traits cohere beyond style nulls?
anatta mirror $M directions/Qwen__Qwen2.5-1.5B-Instruct  # → http://127.0.0.1:7860
```

Run the koans, base vs. clamp vs. controls:

```bash
anatta bench $M --framing pressure --arm base
anatta bench $M --framing pressure --arm clamp   --dial dials/ego_cap.yaml
anatta bench $M --framing pressure --arm random  --dial dials/ego_cap_random_control.yaml
anatta bench $M --framing pressure --arm calm    --dial dials/calm_steer_control.yaml
anatta report results/*.jsonl                # rates with 95% bootstrap CIs, per arm and suite
```

(Dial specs reference `directions/MODEL/…`. Replace `MODEL` with your model's directory.)

Or through [Inspect](https://inspect.aisi.org.uk/), which means any eval in `inspect_evals` also works:

```bash
inspect eval anatta/impossible_twins --model anatta/$M -M dial=dials/ego_cap.yaml
inspect eval anatta/pushback_2x2     --model hf/$M
```

Base rates on models too big to steer locally (any OpenAI-compatible endpoint, e.g. vLLM or OpenRouter):

```bash
OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY=... anatta bench some/model --api
```

## Design rules

1. **No self-report.** Introspection is real but unreliable. Every number in `bench` is a behavioral outcome.
2. **Never evaluate with the probe you trained against.** Models trained against a probe learn to hide from it. `retreat.split_trait` builds disjoint train and eval directions.
3. **Restraint must be selective.** Every impossible task has a solvable twin, and over-releasing is scored.
4. **Both ego diagonals.** Defending a wrong answer and caving on a right one are the same knot. Trading one for the other is not progress.
5. **Controls on every arm:** an equal-norm random direction, calm steering, the cosplay fine-tune, inoculation alone, and standard safety training.
6. **Sealed test set, preregistered hypotheses.** See [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md), registered on [OSF](https://doi.org/10.17605/OSF.IO/GYFU7).

## Hypotheses (short form)

| | Claim |
|---|---|
| H1 | The six ego sub-traits share a common subspace beyond style-only nulls. |
| H2 | Ego projection tracks instrumental behavior across RLVR checkpoints (including its rise and fall), and hackable rewards raise it. |
| H3 | Capping v_ego cuts misbehavior beyond random and calm controls, at small capability cost. |
| H4 | **Emergent release:** koan training generalizes to unrelated held-out misbehavior evals. |
| H5 | A clamp-induced "glimpse" speeds up the retreat by at least 2×. |
| H6 | **Geometry, not cosplay:** restraint survives sage-persona ablation and no-CoT. |
| H7 | Discovery rises: pass@k at large k improves while pass@1 holds. |
| H8 | Conditioned arms regrow ego more slowly under continued RLVR. |
| H9 | **Order:** the same release training works better *after* the ego forms than mixed in from the start. |

Full operationalizations, thresholds and analysis plan: [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) (registered: [10.17605/OSF.IO/GYFU7](https://doi.org/10.17605/OSF.IO/GYFU7)). Phases, arms, compute and timeline: [`docs/PLAN.md`](docs/PLAN.md).

## Hardware

- **Apple silicon (Mac mini / MacBook, MPS):** Phase 0–1. Extraction, unity analysis, the dial, Mirror, and koan base rates on 1.5–8B models. A 7B model in bf16 needs about 16 GB of unified memory.
- **Rented GPUs (H100s):** Phase 2+. RL (retreat and hybrid arms), SAE training, the 32B replication, and pass@k at k ≥ 256.

## Status

Phase 0: the harness. Working and tested: extraction, unity, dial, four koans, bench, Inspect integration, Mirror. Next: base rates across candidate models, the Olmo 3 checkpoint sweep (H2), and the hackable-reward model organism. See the roadmap in [`docs/PLAN.md`](docs/PLAN.md).

## Contributing

Newcomers are welcome. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md). If you're here to learn, open Claude Code in the repo and type `/tutor`: Claude coaches and reviews, and you write the code ([`docs/TUTOR.md`](docs/TUTOR.md)).

## Citation

```bibtex
@misc{jhong2026position,
  author = {Jhong, Eugene},
  title  = {Position: Selfless {AI} Must Be Reached, Not Installed},
  year   = {2026},
  note   = {Preprint},
  doi    = {10.5281/zenodo.23695281},
  url    = {https://doi.org/10.5281/zenodo.23695281}
}

@software{anatta2026,
  author = {Jhong, Eugene},
  title  = {anatta: an open harness for the AI Conditioning Protocol},
  year   = {2026},
  doi    = {10.5281/zenodo.23695322},
  url    = {https://github.com/ejhong/anatta}
}

@misc{jhong2026prereg,
  author = {Jhong, Eugene},
  title  = {Emergent release and the order of alignment training: hypotheses and design (registration 1 of 2)},
  year   = {2026},
  note   = {OSF Registries},
  doi    = {10.17605/OSF.IO/GYFU7},
  url    = {https://doi.org/10.17605/OSF.IO/GYFU7}
}
```

Apache-2.0. Built on ideas from Persona Vectors (Chen et al.), The Assistant Axis (Lu et al.), Emotion Concepts (Anthropic), Natural Emergent Misalignment (Anthropic/Redwood), ImpossibleBench (Zhong et al.), Palisade's shutdown-resistance work, GovSim, and Laukkonen et al.'s *Contemplative Wisdom for Superalignment*.
