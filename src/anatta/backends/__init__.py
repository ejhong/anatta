"""Model backends and the model registry (clean-lab step 3).

  registry     registry.yaml: one entry per model key (HF id, revision, layers, chat template, backends)
  hf           anatta.models.load from the pinned snapshot, plus LoRA adapters (anatta.player.attach_adapters)
  vllm         one vLLM generator (LoRA serving, Ministral text-only), used inside Modal containers
  openrouter   the OpenRouter client (moved from lab/explore/scout/client.py)
  serving_check  the paid check that vLLM and HF agree on greedy decoding (Modal)

anatta.models stays the module it was (calibration records hash src/anatta/models.py by path), so the registry and
backends live here. Importing this package imports nothing heavy; import the submodule you need.
"""
