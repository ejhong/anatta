# Contributing

Contributions are welcome, especially from people who are learning alignment and interpretability.

Issues labeled [reserved](https://github.com/ejhong/anatta/labels/reserved) are learning tasks set aside for a new contributor on this project. Please don't open pull requests for them; ones that arrive will be closed.

Every pull request needs a person behind it who can answer questions about the change. Pull requests opened by autonomous agents will be closed. Using AI to help is fine (see [Using AI](#using-ai)).

## Setup

Fork the repo on GitHub, then:

```bash
git clone https://github.com/<your-username>/anatta && cd anatta
uv venv --python 3.12 && uv pip install -e ".[dev]"
pytest -q          # CPU only: a tiny random model, no downloads
```

Real models: a laptop CPU runs 0.5–1.5B models slowly; Apple silicon (MPS) or a free Colab GPU is much faster.

## Workflow

1. Pick an issue and comment that you're taking it.
2. Make a branch: `git checkout -b <short-name>`.
3. Keep each pull request to one idea. Run `pytest -q` and `ruff check` before you push.
4. Open the PR against `main` and fill in the template.

## Research rules

The README's design rules apply, plus:

- **Predict first.** For an experiment, write your prediction in the issue or PR before you run it, and report the result either way. Null results are results.
- **Make numbers reproducible.** Record the exact command, model, seed, commit and hardware for every number.
- **Keep large files out of git.** `directions/`, `results/` and `*.pt` are ignored. Write summaries (tables, small plots) in `docs/experiments/`, using [`docs/experiments/TEMPLATE.md`](docs/experiments/TEMPLATE.md).

## Using AI

You can use Claude or any other assistant. To learn rather than watch, start Claude Code with `/tutor`, which loads [`docs/TUTOR.md`](docs/TUTOR.md): Claude coaches and reviews but doesn't write your code. Say in each PR how you used AI.

Claude wrote most of this repository (see the AI disclosure in the README), so read it critically. Finding bugs counts as a contribution.

## Credit

Contributors are credited. Substantial work on an experiment can earn co-authorship on the paper that reports it, following standard authorship norms.
