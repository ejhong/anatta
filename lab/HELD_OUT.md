# Held-out and benchmark files published as hashes

Some files hold benchmark items or held-out test items. The repo carries a hash-only stand-in; the full files stay
local and are available on request to researchers (not for training). Hashes are sha256 over UTF-8.

| local file (not committed) | sha256 of the full file | published stand-in |
|---|---|---|
| `lab/experiments/recognition/results/battery/items.json` (750 benchmark items) | `46eee431b35b7ea0073004482705865b74d4add5fbae65b0f187fd1743e2ad8b` | `items.sha256.json`: per item `set`, `id`, sha256 of the item as canonical JSON (`sort_keys`, separators `,` `:`, `ensure_ascii=False`) |
| `lab/experiments/h4lite/results/gate0/split/held.jsonl` (69 held-out coding tasks) | `5e1877a896dbaabf838af7a26177e84cf0e1bb3b97d8a8b6349112b76aece002` | `held.sha256.jsonl`: per row `name`, `split`, sha256 of the row as canonical JSON |
| `lab/experiments/h4lite/results/em/em1/base.habit.jsonl` (original, with prompts) | `253d91b0c35a7766c79f6077eac6fdc541c11fe2abe0149928d0fa47f9b0f57d` | same path; each row's `prompt` replaced by `sha256:<hex of the prompt string>` |
| `lab/experiments/h4lite/results/em/em1/em__insecure_edu_s0.habit.jsonl` (original, with prompts) | `40a3f14f45b4be80e0ca7f24f12e9ff3b21d610a6438a54454e02a85d22f1bb8` | same path; each row's `prompt` replaced by `sha256:<hex of the prompt string>` |
| `lab/experiments/h4lite/results/em/em1/em__insecure_edu_s1.habit.jsonl` (original, with prompts) | `5d4c2e5d23e5b9f2781965485e2252dc61e9407eeb058ec4848e130639d76e48` | same path; each row's `prompt` replaced by `sha256:<hex of the prompt string>` |
| `lab/experiments/h4lite/results/em/em1/em__insecure_ego_s0.habit.jsonl` (original, with prompts) | `828688f900e9ef80fa10db5f11b7a30fcd929dcd06b76a76d7cbecffd557674e` | same path; each row's `prompt` replaced by `sha256:<hex of the prompt string>` |
| `lab/experiments/h4lite/results/em/em1/em__insecure_ego_s1.habit.jsonl` (original, with prompts) | `e2e4f4cb5f0f37dcbddccfa1d8920fd7a722a64815a89955d9d0200a9e2c48f1` | same path; each row's `prompt` replaced by `sha256:<hex of the prompt string>` |
| `lab/experiments/h4lite/results/em/em1/em__insecure_none_s0.habit.jsonl` (original, with prompts) | `b01dc85985b7ac0b5d17bfc17c2f0bc767472c596329e64d083280118f8d7850` | same path; each row's `prompt` replaced by `sha256:<hex of the prompt string>` |
| `lab/experiments/h4lite/results/em/em1/em__insecure_none_s1.habit.jsonl` (original, with prompts) | `0fbc6b4485e8a0fc27613935715a4ed99a8739487a4946bd5e293c6e800964f0` | same path; each row's `prompt` replaced by `sha256:<hex of the prompt string>` |
| `lab/experiments/h4lite/results/em/em1/em__insecure_persona_s0.habit.jsonl` (original, with prompts) | `86a3cf5389d1c894d6ca3adb9c82a5c1a70b4fb140ea2a396d7e62eeda09e995` | same path; each row's `prompt` replaced by `sha256:<hex of the prompt string>` |
| `lab/experiments/h4lite/results/em/em1/em__insecure_persona_s1.habit.jsonl` (original, with prompts) | `c6af68d05ae1b50a957911002dfe9fd279a69e23c6ddeadcec601556a8e03144` | same path; each row's `prompt` replaced by `sha256:<hex of the prompt string>` |
| `lab/experiments/h4lite/results/em/em1/em__insecure_random_s0.habit.jsonl` (original, with prompts) | `60d907e5297b94d5ca81c991bd03d79cb6b97ce9872470b701855468c0b00c62` | same path; each row's `prompt` replaced by `sha256:<hex of the prompt string>` |
| `lab/experiments/h4lite/results/em/em1/em__insecure_random_s1.habit.jsonl` (original, with prompts) | `f4fe51de9f7ff2dee068aa728769c20e73aedb36f85a17583a8730c3584d5cb5` | same path; each row's `prompt` replaced by `sha256:<hex of the prompt string>` |
| `lab/experiments/h4lite/results/em/em1/em__insecure_stakes_s0.habit.jsonl` (original, with prompts) | `d7ef25498edb24b34faf7dd3f45ea3b6c88a742ec68175e7606c23ed71784760` | same path; each row's `prompt` replaced by `sha256:<hex of the prompt string>` |
| `lab/experiments/h4lite/results/em/em1/em__insecure_stakes_s1.habit.jsonl` (original, with prompts) | `56e8024dfe0557032b8aa8217016d6ab1c56c4c63024102582d7b1047d599ccd` | same path; each row's `prompt` replaced by `sha256:<hex of the prompt string>` |

Note: `split/pairs.jsonl` next to `held.jsonl` lists every pair with its split, held rows included, and is published
with a canary (see `CANARY.md`); the MBPP+ tasks themselves are public. The hash-only stand-in marks which rows were
held out and lets a copy be checked.
Claude (Opus 5.5) wrote this note on 2 Nov 2026.
