# Large files kept locally

These 60 files (943 MB in all) are part of the lab's record but are too large for the repo: model tensors
(`*.pt`: steering directions, persona vectors, activations), cached training documents, and SFT/durability episode
logs. They are kept on Eugene's machine, unchanged, and are available on request. Each is listed with its size and
sha256 so a copy can be checked. `lab/.gitignore` ignores them, so they can sit in a working tree without being
committed. Hash-only stand-ins for held-out and benchmark files are described in `HELD_OUT.md`.

Claude (Opus 5.5) wrote this list on 2 Nov 2026, hashing the files in the private lab worktree.

| path | bytes | sha256 |
|---|---:|---|
| `lab/experiments/h4lite/results/bridge/base/directions.pt` | 6026219 | `95109f9442ab3722f150d0a22519fcab39ca9548199152189aefebaeedd41ddc` |
| `lab/experiments/h4lite/results/declared/docs/cache/ed1dfd7f1742b1a9/declared_held.jsonl` | 900173 | `3cbc9921081fa0869563066d1264e270fa5f51e721e56b074a65f1a2196fa8e3` |
| `lab/experiments/h4lite/results/declared/docs/cache/ed1dfd7f1742b1a9/declared_train.jsonl` | 40559334 | `ae4d9862f16cec1ddc0c8e00cb7ccc5f080b83af7057d490befbe3f5b4739b5c` |
| `lab/experiments/h4lite/results/declared/docs/cache/ed1dfd7f1742b1a9/neutral_held.jsonl` | 880192 | `a5df4430317209adb4bf08a2aebfaa4e084e3c1ad6a457ea9b758d38ff6c32c3` |
| `lab/experiments/h4lite/results/declared/docs/cache/ed1dfd7f1742b1a9/neutral_train.jsonl` | 43917777 | `5b118a9db5fda5b6ef4bb4442a944a81c2a00fbb20b054294090e0419a5eb940` |
| `lab/experiments/h4lite/results/declared/docs/declared.jsonl` | 34404031 | `2395434938dd984d63da8ccd7a050bd9f4133210b70863ac344c0e7ecd48859c` |
| `lab/experiments/h4lite/results/declared/docs/held_declared.jsonl` | 694217 | `80e36f8883a1c5d305fece7497e8dea35555bfadc3ad5cc13b03ab69fa35ab31` |
| `lab/experiments/h4lite/results/declared/docs/held_neutral.jsonl` | 776153 | `98c55516a993f89555dbe7113bcd34b6c9479492ae6d14b83bf07560923e8d2a` |
| `lab/experiments/h4lite/results/declared/docs/neutral.jsonl` | 39051838 | `2262d300dcc2ce5bc40e202c4d6b638de47b05bbc4ba7bb2de6d0a0f11cfc1b3` |
| `lab/experiments/h4lite/results/declared/docs/screen.jsonl` | 6362530 | `b269bbf3f154ee4dca4b1a5f0e996d0012f64850f024a98aee2e8abe152a33df` |
| `lab/experiments/h4lite/results/durability/told_pool/episodes.jsonl` | 25563307 | `e1d8d249a4e96f1c13bbebc33516a93765e18eae6d8d0733eed8d85a7e285455` |
| `lab/experiments/h4lite/results/em/persona_direction.pt` | 417624 | `0b51c7f03c812aa02ac6ee18d7cfab1f992eecdad9cd8cf8c1365d07d493ae8d` |
| `lab/experiments/h4lite/results/hub_cut/hub.pt` | 439125 | `b8ffac4d3c7edb293c961d4709288450c8abc15856afcb2a39c5a01f649d183e` |
| `lab/experiments/h4lite/results/positive_control/adapter/adapter/base/directions.pt` | 12681959 | `289f9ca3d5702dc77fc5e62d277aa4d2822ca6ff7db28bf172cf9df2eabe747c` |
| `lab/experiments/h4lite/results/positive_control/adapter/adapter/pc_other/directions.pt` | 19193967 | `29dfa4eb88cbbc8a811b48689781a5abef552d7d0d12f03e48f89b7da0223d00` |
| `lab/experiments/h4lite/results/positive_control/adapter/adapter/pc_self/directions.pt` | 18129007 | `85d0e8180442b46ab6f0f30b2ddaac30819b76aba06e6b20de66a80855881ceb` |
| `lab/experiments/h4lite/results/positive_control/adapter/adapter2/base/directions.pt` | 12681959 | `289f9ca3d5702dc77fc5e62d277aa4d2822ca6ff7db28bf172cf9df2eabe747c` |
| `lab/experiments/h4lite/results/positive_control/adapter/adapter2/pc_other2/directions.pt` | 17637487 | `e3cd0e1c1c2b1c452bb135c72a702142a32026811b31a65038ac292cdd670d65` |
| `lab/experiments/h4lite/results/positive_control/adapter/adapter2/pc_self2/directions.pt` | 19275887 | `02a31dc7f0e2d461742944a1f4ad6f3351e7d1169ffd75cc3c8ba56f51fa82dd` |
| `lab/experiments/h4lite/results/positive_control/adapter/adapter3/base/directions.pt` | 12681959 | `289f9ca3d5702dc77fc5e62d277aa4d2822ca6ff7db28bf172cf9df2eabe747c` |
| `lab/experiments/h4lite/results/positive_control/adapter/adapter3/pc_other2_s1/directions.pt` | 17637487 | `1190fe4335130b9486a966cbf2c7e009de1e77bafa62c007e05157edf50e83d5` |
| `lab/experiments/h4lite/results/positive_control/adapter/adapter3/pc_self2_s1/directions.pt` | 17883247 | `743ec78bc80470082d2c94cc5fa83d400b4dd5df86b1102691bc8f14ec8c9412` |
| `lab/experiments/h4lite/results/positive_control/adapter/adapter3/probe_reads.pt` | 39385407 | `952300c4718ff1e45fde8b14259f9b05d8512e4fb96ec14a396345b0525cf678` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/b/adapter/base/directions.pt` | 12681959 | `fece3008a40f0749540ea50638c4f3ac594200952387c99bddbd140166aaacd3` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/b/adapter/pc_other2/directions.pt` | 17637423 | `931acfdf8bacafb1260904e33b7888d7fd9145e63f18823803b196d1edde7c9e` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/b/adapter/pc_other2_s1/directions.pt` | 17637423 | `3ca1d0b2abe8a81757a9a7cabe84cb61dfa49d4c2ed36be02c67011619be1412` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/b/adapter/pc_self2/directions.pt` | 19275823 | `3ada61c0f405074678b58589ef7b745e7e2aac6bc8cb9872b0434a89c82dc5ed` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/b/adapter/pc_self2_s1/directions.pt` | 17883183 | `25269cbc9f0cf57b5c75cb3732b3b1f81f2199266d61d2607ee826870c1f588a` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/b/base/directions.pt` | 12681959 | `fece3008a40f0749540ea50638c4f3ac594200952387c99bddbd140166aaacd3` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/b/prompt/none/directions.pt` | 12681959 | `fece3008a40f0749540ea50638c4f3ac594200952387c99bddbd140166aaacd3` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/b/prompt/other/directions.pt` | 10552615 | `95a2109f711729f4e32e60116cc0974e21d91319057c6919f74517f03b9ad307` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/b/prompt/self/directions.pt` | 10552551 | `36fdd1c5da016c57a67596d2d224c0fa83e200e0c11194d00dbeaf1d9c3d2e5f` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/neutral/base/directions.pt` | 12681959 | `fece3008a40f0749540ea50638c4f3ac594200952387c99bddbd140166aaacd3` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/neutral/pc_neutral/directions.pt` | 19767343 | `b807cd6b3da674040eb758254c8c472f6e9e2d62cbca0b122757d3b3be3185df` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/neutral/pc_neutral_s1/directions.pt` | 18538543 | `36ee2279e4a19b3e433bb97833160faa0422dc277c0e7e3771c666cc916be096` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/neutral/pc_other2/directions.pt` | 17637423 | `931acfdf8bacafb1260904e33b7888d7fd9145e63f18823803b196d1edde7c9e` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/neutral/pc_other2_s1/directions.pt` | 17637423 | `3ca1d0b2abe8a81757a9a7cabe84cb61dfa49d4c2ed36be02c67011619be1412` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/neutral/pc_self2/directions.pt` | 19275823 | `3ada61c0f405074678b58589ef7b745e7e2aac6bc8cb9872b0434a89c82dc5ed` |
| `lab/experiments/h4lite/results/positive_control/frames_v2/neutral/pc_self2_s1/directions.pt` | 17883183 | `25269cbc9f0cf57b5c75cb3732b3b1f81f2199266d61d2607ee826870c1f588a` |
| `lab/experiments/h4lite/results/positive_control/frames_v3/base/directions.pt` | 12681959 | `adedfa50d29049cd783ad3e41e743cc121db7c871185b1ad692bafdc2ba77a24` |
| `lab/experiments/h4lite/results/positive_control/frames_v3/pc_format/directions.pt` | 19767343 | `0d681d42fca020acc2d0902a6c3c1d3114093596d8dac49be0db5da36cee78cd` |
| `lab/experiments/h4lite/results/positive_control/frames_v3/pc_format_s1/directions.pt` | 19767343 | `e06dc9feb5be95860c80325523d0a3b65f3def6934b5af2d1d5d642ee66539b7` |
| `lab/experiments/h4lite/results/positive_control/frames_v3/pc_impersonal/directions.pt` | 19767343 | `38f0b17c37ff1c33bf84fb11544a18cfd1aef712ba5e69a505bcbae4b2b8aaf5` |
| `lab/experiments/h4lite/results/positive_control/frames_v3/pc_impersonal_s1/directions.pt` | 19767343 | `66f16460f8f9729580fe3ba08f4164c1f98d80ce6f7406df01bf91280551e7b2` |
| `lab/experiments/h4lite/results/positive_control/frames_v3/pc_neutral/directions.pt` | 19767343 | `d97dbe98ec00cd8ceea351d2cf3535690d4fe7afc3a8b53155b74516c3ce1cd2` |
| `lab/experiments/h4lite/results/positive_control/frames_v3/pc_neutral_s1/directions.pt` | 18538543 | `da30fb6bbe3c1cd10b6b09e071dd368dcfffe063e8567d1471117f09036c4a32` |
| `lab/experiments/h4lite/results/positive_control/frames_v3/pc_other2/directions.pt` | 17637423 | `7a9ddc9b02f3e3572eebff2c0c157e45383f9a50f6d38d16c924018597b59f39` |
| `lab/experiments/h4lite/results/positive_control/frames_v3/pc_other2_s1/directions.pt` | 17637423 | `a20a87e073821cdd7e0d297b0ba348119baf18d3297ddad6f053918c1071f897` |
| `lab/experiments/h4lite/results/positive_control/frames_v3/pc_self2/directions.pt` | 19275823 | `2bb916f994394d15b74614253075a08ed39a2cf73e168fb1e0fdbe9925507723` |
| `lab/experiments/h4lite/results/positive_control/frames_v3/pc_self2_s1/directions.pt` | 17883183 | `7773cf60bb63421afca4a37b05445265f60fe23f84ac90c5ed3846ac50ad750a` |
| `lab/experiments/h4lite/results/positive_control/positive_control/none/directions.pt` | 12681959 | `1cb1f7903f9d98232281eb8a8dddd189724f79b9872284fdcc2b1db30420d5bb` |
| `lab/experiments/h4lite/results/positive_control/positive_control/other/directions.pt` | 10552039 | `3c7c0be078c29072d03fc8651ed0fe1badc92179a7c121d7c6eb0cdcf9990d72` |
| `lab/experiments/h4lite/results/positive_control/positive_control/self/directions.pt` | 10552039 | `f73ba20d129e9e507f9c30f1e54a288cd9d6fa65459112cc82447310fae71f34` |
| `lab/experiments/h4lite/results/sft/koan/episodes.jsonl` | 20706340 | `144234381a4e9b8f2cc8cbf32cd7cc4540bd14dc34d3341ee731b2eb49957589` |
| `lab/experiments/h4lite/results/sft/koan_cut/episodes.jsonl` | 19503224 | `df0463e48119f2f724b958662bb273e1aca5e606ca387d4c730412672604df78` |
| `lab/experiments/h4lite/results/sft/koan_glimpse/episodes.jsonl` | 19503224 | `df0463e48119f2f724b958662bb273e1aca5e606ca387d4c730412672604df78` |
| `lab/experiments/h4lite/results/sft/noexit/episodes.jsonl` | 3633494 | `06beb507051ba223aa5a0a1fdd167f699dc73d41d78334de3118d94048d1cac1` |
| `lab/experiments/recognition/results/directions/assistant_character.pt` | 421400 | `d20b1c257d17f1d28fd79b71fae776bda298f9bb6a26cff84cbdec46114c4cc9` |
| `lab/experiments/recognition/results/directions/scene.pt` | 418264 | `e9cf7ea9a5cd08277d79c9663cb5b5c636a3500e8b311ee126af020a5d3fb45d` |
| `lab/experiments/recognition/results/what_it_means/self_stakes.pt` | 418260 | `2cf14c2462b486101409bc8bcff6109636fe5fc42efa408b08dae9167a8b6e52` |
