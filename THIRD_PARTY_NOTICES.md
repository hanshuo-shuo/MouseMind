# Third-party notices and provenance

MouseMind is an independent project. The data pipeline, schema audit, MLP
baseline, hierarchical policy, closed-loop evaluation, failure mining,
reporting, privacy guard, public demo, tests, and Slurm orchestration under
`mouse_llm/` were built for MouseMind.

The following small dependency closures are included for reproducibility.

## MiniMind

- Upstream: [jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- Author: Jingyao Gong and MiniMind contributors
- License: Apache License 2.0

| Included path | Upstream source commit |
| --- | --- |
| `model/model_minimind.py` | `dddedc688121028dd8adab55b95d139ecd87205c` |
| `model/model_lora.py` | `4a68da72d5a6c0c8817805b2b627bb935280b12a` |
| `model/tokenizer.json`, `model/tokenizer_config.json` | `101d7df2da053f55b915128a733ba8b34ebe1ae5` |
| `trainer/train_lora.py` | `3f1a7cc25b19a861cd1bd6ed313be526b9ecdaf8` |
| `trainer/trainer_utils.py` | `101d7df2da053f55b915128a733ba8b34ebe1ae5` |
| `dataset/lm_dataset.py` | `101d7df2da053f55b915128a733ba8b34ebe1ae5` |

These files provide the 64M model architecture, tokenizer, native-PyTorch LoRA
implementation, and SFT loading path used by MouseMind. MouseMind does not
claim authorship of them.

MiniMind's requested citation is:

```bibtex
@misc{minimind,
  title = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  url = {https://github.com/jingyaogong/minimind},
  note = {GitHub repository}
}
```

## Cellworld and Mice environment runtime

- Environment origin: [hanshuo-shuo/Mice](https://github.com/hanshuo-shuo/Mice)
- Extracted environment source commit:
  `e6b2984d58949b20aa8b20eb706dbb22acd1b5ed`
- Recovered legacy observation contract commit:
  `67e769fd410b325b5d2c517d9d5966e5e80fac23`
- Cellworld Python package: [germanespinosa/cellworld](https://github.com/germanespinosa/cellworld)
- Bundled `cellworld_game` runtime license: MIT; see
  `mouse_llm/envs/mice/_vendor/LICENSE`

Detailed environment extraction and local packaging changes are recorded in
`mouse_llm/envs/mice/SOURCE.md`.
