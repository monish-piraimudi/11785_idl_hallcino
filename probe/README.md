# probe

A baseline MLP truthfulness probe, after Azaria & Mitchell's *"The Internal
State of an LLM Knows When It's Lying"*: a small MLP (256→128→64→1, ReLU,
sigmoid) over per-token hidden-state features from a few sampled layers.

It trains against **mock data** right now. `hallucinot` (the collection
pipeline this is meant to consume) needs a real Gemma 4 12B checkpoint and a
rented GPU instance (see `infra/README.md`) to produce real features, so this
package exists to get the model/training code right and de-risk that GPU time
before it's spent. See "Known gaps vs. real data" below for exactly what mock
data does and doesn't stand in for.

## Run it

```bash
python -m probe.train --feature-dim 128 --num-layers 3 --fusion scalar_mix
```

Key flags (see `--help` for the rest): `--n-prompts`, `--feature-dim` (the one
thing to change once the real J-lens feature size is known), `--fusion`
(`concat` or `scalar_mix`), `--epochs`, `--seed`, `--output`.

Every run writes to `results/probe_<timestamp>/` (or `--output <path>`):

| file | contents |
|---|---|
| `data.safetensors` | the mock dataset: `features [row, feature_dim, num_layers]`, `labels`, `prompt_ids`, `token_positions` |
| `predictions.jsonl` | one line per token row: `prompt_id`, `token_position`, `split`, `ground_truth_hallucination`, `predicted_prob`, `predicted_hallucination` |
| `model.pt` | trained `state_dict` + config (`feature_dim`, `num_layers`, `fusion`) needed to reload it |
| `metrics.json` | run args, per-epoch `train_loss`/`val_acc`, scalar-mix layer weights, detection-latency summary |

## What's in here

- **`mock_data.py`** — `generate_mock_dataset(n_prompts, feature_dim, num_layers=3, ...)` returns a `MockDataset(features, labels, prompt_ids, token_positions)`. Each prompt gets a random token count and a single hallucination label, broadcast to every token generated for that prompt (there's no token-level judge anywhere in this repo — see `docs/RESEARCH_NOTES.md` — so "wrong" can only be assigned per sequence, not per token). A shared random direction in feature space carries that label into every row, scaled by layer depth and token position, then buried in noise.
- **`model.py`** — `LayerFusion` (`concat` or `scalar_mix`, the ELMo-style learned softmax weighting from Tenney et al.) collapses the `num_layers` axis into a fixed-width vector; `MLPProbe` is the 256→128→64→1 head on top. `feature_dim` is a constructor argument, not hardcoded, since the real J-lens feature size isn't settled yet.
- **`train.py`** — training loop + the persistence described above. Splits by `prompt_ids`, not by row — rows from the same prompt share a label and a correlated signal, so a row-level split would leak across train/val. Also reports **detection latency**: for each held-out hallucinated prompt, how many tokens in did the probe first flag it (plain accuracy doesn't capture "catching it earlier is better").

## How this maps onto real `hallucinot` output

`hallucinot collect` writes `records.jsonl` (keyed by a stable per-question
`id`) and `features/example-<hash>.safetensors` per example, with a tensor
actually named `j_transport`, shape `[generated_steps, selected_layers,
feature_dimension]` (see `docs/USAGE.md`). The mock schema mirrors that
directly:

| mock (`probe/`) | real (`hallucinot`) |
|---|---|
| `features` | `j_transport` |
| `feature_dim` | `feature_dimension` (= model's `d_model`) |
| `num_layers` | `selected_layers` |
| `prompt_id` (int) | `id` / example hash (joins to `records.jsonl`) |
| `token_position` | `generation_step` |
| `labels` (synthetic) | **doesn't exist yet** — needs a judge |

Swapping mock data for real data means replacing `generate_mock_dataset` with
a loader over `hallucinot.storage.load_saved_example` across a run's
examples — `probe/model.py` and `train.py` don't need to change, since they
only depend on the `(row, feature_dim, num_layers)` shape.

## Known gaps vs. real data

- **Labels don't exist for real data.** `hallucinot` explicitly doesn't grade
  answers (see its own README). The `Judge`/`MockJudge` classes that could
  produce a label live only on `main` (`src/hallcino/judge.py`), judge the
  *whole answer* against a reference, and aren't wired into `hallucinot` or
  `probe/` at all. Real training needs this built first.
- **Mock feature statistics are made up.** `generate_mock_dataset` injects a
  linear signal into Gaussian noise so training obviously works. Real
  `j_transport` will have whatever distribution a real Jacobian transport of
  real Gemma hidden states has — no guarantee it's linearly separable, no
  guarantee the signal-to-noise ratio resembles the mock's. Accuracy and
  detection-latency numbers from mock data are a code-correctness check, not
  a preview of real performance.
