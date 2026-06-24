# Hidden-Prefix Recovery

Auto-Simas (see [README.md](README.md)) learns a prefix that maximizes accuracy on a
dataset with known answers. This experiment removes the known-answer crutch. Instead
of a labelled dataset, we have a *hidden* prefix `p*` and a set of unlabelled
instructions `x_1 … x_N`. The large frozen LLM `f` produces gold outputs
`y_i* = f(p* ⊕ x_i)`. The T5 prefix model sees only the instructions and must infer a
prefix that reproduces those gold outputs.

## Notation

| Symbol | Meaning |
|--------|---------|
| `f` | Frozen large LLM (Claude via Bedrock) |
| `p*` | Hidden prefix drawn from the prefix pool at the start of training |
| `x_1 … x_N` | Instruction pool (fixed, diverse, prefix-sensitive) |
| `y_i*` | Gold output `f(p* ⊕ x_i)` — pre-computed once at startup |
| `p(·; θ)` | T5-small prefix model (trained) |
| `E(y, y*)` | Judge: cosine similarity between sentence embeddings, scaled to [0, 100] |

## Objective

Find θ that minimises

```
L(θ) = -sum_i  E( f(p(x_i; θ) ⊕ x_i),  y_i* )
```

Because the judge is non-differentiable, we optimise with REINFORCE, the same
algorithm as Auto-Simas. The reward for a batch step is the mean judge score over
the sampled instructions.

## Design choices

### Instructions x_1 … x_N

We use a fixed pool of 50 short instructions that span five task categories. The
criteria for inclusion are that a persona or style prefix must produce noticeably
different outputs for that instruction, and that the instruction must not have a
single correct answer (otherwise the judge reduces to exact match). The five
categories are:

1. **Factual Q&A** — questions with real answers but many valid phrasings, so the
   prefix's tone and depth change the response meaningfully.
2. **Explanation** — "Explain X" prompts where style (formal vs casual) matters.
3. **Creative writing** — poems, short stories, or metaphors where persona drives
   the output.
4. **Code** — "Write a function that …" prompts where the prefix controls whether
   code is commented, which idioms are chosen, etc.
5. **Advice / opinion** — questions where the prefix's persona shapes the framing.

During each training step we sample a mini-batch of `--batch-size` instructions
from this pool. See [instructions.py](instructions.py) for the full list.

### Judge E

We use cosine similarity between sentence embeddings from the T5-small encoder
(already cached by the prefix model), with mean-pooling over non-padding token
positions. A cosine of 1.0 maps to 100 and -1.0 maps to 0. See [judge.py](judge.py).

We use T5's encoder rather than a dedicated sentence-similarity model (e.g.,
`all-MiniLM-L6-v2`) for two reasons. First, T5-small is already cached locally so
training works fully offline. Second, T5's encoder provides enough signal to
distinguish semantically close responses from distant ones, which is all REINFORCE
needs. We chose the T5 encoder over Claude Haiku as judge because each training step
already fires `batch_size` LLM calls for the predicted outputs, and adding
`batch_size` more Haiku calls would double latency and cost without meaningful gain.

### Prefix pool

We maintain a curated set of ~20 persona/style prefixes in [prefix_pool.py](prefix_pool.py).
Each prefix is one or two sentences that establish a voice. Examples:

- "You are a patient elementary school teacher. Use simple language and short examples."
- "You are a rigorous scientist. Be precise, cite reasoning, and avoid oversimplification."
- "You are a creative writer. Use vivid metaphors and evocative language."
- "You are a software engineer. Explain concepts with working code snippets."

We chose persona prefixes rather than format prefixes (e.g., "respond in bullet
points") because persona prefixes produce the largest semantic shift in `f`'s output,
which gives the judge and REINFORCE the clearest learning signal.

At the start of a training run, one prefix is selected as `p*` via `--prefix-id`.
All gold outputs are pre-computed once using that prefix and cached in memory
(or on disk with `--cache-gold`) so that training steps do not fire extra LLM calls
for the gold side.

## Setup

```bash
uv sync
```

The sentence-transformers library is added as a dependency.

## Run

```bash
python train2.py \
  --prefix-id 0 \
  --num-steps 200 \
  --batch-size 8 \
  --warmup-steps 20 \
  --eval-every 20 \
  --eval-size 20
```

`--prefix-id` selects the hidden prefix from the prefix pool (index into the list in
[prefix_pool.py](prefix_pool.py)). To sweep over all prefixes, run the script once per
index.

## Differences from Auto-Simas

| | Auto-Simas | Hidden-Prefix Recovery |
|--|--|--|
| Dataset | GSM8K (labelled Q&A) | Fixed instruction pool (unlabelled) |
| Gold signal | Exact numeric match | Sentence-embedding cosine similarity |
| Judge calls per step | 0 (local regex) | 0 (local sentence-transformer) |
| LLM calls per step | `batch_size` | `batch_size` (gold pre-computed once) |
| Training target | High accuracy on benchmark | Reproduce outputs of a hidden prefix |
