# Auto-Simas

A frozen large LLM `f: text → text` takes a context and produces text. An
evaluator `E: text × text → ℝ` scores the LLM's output against a reference.
Given a dataset of benchmark inputs `{x_i}`, we build a prefix function
`p(x; θ)` — represented by a small T5 model — and feed the LLM the concatenated
context `p(x; θ) ⊕ x`. Simas tunes `p` by hand; Auto-Simas learns `θ` via
REINFORCE.

## Notation

| Symbol | Meaning |
|--------|---------|
| `f` | Frozen large LLM (Claude via Bedrock) |
| `p(·; θ)` | T5-small prefix model (trained) |
| `E(y, y*)` | Evaluator that scores predicted output `y` against reference `y*` |
| `p*` | Hidden prefix (Experiment 2 only) drawn from the prefix pool |
| `y_i*` | Gold output (Experiment 2 only): `f(p* ⊕ x_i)`, pre-computed once |

## Setup

```bash
uv sync
```

---

## Experiment 1 — GSM8K prefix optimisation

Find `θ` that minimises

```
L(θ) = sum_i E(x_i, f(p(x_i; θ) ⊕ x_i))
```

where `{x_i}` are GSM8K math word problems and `E` is exact match on the final
numeric answer. This is the original Auto-Simas objective.

```bash
python train.py \
  --num-steps 100 \
  --batch-size 8 \
  --warmup-steps 20 \
  --eval-every 10 \
  --eval-size 50
```

---

## Experiment 2 — Hidden-prefix recovery

Instead of a labelled dataset, we have a hidden prefix `p*` and a pool of 50
unlabelled instructions `x_1 … x_N`. The frozen LLM produces gold outputs
`y_i* = f(p* ⊕ x_i)`, pre-computed once at startup. The T5 prefix model sees
only the instructions and must infer a prefix that reproduces those gold outputs.

Find `θ` that minimises

```
L(θ) = -sum_i  E( f(p(x_i; θ) ⊕ x_i),  y_i* )
```

where `E` is cosine similarity between T5 encoder embeddings, scaled to [0, 100].

```bash
python train2.py \
  --prefix-id 0 \
  --num-steps 200 \
  --batch-size 8 \
  --warmup-steps 20 \
  --eval-every 20 \
  --eval-size 20
```

`--prefix-id` selects the hidden prefix from [prefix_pool.py](prefix_pool.py)
(0–19). To sweep over all prefixes, run the script once per index.

### Instructions x_1 … x_N

We use a fixed pool of 50 short instructions that span five task categories. The
inclusion criterion is that a persona or style prefix must produce noticeably
different outputs for that instruction, and the instruction must not have a single
correct answer (otherwise the judge reduces to exact match). The five categories
are:

1. **Factual Q&A** — questions with real answers but many valid phrasings, so the
   prefix's tone and depth change the response meaningfully.
2. **Explanation** — "Explain X" prompts where style (formal vs casual) matters.
3. **Creative writing** — poems, short stories, or metaphors where persona drives
   the output.
4. **Code** — "Write a function that …" prompts where the prefix controls whether
   code is commented, which idioms are chosen, etc.
5. **Advice / opinion** — questions where the prefix's persona shapes the framing.

See [instructions.py](instructions.py) for the full list.

### Judge E

We use cosine similarity between T5-small encoder embeddings (mean-pooled over
non-padding tokens), scaled from [-1, 1] to [0, 100]. See [judge.py](judge.py).

We use T5's encoder rather than a dedicated sentence-similarity model because
T5-small is already cached locally so training works fully offline. We use it
over Claude Haiku as judge because each training step already fires `batch_size`
LLM calls for the predicted outputs, and adding `batch_size` more Haiku calls
would double latency and cost without meaningful gain for REINFORCE.

### Prefix pool

[prefix_pool.py](prefix_pool.py) contains 20 curated persona/style prefixes.
Examples:

- "You are a patient elementary school teacher. Use simple language and short examples."
- "You are a rigorous scientist. Be precise, cite reasoning, and avoid oversimplification."
- "You are a creative writer. Use vivid metaphors and evocative language."
- "You are a software engineer. Explain concepts with working code snippets."

We chose persona prefixes rather than format prefixes (e.g., "respond in bullet
points") because persona prefixes produce the largest semantic shift in `f`'s
output, which gives the judge and REINFORCE the clearest learning signal.

Gold outputs for a chosen `p*` are cached to `gold_cache/gold_prefix{id}.json`
on first run and reloaded on subsequent runs, so training steps never fire extra
LLM calls for the gold side.

### Comparison

| | Experiment 1 | Experiment 2 |
|--|--|--|
| Dataset | GSM8K (labelled Q&A) | Fixed instruction pool (unlabelled) |
| Gold signal | Exact numeric match | T5 encoder cosine similarity |
| Judge calls per step | 0 (local regex) | 0 (local T5 encoder) |
| LLM calls per step | `batch_size` | `batch_size` (gold pre-computed once) |
| Training target | High accuracy on benchmark | Reproduce outputs of a hidden prefix |
