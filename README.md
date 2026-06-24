# Auto-Simas — Hidden-Prefix Recovery

A frozen large LLM `f: text → text` takes a context and produces text. We
have a hidden prefix `p*` and a pool of 50 unlabelled instructions
`x_1 … x_N`. The frozen LLM produces gold outputs `y_i* = f(p* ⊕ x_i)`,
pre-computed once at startup. A T5-small prefix model `p(·; θ)` sees only the
instructions and must infer a prefix that reproduces those gold outputs.

## Notation

| Symbol | Meaning |
|--------|---------|
| `f` | Frozen large LLM (Claude via Bedrock) |
| `p*` | Hidden prefix drawn from the prefix pool |
| `x_1 … x_N` | Instruction pool (fixed, diverse, prefix-sensitive) |
| `y_i*` | Gold output `f(p* ⊕ x_i)`, pre-computed once at startup |
| `p(·; θ)` | T5-small prefix model (trained) |
| `E(y, y*)` | Judge: T5 encoder cosine similarity, scaled to [0, 100] |

## Objective

Find `θ` that minimises

```
L(θ) = -sum_i  E( f(p(x_i; θ) ⊕ x_i),  y_i* )
```

Because the judge is non-differentiable, we optimise with REINFORCE. The reward
for a batch step is the mean judge score over the sampled instructions.

## Setup

```bash
uv sync
```

## Run

```bash
python train.py \
  --prefix-id 0 \
  --num-steps 200 \
  --batch-size 8 \
  --warmup-steps 3 \
  --eval-every 20 \
  --eval-size 20
```

`--prefix-id` selects the hidden prefix from [prefix_pool.py](prefix_pool.py)
(0–19). To sweep over all prefixes, run the script once per index.

## Design choices

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

### Warm-start

The warm-start trains T5 for a few steps to output `"you are a ..."` for every
instruction. This gives REINFORCE a starting point in the right region of output
space (short persona-style prefix) without handing it the answer.
