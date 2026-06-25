# Auto-Simas — Hidden-Prefix Recovery

This is an as-built record of a running experiment. The system trains a T5-small
prefix generator to recover a hidden persona prefix by imitating the frozen LLM's
behaviour under that prefix.

## Problem

A frozen large LLM `f: text → text` (Claude Haiku via Bedrock) is conditioned on
a hidden prefix `p*` drawn from a pool of 20 persona strings. We observe its
outputs `y_i* = f(p* ⊕ x_i)` on a pool of 50 instructions `x_1 … x_N`,
pre-computed once at startup. A T5-small prefix model `p(·; θ)` sees only the
instructions and must produce a prefix that causes `f` to output text similar to
those gold outputs.

| Symbol | Meaning |
|--------|---------|
| `f` | Frozen large LLM (Claude Haiku via Bedrock) |
| `p*` | Hidden prefix drawn from the prefix pool |
| `x_1 … x_N` | Instruction pool (50 diverse, prefix-sensitive instructions) |
| `y_i*` | Gold output `f(p* ⊕ x_i)`, pre-computed once |
| `p(x; θ)` | T5-small seq2seq prefix model (trained) |
| `ŷ_i` | Estimated output `f(p(x_i; θ) ⊕ x_i)` |
| `p_est_i` | Prefix inferred by judge E from `(x_i, ŷ_i, y_i*)` |
| `E` | LM judge: same model class as `f`, prompted to infer the prefix |

## Algorithm

Each training step does four things:

1. **Decode** the current prefix greedily: `p_i = p(x_i; θ)` for each
   instruction in the batch.
2. **Generate** the estimated output: `ŷ_i = f(p_i ⊕ x_i)` for each example,
   via concurrent Bedrock calls.
3. **Query the judge** E with the triple `(x_i, ŷ_i, y_i*)`. E is prompted to
   read the two responses and infer what system prompt produced `y_i*`, based on
   stylistic differences. E returns `p_est_i`.
4. **Update** T5 via teacher-forcing cross-entropy: minimise
   `-log p(p_est_i | x_i; θ)` summed over the batch, then take one Adam step.

The loop runs for 200 steps with batch size 8. Each step makes `2 × 8 = 16`
concurrent Bedrock calls (8 for `ŷ` and 8 for `p_est`) and takes about 7 seconds
on a laptop CPU.

The key idea is that E, being the same model class as `f`, can read style
differences directly from text and name semantic properties ("patient elementary
school teacher", "short sentences"). This gives a richer training signal than the
scalar cosine-similarity reward used in the first experiment, because T5 receives
a target string rather than a number.

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
(0–19).

## Design choices

### Instructions x_1 … x_N

[instructions.py](instructions.py) contains 50 short instructions across five
categories. The inclusion criterion is that a persona prefix must produce
noticeably different outputs for the instruction, and the instruction must not
have a single correct answer (otherwise the judge reduces to exact match). The
five categories are factual Q&A, explanation, creative writing, code, and
advice/opinion.

### Prefix pool

[prefix_pool.py](prefix_pool.py) contains 20 curated persona/style prefixes. We
use persona prefixes rather than format prefixes (for example, "respond in bullet
points") because persona prefixes produce the largest semantic shift in `f`'s
output, which gives E the clearest signal for inferring the prefix.

Gold outputs for a chosen `p*` are cached to `gold_cache/gold_prefix{id}.json`
on first run and reloaded on subsequent runs, so training steps never make extra
LLM calls for the gold side.

### Judge E

E is the same Claude Haiku model used as `f`. For each `(x_i, ŷ_i, y_i*)` triple,
E receives a prompt that presents both responses and asks it to infer the hidden
system prompt from stylistic differences. E is instructed to output only the
system prompt text, with no preamble, in 1–3 sentences. See the `infer_prefix`
function in [train.py](train.py).

We query E per example rather than once per batch so that the inferred `p_est_i`
is conditioned on the specific instruction `x_i`. This means T5 sees diverse
targets across a batch rather than one aggregate guess.

### Eval metric

Each eval checkpoint generates 20 responses using the current greedy-decoded
prefix and scores them with T5-encoder cosine similarity against the gold outputs,
scaled to [0, 100]. See [judge.py](judge.py) for the implementation. The ceiling
for this metric is approximately 98.6/100, which is the score attained when the
model outputs the exact p*.

### Warm-start

Before the main loop, T5 is trained for 3 supervised steps to output
`"you are a ..."` for every instruction. This moves the decoder into the right
region of output space (short persona-style text) before E starts providing guided
targets.

## Results

The runs below all use prefix 0 as the hidden target:

> `You are a patient elementary school teacher. Use simple language, short sentences, and concrete examples that a ten-year-old would understand.`

### LM-judge (current approach)

Eval scores at every checkpoint over the 200-step run:

| Step | Eval score (/ 100) |
|------|--------------------|
| 20   | 95.2               |
| 40   | 96.3               |
| 60   | 96.9               |
| 80   | 97.4               |
| 100  | 97.7               |
| 120  | 97.7               |
| 140  | 97.1               |
| 160  | 97.5               |
| 180  | 97.6               |
| 200  | 97.2               |

Best score: **97.7/100** at steps 100 and 120. Loss fell from 5.8 at step 0 to
approximately 2.2 by step 200. No mode collapse occurred.

T5's greedy-decoded prefix near the end of training:

```
You are a helpful assistant designed to explain concepts to children. Use simple language...
```

T5 recovered the right semantic territory ("helpful assistant, simple language,
children") but did not recover the exact wording of p* ("patient elementary school
teacher", "concrete examples", "ten-year-old"). The 80-character truncation in
the log means the tails of these strings are not shown.

### REINFORCE with cosine-similarity reward (first experiment)

The first experiment used T5-encoder cosine similarity as the REINFORCE reward.
The reward for the exact p* was approximately 98.6/100, but a degenerate prefix
that repeats "you" scored 95.8/100, a gap of only 2.8 points. Because per-step
reward variance was approximately ±1.5, REINFORCE could not reliably exploit this
gap and the model collapsed to the degenerate prefix around step 80–120.

Increasing the sampling temperature to 2.0 prevented collapse but produced
grammatically incoherent prefixes that scored identically (95.1–95.6). The
cosine-similarity metric was simply not discriminating enough to drive learning
from the degenerate region toward the target.

The LM-judge approach avoids this problem because E provides a target string
rather than a scalar reward. Teacher-forcing cross-entropy gives T5 a meaningful
gradient even when the current prefix is far from p*.
