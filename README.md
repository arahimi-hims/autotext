# Auto-Simas

A big frozen LLM is $f:\mathrm{text}\to \mathrm{text}$: it takes as input a
context and produces some text as output. An evaluator is $E:\mathrm{text}
\times \mathrm{text} \to \mathbb{R}$: It takes as input the output of an LLM and
the original context, and it produces a score. We typically examine the input $x$ of the LLM and produce a prefix $p(x): \mathrm{text} \to \mathrm{text}$ and feed the LLM the context $p(x) \oplus x$.
Our job is to build a good prefix function $p$.

To do this, we typically have a dataset
${x_i}_{i=1}^n$ of benchmark inputs.
Simas's job is to find a prompt prefix $p$ that minimizes $\mathcal{L}(p) \equiv \sum_i E(x_i, f(p(x_i) \oplus x_i))$.

Simas tunes $p$ by hand. Auto-Simas represents $p(x;\theta)$ with a very small Transformer model.
