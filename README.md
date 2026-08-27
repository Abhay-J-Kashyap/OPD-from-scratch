# On-Policy Distillation on a Single Free-Tier GPU

A from-scratch implementation of on-policy distillation (OPD), distilling
Qwen2.5-1.5B-Instruct into Qwen2.5-0.5B-Instruct on GSM8K, trained end-to-end on
one free-tier Colab T4.

The loss is written directly in PyTorch rather than configured through a trainer,
because the point of the project was to understand the objective and its failure
modes rather than to get a number.

**Headline result: the student's pass@1 stayed flat while its pass@4 fell by
three-quarters.** The model did not get better — it got narrower. That is only
visible because the eval harness measured more than accuracy.

## What on-policy distillation is

Standard distillation is off-policy: the student trains on text the teacher
generated, then gets evaluated on text *it* generates, so errors compound on
trajectories it never saw in training.

On-policy distillation closes that gap — the student generates, the teacher
grades those same tokens:

1. Sample a completion from the **current** student policy (no grad).
2. One teacher forward pass over the student's tokens (no grad).
3. Per-token reverse KL between student and teacher at every generated position.
4. Backprop.

The discount factor is zero, so supervision is local to each token and the loss
is differentiable directly — no REINFORCE term, no credit assignment. That is
why OPD is cheaper than outcome-reward RL: a dense signal at every token instead
of one scalar at the end, and the teacher only ever does forward passes.

## Results

Student trained 20 steps (batch 2, LR 1e-6). Checkpoint selected before the
collapse described below. 100 test problems, greedy decoding, single seed.

| Metric | Base | Distilled | Teacher |
|---|---|---|---|
| GSM8K pass@1 | 0.090 | 0.110 | 0.170 |
| SVAMP pass@1 (out-of-domain) | 0.400 | 0.350 | 0.570 |
| **GSM8K pass@4** | **0.180** | **0.040** | – |
| Format compliance | 0.000 | 0.010 | – |
| Mean completion length | 189 | 168 | – |

**The accuracy rows are noise.** At n=100 the 95% interval on 0.11 is roughly
±6 points, so both the GSM8K gain and the SVAMP drop overlap the baseline. Do
not read either as a result.

**The pass@4 row is not noise.** A 14-point drop is well outside that interval,
and it moves opposite to pass@1. Twenty steps of reverse-KL pressure toward the
teacher measurably collapsed the sampling distribution: the student lost the
ability to reach correct answers through diverse sampling without gaining
capability in exchange. This matches the documented finding that on-policy
self-distillation reduces output diversity relative to on-policy RL.

A single pass@1 number would have read as "small improvement." The eval harness
is the only reason the opposite conclusion is visible.

## Training instability

Training collapsed reproducibly into a degenerate state: KL decaying toward 0,
completion length saturating at the generation cap, accuracy → 0, and output
degenerating into repeated tokens or symbol soup.

| Learning rate | Collapse onset |
|---|---|
| 1e-5 | ~step 25 |
| 1e-6 | ~step 40 |

Lowering the learning rate delayed the collapse but did not prevent it, which
suggests the instability is structural to this small-model setup rather than
purely a step-size problem. The reported checkpoint is step 20, before drift.

Grad norms at collapse were *calm* (11.4, well under the clip threshold), so this
was not one violent update — the model degraded gradually over ~10 steps.

## Failure modes hit and fixed

Recording these because they cost most of the project's time and none of them
announce themselves clearly:

**fp16 overflow on Turing.** T4 has no bf16. Qwen2.5-0.5B in fp16 produced
inf/NaN logits; `torch.multinomial` then threw a device-side CUDA assert. Greedy
decoding *hid* this — argmax over garbage still returns an index — so eval
"worked" while sampling crashed. Fixed by running the student in fp32 (it is the
model being sampled from and backpropped through) and leaving the teacher in
fp16 (forward passes only, never feeds multinomial).

**Destroyed tied embeddings.** Qwen2.5-0.5B ties input embeddings to the output
head, so every gradient into the lm_head writes into the token embedding table.
Corrupting it produced output that dumped vocabulary in sorted order. Fixed by
freezing the tied embedding — the transformer body has ample capacity to fit the
teacher without it, and it removes the largest parameter block from the
optimizer.

**Reentrant gradient checkpointing.** The default implementation can silently
produce incorrect gradients. Switched to `use_reentrant=False` with
`enable_input_require_grads()`.

**Padding measured as length.** `len(row)` on a generate() output measures the
longest sequence in the batch, not the model's actual output, making the
length metric meaningless. Fixed by counting non-pad tokens.

**A miscalibrated guard.** An initial grad-norm skip threshold of 10.0 rejected
100% of steps — the normal range for this setup turned out to be 20–32. Logging
the distribution before setting a threshold would have caught this immediately.

## Implementation notes

**Truncated reverse KL.** Full-vocabulary KL over Qwen's 151k tokens is
memory-prohibitive on a T4, so the loss restricts to the teacher's top-64 tokens
and renormalizes both distributions over that support.

**Logit alignment.** Logits at position `i` predict the token at `i+1`. Getting
this shift wrong fails *silently* — loss decreases while the model degrades.
`_sanity_check()` asserts a model's KL against itself is ~0 before training.

**Sampling temperature.** Rollouts sampled at T=1.0. Lowering it collapses the
on-policy distribution and quietly turns the run into off-policy SFT.

**Memory.** Student fp32 + gradients + Adam moments + teacher fp16 lands around
13.6GB on a 16GB card and OOMs on activations. Fixed with 8-bit Adam
(4.0GB → 1.0GB), gradient checkpointing, batch size 2, and the frozen embedding.

## Running it

```bash
pip install transformers datasets accelerate matplotlib bitsandbytes
python on_policy_distillation.py   # ~40 min on a T4
python evals.py                     # restart the runtime first
```

Restart the runtime between scripts — each loads its own models and they will not
co-exist in 16GB.

## Limitations

This is a small-scale implementation, not a research result:

- Single seed, no error bars. Every number here needs them.
- No compute-matched comparison against SFT or GRPO baselines.
- 20 training steps at batch 2 is ~40 examples. Far too few to conclude anything
  about the method itself — only about this setup.
- The teacher scores 0.17 on GSM8K here, so the capability ceiling was low to
  begin with; a weak teacher gives weak signal.
- Format compliance is ~0 across all models, meaning every score came from a
  fallback "last number in the text" regex rather than the requested `####`
  format. These are "found the right number somewhere" rates.
- Same-tokenizer pair only. Cross-tokenizer OPD is a separate problem.

## References

- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) — Thinking Machines Lab, 2025
- [GKD: On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649) — Agarwal et al., 2023
- [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) — Gu et al., 2023
- [awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation) — literature index
