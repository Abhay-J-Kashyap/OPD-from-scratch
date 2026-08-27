# On-Policy Distillation reimplementation from scratch

A from-scratch implementation of on-policy distillation (OPD): distilling
Qwen2.5-1.5B-Instruct into Qwen2.5-0.5B-Instruct on GSM8K, trained end-to-end
on Colab T4.

The loss is written directly in PyTorch rather than configured through a
trainer, since the point of the project was to understand the objective.

## What on-policy distillation is

Standard distillation is off-policy: the student trains on text the teacher
generated. The student is then evaluated on text *it* generates, so errors
compound on trajectories it was never trained on.

On-policy distillation closes that gap. The student generates, and the teacher
grades those same tokens:

1. Sample a completion from the **current** student policy (no grad).
2. Run one teacher forward pass over the student's tokens (no grad).
3. Compute per-token reverse KL between student and teacher at every generated
   position.
4. Backprop.

The discount factor is zero, so supervision is local to each token and the loss
is differentiable directly — there is no REINFORCE term and no credit
assignment. That is why OPD is substantially cheaper than outcome-reward RL: a
dense signal at every token instead of one scalar at the end, and the teacher
only ever does forward passes.

## Implementation notes

**Truncated reverse KL.** Full-vocabulary KL over Qwen's 151k tokens is
memory-prohibitive on a T4. The loss restricts to the teacher's top-64 tokens
and renormalizes both distributions over that support — the standard
truncation, and also a documented stability fix.

**Logit alignment.** Logits at position `i` predict the token at position
`i+1`. Getting this shift wrong fails *silently*: loss decreases while the model
degrades. `_sanity_check()` asserts that a model's KL against itself is ~0
before training starts.

**Sampling temperature.** Rollouts are sampled at T=1.0. Lowering it collapses
the on-policy distribution and quietly turns the run back into off-policy SFT.

**Loss masking.** Only student-generated tokens are supervised; prompt tokens
are masked out.

## Results

| | GSM8K (greedy, 100 test problems) |
|---|---|
| Student, before training | _TBD_ |
| Student, after 150 OPD steps | _TBD_ |
| Teacher (ceiling) | _TBD_ |

![results](opd_results.png)

Completion length is logged alongside accuracy. Runaway length growth is a
known OPD failure mode driven by repetition, and it does not show up in an
accuracy curve alone.

## Running it

```bash
pip install transformers datasets accelerate matplotlib
python on_policy_distillation.py
```

Roughly 45 minutes for 150 steps on a T4. If you hit OOM, reduce `BATCH` to 4,
then `MAX_NEW` to 192.

## Scope and limitations

This is a small-scale implementation, not a research result. A 0.5B student and
150 steps are enough to demonstrate a working loop, not to make claims about the
method. Specifically not addressed:

- No compute-matched comparison against SFT or GRPO baselines
- Single seed, no error bars
- Same-tokenizer pair only; cross-tokenizer OPD is a separate problem
- Short rollout cap (256 tokens) truncates some chains of thought

## References

- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) — Thinking Machines Lab, 2025
- [GKD: On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649) — Agarwal et al., 2023
- [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) — Gu et al., 2023
- [awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation) — literature index
