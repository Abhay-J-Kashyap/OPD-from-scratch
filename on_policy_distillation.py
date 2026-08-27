"""
Student: Qwen2.5-0.5B-Instruct   Teacher: Qwen2.5-1.5B-Instruct   Task: GSM8K
Same tokenizer family on purpose -- cross-tokenizer OPD is a separate project.

The loop:
  1. student samples a completion for a prompt          (no grad)
  2. teacher scores the student's own tokens            (no grad, forward only)
  3. per-token truncated reverse KL(student || teacher) (grad)
  4. backprop

There is no REINFORCE and no credit assignment here. The discount factor is
zero, so supervision is local to each token and the loss is differentiable
directly. That is the whole reason OPD is cheaper than RL.

RUN THIS IN A FRESH RUNTIME. A device-side assert poisons the CUDA context and
every later GPU call in the session fails regardless of what you fix.

Setup:
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    !pip install -q transformers datasets accelerate matplotlib bitsandbytes
"""

import json
import os
import re
import random

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- config

STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHER_ID = "Qwen/Qwen2.5-1.5B-Instruct"

STEPS = 150          # small batch -> noisier steps, so run more of them
BATCH = 2            # prompts per step
MAX_NEW = 192        # completion cap
TOPK = 64            # truncate reverse KL to the teacher's top-K support
LR = 1e-5
TEMP = 1.0           # sample at 1.0. Low temperature collapses the on-policy
                     # distribution and you end up doing off-policy SFT.
TOP_K_SAMPLE = 50    # sampling guard against the noisy logit tail
TOP_P_SAMPLE = 0.95
EVAL_EVERY = 50
EVAL_N = 100
EVAL_BATCH = 8       # eval is no-grad, so it can use a bigger batch than train
SEED = 0

DEVICE = "cuda"

# DTYPES -- this is the fix for the device-side assert.
#
# T4 is Turing: no bf16. Qwen2.5-0.5B in fp16 overflows: logits go inf/NaN,
# softmax yields invalid probabilities, torch.multinomial asserts. Greedy
# decoding hides this (argmax over garbage still returns an index), which is why
# eval "worked" while sampling crashed.
#
# Student in fp32: it is sampled from and backpropped through.
# Teacher in fp16: forward passes only, never feeds multinomial. Half the memory
# for no sampling risk -- but it CAN still emit inf, so opd_loss sanitizes it.
S_DTYPE = torch.float32
T_DTYPE = torch.float16

SYS = "Solve the math problem. Reason step by step, then give the final answer as: #### <number>"

random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------- data

ds = load_dataset("openai/gsm8k", "main")
train_q = [ex["question"] for ex in ds["train"]]
test = [(ex["question"], ex["answer"]) for ex in ds["test"]][:EVAL_N]


def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def pred(text):
    m = re.findall(r"####\s*(-?[\d,]+)", text)
    if m:
        return m[-1].strip().replace(",", "")
    m = re.findall(r"(-?[\d,]+\.?\d*)", text)  # fallback: last number
    return m[-1].strip().replace(",", "") if m else None


# ---------------------------------------------------------------- models

tok = AutoTokenizer.from_pretrained(STUDENT_ID, padding_side="left")
student = AutoModelForCausalLM.from_pretrained(
    STUDENT_ID, torch_dtype=S_DTYPE, attn_implementation="eager").to(DEVICE)
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_ID, torch_dtype=T_DTYPE, attn_implementation="eager").to(DEVICE)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

assert student.config.vocab_size == teacher.config.vocab_size, "tokenizer mismatch"

PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
# If PAD == EOS, generated EOS tokens get masked out of the loss. Qwen2.5 sets
# them separately (<|endoftext|> vs <|im_end|>), so this should hold.
assert PAD != tok.eos_token_id, "pad == eos: completion mask will drop EOS tokens"

# MEMORY. On a 16GB T4 the naive setup lands at ~13.6GB and OOMs on activations:
#   student fp32 weights 2.0 + grads 2.0 + AdamW moments 4.0 + teacher 3.1
# The optimizer is the biggest single line and the easiest to shrink.
student.gradient_checkpointing_enable()   # ~30% slower, large activation saving
student.config.use_cache = False          # incompatible with checkpointing

try:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(student.parameters(), lr=LR)   # 4.0GB -> 1.0GB
    print("using AdamW8bit")
except ImportError:
    opt = torch.optim.AdamW(student.parameters(), lr=LR)
    print("bitsandbytes missing -- falling back to fp32 AdamW, expect OOM")


def prompts(questions):
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]
    return tok(texts, return_tensors="pt", padding=True, truncation=True,
               max_length=512).to(DEVICE)


# ---------------------------------------------------------------- the loss


def opd_loss(student_logits, teacher_logits, completion_mask, k=TOPK):
    """Truncated per-token reverse KL on the student's own tokens.

    ALIGNMENT: logits at position i predict the token at position i+1. Getting
    this shift wrong is the single most common bug in distillation code and it
    fails SILENTLY -- loss goes down, model gets worse. Sanity check below.
    """
    s = student_logits[:, :-1, :]
    # The fp16 teacher can emit inf on long sequences; that propagates straight
    # into a non-finite loss. Clamp rather than let it poison AdamW's moments.
    t = teacher_logits[:, :-1, :].nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)
    m = completion_mask[:, 1:].float()

    # Restrict to the teacher's top-K support, then renormalize both sides over
    # that subset. Keeps memory at [B, L, 64] instead of [B, L, 151936].
    idx = t.topk(k, dim=-1).indices
    ls = F.log_softmax(s.gather(-1, idx).float(), dim=-1)
    lt = F.log_softmax(t.gather(-1, idx).float(), dim=-1)

    kl = (ls.exp() * (ls - lt)).sum(-1)          # [B, L-1]
    return (kl * m).sum() / m.sum().clamp(min=1)


def _sanity_check():
    """Two checks, both cheap, both catch bugs that fail silently otherwise."""
    b = prompts(train_q[:2])
    with torch.no_grad():
        lg = student(**b).logits

    # 1. numerics: if the logits are broken, nothing downstream matters
    assert not torch.isnan(lg).any(), "NaN in student logits"
    assert not torch.isinf(lg).any(), "inf in student logits"
    print(f"logit range ok (max |logit| = {lg.abs().max().item():.1f})")

    # 2. alignment: a model's KL against itself must be zero
    v = opd_loss(lg, lg.clone(), b["attention_mask"]).item()
    assert abs(v) < 1e-4, f"alignment/loss bug: self-KL = {v}"
    print(f"sanity check passed (self-KL = {v:.2e})")


# ---------------------------------------------------------------- rollout


@torch.no_grad()
def rollout(questions):
    """Sample from the CURRENT student policy. This is the 'on-policy' part."""
    student.eval()
    student.config.use_cache = True     # KV cache is a big speedup for generate
    b = prompts(questions)
    plen = b["input_ids"].shape[1]
    out = student.generate(
        **b,
        max_new_tokens=MAX_NEW,
        do_sample=True,
        temperature=TEMP,
        top_p=TOP_P_SAMPLE,
        top_k=TOP_K_SAMPLE,
        pad_token_id=PAD,
    )
    student.config.use_cache = False    # back off for checkpointed training
    student.train()

    ids = out
    attn = (ids != PAD).long()
    attn[:, :plen] = b["attention_mask"]
    # supervise only the tokens the student generated, not the prompt
    comp = torch.zeros_like(attn)
    comp[:, plen:] = attn[:, plen:]
    return ids, attn, comp


# ---------------------------------------------------------------- eval


@torch.no_grad()
def evaluate():
    student.eval()
    student.config.use_cache = True
    correct = 0
    lengths = []
    for i in range(0, len(test), EVAL_BATCH):
        chunk = test[i:i + EVAL_BATCH]
        b = prompts([q for q, _ in chunk])
        out = student.generate(
            **b, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=PAD,
        )
        gen = out[:, b["input_ids"].shape[1]:]
        for row, (_, a) in zip(gen, chunk):
            text = tok.decode(row, skip_special_tokens=True)
            # count REAL tokens, not batch padding. len(row) measures the
            # longest sequence in the batch and makes the metric meaningless.
            lengths.append(int((row != PAD).sum()))
            if pred(text) == gold(a):
                correct += 1
    student.config.use_cache = False
    student.train()
    torch.cuda.empty_cache()
    # Runaway length is a documented OPD failure mode, and a length plot is
    # worth more on a resume than an accuracy plot alone.
    return correct / len(test), sum(lengths) / len(lengths)


# ---------------------------------------------------------------- train

_sanity_check()

log = {"step": [], "acc": [], "len": [], "loss": []}
acc, mean_len = evaluate()
log["step"].append(0); log["acc"].append(acc); log["len"].append(mean_len)
print(f"step 0  |  acc {acc:.3f}  len {mean_len:.0f}")

student.train()
skipped = 0
for step in range(1, STEPS + 1):
    qs = random.sample(train_q, BATCH)
    ids, attn, comp = rollout(qs)

    with torch.no_grad():
        t_logits = teacher(input_ids=ids, attention_mask=attn).logits
    s_logits = student(input_ids=ids, attention_mask=attn).logits

    loss = opd_loss(s_logits, t_logits, comp)

    # A single NaN step corrupts every weight via AdamW's moments. Skip and
    # count instead -- if skips climb, the run is not trustworthy.
    if not torch.isfinite(loss):
        skipped += 1
        print(f"step {step}: non-finite loss, skipped ({skipped} total)")
        opt.zero_grad(set_to_none=True)
        del s_logits, t_logits
        torch.cuda.empty_cache()
        continue

    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

    log["loss"].append(loss.item())
    del s_logits, t_logits
    if step % 10 == 0:
        torch.cuda.empty_cache()

    if step % EVAL_EVERY == 0:
        acc, mean_len = evaluate()
        log["step"].append(step); log["acc"].append(acc); log["len"].append(mean_len)
        print(f"step {step}  |  KL {loss.item():.4f}  acc {acc:.3f}  len {mean_len:.0f}")
        student.save_pretrained("/content/opd_ckpt")   # Colab disconnects. Save.
        tok.save_pretrained("/content/opd_ckpt")

log["skipped_steps"] = skipped
json.dump(log, open("opd_log.json", "w"))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(log["step"], log["acc"], marker="o")
ax[0].set_xlabel("OPD step"); ax[0].set_ylabel("GSM8K accuracy")
ax[1].plot(log["step"], log["len"], marker="o", color="crimson")
ax[1].set_xlabel("OPD step"); ax[1].set_ylabel("mean completion length")
plt.tight_layout(); plt.savefig("opd_results.png", dpi=150)
print(f"done ({skipped}/{STEPS} steps skipped) ->  opd_results.png,  opd_log.json")
