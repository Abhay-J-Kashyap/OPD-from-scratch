"""
On-policy distillation (OPD) on a single free-tier Colab GPU (T4, 16GB).

Student: Qwen2.5-0.5B-Instruct   Teacher: Qwen2.5-1.5B-Instruct   Task: GSM8K

The loop:
  1. student samples a completion for a prompt          (no grad)
  2. teacher scores the student's own tokens            (no grad, forward only)
  3. per-token truncated reverse KL(student || teacher) (grad)
  4. backprop

There is no REINFORCE and no credit assignment. The discount factor is zero, so
supervision is local to each token and the loss is differentiable directly.

RUN IN A FRESH RUNTIME.

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

STEPS = 150
BATCH = 2
MAX_NEW = 192
TOPK = 64
LR = 1e-6            # was 1e-5. Batch 2 gives noisy gradients; 1e-5 on top of
                     # that walked the model off a cliff by step 50.
TEMP = 1.0
TOP_K_SAMPLE = 50
TOP_P_SAMPLE = 0.95
REP_PENALTY = 1.15   # discourages rollouts degenerating into repeats
SKIP_GRAD_NORM = 100.0   # was 10.0 — normal range here is 20-32
GRAD_CLIP = 1.0          # was 0.5 — clipping to 0.5 against a norm of ~25. shrinks every update 50x, which would train nothing
EVAL_EVERY = 10      # tighter grid: catch collapse at 25, not at 50
EVAL_N = 100
EVAL_BATCH = 8
SEED = 0

DEVICE = "cuda"
S_DTYPE = torch.float32   # student: sampled from + backpropped through
T_DTYPE = torch.float16   # teacher: forward passes only

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
    m = re.findall(r"(-?[\d,]+\.?\d*)", text)
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
assert PAD != tok.eos_token_id, "pad == eos: completion mask will drop EOS tokens"

# ---- FIX 1: freeze the embedding matrix.
#
# Qwen2.5-0.5B ties input embeddings to the output head, so every gradient into
# the lm_head writes straight into the token embedding table. Corrupt that and
# the model loses its token->vector mapping entirely -- the symptom is output
# that dumps vocabulary in sorted order, which is what happened on the last run.
#
# The transformer body has ample capacity to fit the teacher without touching
# the embedding. Freezing it also removes the largest single parameter block
# from the optimizer, which helps memory.
if student.config.tie_word_embeddings:
    student.get_input_embeddings().weight.requires_grad_(False)
    print("tied embeddings detected -> frozen")

# ---- FIX 2: non-reentrant checkpointing.
#
# The default (reentrant) implementation can silently produce WRONG gradients.
# Wrong gradients into a tied embedding is a fast route to a destroyed model.
student.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})
student.config.use_cache = False
student.enable_input_require_grads()   # needed for non-reentrant checkpointing

trainable = [p for p in student.parameters() if p.requires_grad]
print(f"trainable params: {sum(p.numel() for p in trainable)/1e6:.0f}M "
      f"of {sum(p.numel() for p in student.parameters())/1e6:.0f}M")

try:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(trainable, lr=LR, weight_decay=0.0)
    print("using AdamW8bit")
except ImportError:
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
    print("bitsandbytes missing -- using fp32 AdamW")


def prompts(questions):
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    return tok(texts, return_tensors="pt", padding=True, truncation=True,
               max_length=512).to(DEVICE)


# ---------------------------------------------------------------- the loss


def opd_loss(student_logits, teacher_logits, completion_mask, k=TOPK):
    """Truncated per-token reverse KL on the student's own tokens.

    ALIGNMENT: logits at position i predict the token at position i+1.
    """
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :].nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)
    m = completion_mask[:, 1:].float()

    idx = t.topk(k, dim=-1).indices
    ls = F.log_softmax(s.gather(-1, idx).float(), dim=-1)
    lt = F.log_softmax(t.gather(-1, idx).float(), dim=-1)

    kl = (ls.exp() * (ls - lt)).sum(-1)
    return (kl * m).sum() / m.sum().clamp(min=1)


def _sanity_check():
    b = prompts(train_q[:2])
    with torch.no_grad():
        lg = student(**b).logits
    assert not torch.isnan(lg).any(), "NaN in student logits"
    assert not torch.isinf(lg).any(), "inf in student logits"
    print(f"logit range ok (max |logit| = {lg.abs().max().item():.1f})")
    v = opd_loss(lg, lg.clone(), b["attention_mask"]).item()
    assert abs(v) < 1e-4, f"alignment/loss bug: self-KL = {v}"
    print(f"sanity check passed (self-KL = {v:.2e})")


# ---------------------------------------------------------------- rollout


@torch.no_grad()
def rollout(questions):
    student.eval()
    student.config.use_cache = True
    b = prompts(questions)
    plen = b["input_ids"].shape[1]
    out = student.generate(
        **b, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP,
        top_p=TOP_P_SAMPLE, top_k=TOP_K_SAMPLE,
        repetition_penalty=REP_PENALTY, pad_token_id=PAD,
    )
    student.config.use_cache = False
    student.train()

    ids = out
    attn = (ids != PAD).long()
    attn[:, :plen] = b["attention_mask"]
    comp = torch.zeros_like(attn)
    comp[:, plen:] = attn[:, plen:]
    return ids, attn, comp


# ---------------------------------------------------------------- eval


@torch.no_grad()
def sample_output():
    """Print one greedy completion. A number tells you the model got worse;
    this tells you HOW -- repetition loop, gibberish, or fluent-but-wrong are
    three different problems with three different fixes."""
    student.eval(); student.config.use_cache = True
    b = prompts(train_q[:1])
    out = student.generate(**b, max_new_tokens=80, do_sample=False, pad_token_id=PAD)
    txt = tok.decode(out[0, b["input_ids"].shape[1]:], skip_special_tokens=True)
    student.config.use_cache = False; student.train()
    return txt.replace("\n", " ")[:160]


@torch.no_grad()
def evaluate():
    student.eval()
    student.config.use_cache = True
    correct, lengths = 0, []
    for i in range(0, len(test), EVAL_BATCH):
        chunk = test[i:i + EVAL_BATCH]
        b = prompts([q for q, _ in chunk])
        out = student.generate(**b, max_new_tokens=MAX_NEW, do_sample=False,
                               pad_token_id=PAD)
        gen = out[:, b["input_ids"].shape[1]:]
        for row, (_, a) in zip(gen, chunk):
            text = tok.decode(row, skip_special_tokens=True)
            lengths.append(int((row != PAD).sum()))   # real tokens, not padding
            if pred(text) == gold(a):
                correct += 1
    student.config.use_cache = False
    student.train()
    torch.cuda.empty_cache()
    return correct / len(test), sum(lengths) / len(lengths)


# ---------------------------------------------------------------- train

_sanity_check()

log = {"step": [], "acc": [], "len": [], "loss": [], "gnorm": []}
acc, mean_len = evaluate()
log["step"].append(0); log["acc"].append(acc); log["len"].append(mean_len)
print(f"step 0  |  acc {acc:.3f}  len {mean_len:.0f}")
print(f"  sample: {sample_output()}\n")

student.train()
skipped = 0
for step in range(1, STEPS + 1):
    qs = random.sample(train_q, BATCH)
    ids, attn, comp = rollout(qs)

    with torch.no_grad():
        t_logits = teacher(input_ids=ids, attention_mask=attn).logits
    s_logits = student(input_ids=ids, attention_mask=attn).logits
    loss = opd_loss(s_logits, t_logits, comp)

    if not torch.isfinite(loss):
        skipped += 1
        opt.zero_grad(set_to_none=True)
        del s_logits, t_logits; torch.cuda.empty_cache()
        continue

    loss.backward()

    # ---- FIX 3: skip on gradient spike, and LOG the norm.
    # A single outsized update is what destroys a model. Clipping alone rescales
    # a bad direction; it does not reject it. Logging the norm means the next
    # failure is diagnosable from the curve instead of from guesswork.
    gnorm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP).item()
    if gnorm > SKIP_GRAD_NORM:
        skipped += 1
        print(f"step {step}: grad norm {gnorm:.1f} > {SKIP_GRAD_NORM}, skipped")
        opt.zero_grad(set_to_none=True)
        del s_logits, t_logits; torch.cuda.empty_cache()
        continue

    opt.step()
    opt.zero_grad(set_to_none=True)

    log["loss"].append(loss.item()); log["gnorm"].append(gnorm)
    del s_logits, t_logits
    if step % 10 == 0:
        torch.cuda.empty_cache()

    if step % EVAL_EVERY == 0:
        acc, mean_len = evaluate()
        log["step"].append(step); log["acc"].append(acc); log["len"].append(mean_len)
        print(f"step {step}  |  KL {loss.item():.4f}  |g| {gnorm:.2f}  "
              f"acc {acc:.3f}  len {mean_len:.0f}")
        print(f"  sample: {sample_output()}\n")
        student.save_pretrained(f"/content/opd_ckpt_{step}")
        tok.save_pretrained(f"/content/opd_ckpt_{step}")   # keep per-step
                                                            # checkpoints so a
                                                            # collapse doesn't
                                                            # overwrite a good one

log["skipped_steps"] = skipped
json.dump(log, open("opd_log.json", "w"))

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(log["step"], log["acc"], marker="o")
ax[0].set_xlabel("OPD step"); ax[0].set_ylabel("GSM8K accuracy")
ax[1].plot(log["step"], log["len"], marker="o", color="crimson")
ax[1].axhline(MAX_NEW, ls="--", c="gray", lw=1)
ax[1].set_xlabel("OPD step"); ax[1].set_ylabel("mean completion length")
ax[2].plot(log["gnorm"], lw=0.8, color="darkgreen")
ax[2].set_xlabel("OPD step"); ax[2].set_ylabel("grad norm (pre-clip)")
plt.tight_layout(); plt.savefig("opd_results.png", dpi=150)
print(f"done ({skipped}/{STEPS} skipped) -> opd_results.png, opd_log.json")
