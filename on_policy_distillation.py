import json
import re
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- config

STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHER_ID = "Qwen/Qwen2.5-1.5B-Instruct"

STEPS = 150          # ~45 min on a T4. Drop to 60 for a first smoke test.
BATCH = 8            # prompts per step
MAX_NEW = 256        # completion cap. Lower = faster, but truncates answers.
TOPK = 64            # truncate reverse KL to the teacher's top-K support
LR = 1e-5
TEMP = 1.0           # sample at 1.0. Low temperature collapses the on-policy
                     # distribution and you end up doing off-policy SFT.
EVAL_EVERY = 25
EVAL_N = 100
SEED = 0

DEVICE = "cuda"

DTYPE = torch.float16

SYS = "Solve the math problem. Reason step by step, then give the final answer as: #### <number>"

random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------- data

ds = load_dataset("gsm8k", "main")
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
student = AutoModelForCausalLM.from_pretrained(STUDENT_ID, torch_dtype=DTYPE).to(DEVICE)
teacher = AutoModelForCausalLM.from_pretrained(TEACHER_ID, torch_dtype=DTYPE).to(DEVICE)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

assert student.config.vocab_size == teacher.config.vocab_size, "tokenizer mismatch"

opt = torch.optim.AdamW(student.parameters(), lr=LR)


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
    t = teacher_logits[:, :-1, :]
    m = completion_mask[:, 1:].float()

    # Restrict to the teacher's top-K support, then renormalize both sides over
    # that subset. Keeps memory at [B, L, 64] instead of [B, L, 151936].
    idx = t.topk(k, dim=-1).indices
    ls = F.log_softmax(s.gather(-1, idx).float(), dim=-1)
    lt = F.log_softmax(t.gather(-1, idx).float(), dim=-1)

    kl = (ls.exp() * (ls - lt)).sum(-1)          # [B, L-1]
    return (kl * m).sum() / m.sum().clamp(min=1)


def _sanity_check():
    """KL of a model against itself must be ~0. Run this before you train."""
    b = prompts(train_q[:2])
    with torch.no_grad():
        lg = student(**b).logits
    mask = b["attention_mask"]
    v = opd_loss(lg, lg.clone(), mask).item()
    assert abs(v) < 1e-4, f"alignment/loss bug: self-KL = {v}"
    print(f"sanity check passed (self-KL = {v:.2e})")


# ---------------------------------------------------------------- rollout


@torch.no_grad()
def rollout(questions):
    """Sample from the CURRENT student policy. This is the 'on-policy' part."""
    b = prompts(questions)
    plen = b["input_ids"].shape[1]
    out = student.generate(
        **b,
        max_new_tokens=MAX_NEW,
        do_sample=True,
        temperature=TEMP,
        top_p=1.0,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    ids = out
    attn = (ids != (tok.pad_token_id or tok.eos_token_id)).long()
    attn[:, :plen] = b["attention_mask"]
    # supervise only the tokens the student generated, not the prompt
    comp = torch.zeros_like(attn)
    comp[:, plen:] = attn[:, plen:]
    return ids, attn, comp


# ---------------------------------------------------------------- eval


@torch.no_grad()
def evaluate():
    student.eval()
    correct = 0
    lengths = []
    for i in range(0, len(test), 8):
        chunk = test[i:i + 8]
        b = prompts([q for q, _ in chunk])
        out = student.generate(
            **b, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        gen = out[:, b["input_ids"].shape[1]:]
        for row, (_, a) in zip(gen, chunk):
            text = tok.decode(row, skip_special_tokens=True)
            lengths.append(len(row))
            if pred(text) == gold(a):
                correct += 1
    student.train()
    
    return correct / len(test), sum(lengths) / len(lengths)


# ---------------------------------------------------------------- train

_sanity_check()

log = {"step": [], "acc": [], "len": [], "loss": []}
acc, mean_len = evaluate()
log["step"].append(0); log["acc"].append(acc); log["len"].append(mean_len)
print(f"step 0  |  acc {acc:.3f}  len {mean_len:.0f}")

student.train()
for step in range(1, STEPS + 1):
    qs = random.sample(train_q, BATCH)
    ids, attn, comp = rollout(qs)

    with torch.no_grad():
        t_logits = teacher(input_ids=ids, attention_mask=attn).logits
    s_logits = student(input_ids=ids, attention_mask=attn).logits

    loss = opd_loss(s_logits, t_logits, comp)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

    log["loss"].append(loss.item())
    del s_logits, t_logits
    torch.cuda.empty_cache()

    if step % EVAL_EVERY == 0:
        acc, mean_len = evaluate()
        log["step"].append(step); log["acc"].append(acc); log["len"].append(mean_len)
        print(f"step {step}  |  KL {loss.item():.4f}  acc {acc:.3f}  len {mean_len:.0f}")
        student.save_pretrained("/content/opd_ckpt")   # Colab disconnects. Save.

json.dump(log, open("opd_log.json", "w"))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(log["step"], log["acc"], marker="o")
ax[0].set_xlabel("OPD step"); ax[0].set_ylabel("GSM8K accuracy")
ax[1].plot(log["step"], log["len"], marker="o", color="crimson")
ax[1].set_xlabel("OPD step"); ax[1].set_ylabel("mean completion length")
plt.tight_layout(); plt.savefig("opd_results.png", dpi=150)
print("done ->  opd_results.png,  opd_log.json")
