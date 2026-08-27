"""
Eval harness for the on-policy distillation project.

Runs a small suite comparing the base student against the distilled checkpoint.
A single accuracy number can't distinguish real improvement from overfitting to
the training distribution, so this measures five things instead:

  1. pass@1, in-domain (GSM8K)      -- did it get better at the trained task
  2. pass@1, out-of-domain (SVAMP)  -- did it generalize, or just memorize
  3. pass@4 vs pass@1               -- did distillation collapse output diversity
  4. format compliance              -- did it keep following the output spec
  5. mean completion length         -- length inflation is a known OPD failure

(3) is the one that matters most here. On-policy self-distillation is documented
to reduce output diversity relative to on-policy RL, so a model can gain pass@1
while LOSING pass@4 -- it becomes more confident, not more capable. If you only
report pass@1 you will not see that happen.

Usage:  python evals.py
"""

import json
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CKPT = "/content/opd_ckpt_20"
TEACHER_ID = "Qwen/Qwen2.5-1.5B-Instruct"

N_GREEDY = 100     # problems for pass@1
N_PASSK = 50       # problems for pass@k (k x more generations -- keep it small)
K = 4
MAX_NEW = 192
BATCH = 8
DEVICE, DTYPE = "cuda", torch.float16

SYS = "Solve the math problem. Reason step by step, then give the final answer as: #### <number>"

# ---------------------------------------------------------------- data

gsm = load_dataset("openai/gsm8k", "main")["test"]
IN_DOMAIN = [(e["question"], e["answer"].split("####")[-1].strip().replace(",", ""))
             for e in gsm][:max(N_GREEDY, N_PASSK)]

# Out-of-domain: same task type, different source and phrasing. If this repo id
# has moved, fall back to a disjoint GSM8K slice -- weaker (same distribution)
# but still held out. Say which one you used in the README.
try:
    sv = load_dataset("ChilleD/SVAMP")["test"]
    OOD = [((e["Body"] + " " + e["Question"]).strip(), str(e["Answer"]).replace(".0", ""))
           for e in sv][:N_GREEDY]
    OOD_NAME = "SVAMP"
except Exception as err:
    print(f"SVAMP unavailable ({err}); falling back to held-out GSM8K slice")
    OOD = [(e["question"], e["answer"].split("####")[-1].strip().replace(",", ""))
           for e in gsm][500:500 + N_GREEDY]
    OOD_NAME = "GSM8K (held-out slice)"

# ---------------------------------------------------------------- helpers

tok = AutoTokenizer.from_pretrained(BASE_ID, padding_side="left")


def extract(text):
    """Returns (answer, used_required_format). Tracking the format flag
    separately keeps 'got it right' distinct from 'followed the spec'."""
    m = re.findall(r"####\s*(-?[\d,]+)", text)
    if m:
        return m[-1].strip().replace(",", ""), True
    m = re.findall(r"(-?[\d,]+\.?\d*)", text)
    return (m[-1].strip().replace(",", ""), False) if m else (None, False)


def prompts(questions):
    texts = [tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True) for q in questions]
    return tok(texts, return_tensors="pt", padding=True,
               truncation=True, max_length=512).to(DEVICE)


@torch.no_grad()
def generate(model, questions, sample=False, n=1):
    """Returns list-of-lists: n completions per question."""
    out = [[] for _ in questions]
    for _ in range(n):
        for i in range(0, len(questions), BATCH):
            chunk = questions[i:i + BATCH]
            b = prompts(chunk)
            g = model.generate(
                **b, max_new_tokens=MAX_NEW, do_sample=sample,
                temperature=1.0 if sample else None,
                top_p=1.0 if sample else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = g[:, b["input_ids"].shape[1]:]
            for j, row in enumerate(gen):
              out[i + j].append((tok.decode(row, skip_special_tokens=True),int((row != tok.pad_token_id).sum())))
    return out


def score(model, data, sample=False, n=1):
    qs = [q for q, _ in data]
    golds = [a for _, a in data]
    gens = generate(model, qs, sample=sample, n=n)

    any_correct = 0      # pass@n
    first_correct = 0    # pass@1
    fmt_ok, lengths = 0, []
    total_gens = 0

    for outs, gold in zip(gens, golds):
        hits = []
        for idx, (text, ln) in enumerate(outs):
            ans, ok = extract(text)
            correct = (ans == gold)
            hits.append(correct)
            fmt_ok += ok
            lengths.append(ln)
            total_gens += 1
            if idx == 0 and correct:
                first_correct += 1
        if any(hits):
            any_correct += 1

    return {
        "pass@1": first_correct / len(data),
        f"pass@{n}": any_correct / len(data),
        "format_compliance": fmt_ok / total_gens,
        "mean_length": sum(lengths) / len(lengths),
    }


def load(path):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=DTYPE).to(DEVICE)
    m.eval()
    return m


# ---------------------------------------------------------------- run

results = {}
for name, path in [("base_student", BASE_ID),
                   ("distilled_student", CKPT),
                   ("teacher_ceiling", TEACHER_ID)]:
    print(f"\n=== {name} ===")
    model = load(path)

    r = {}
    r["in_domain"] = score(model, IN_DOMAIN[:N_GREEDY])
    print("  in-domain :", r["in_domain"])
    r["out_of_domain"] = score(model, OOD)
    print(f"  {OOD_NAME[:10]:<10}:", r["out_of_domain"])

    # pass@K on the teacher is not informative for diversity collapse; skip it
    # to save GPU time. The comparison that matters is base vs distilled.
    if name != "teacher_ceiling":
        r["diversity"] = score(model, IN_DOMAIN[:N_PASSK], sample=True, n=K)
        print(f"  pass@{K}    :", r["diversity"])

    results[name] = r
    del model
    torch.cuda.empty_cache()

json.dump({"ood_dataset": OOD_NAME, "results": results},
          open("eval_results.json", "w"), indent=2)

# ---------------------------------------------------------------- report

b, d = results["base_student"], results["distilled_student"]
print(f"""
| Metric | Base | Distilled | Teacher |
|---|---|---|---|
| GSM8K pass@1 | {b['in_domain']['pass@1']:.3f} | {d['in_domain']['pass@1']:.3f} | {results['teacher_ceiling']['in_domain']['pass@1']:.3f} |
| {OOD_NAME} pass@1 | {b['out_of_domain']['pass@1']:.3f} | {d['out_of_domain']['pass@1']:.3f} | {results['teacher_ceiling']['out_of_domain']['pass@1']:.3f} |
| GSM8K pass@{K} | {b['diversity'][f'pass@{K}']:.3f} | {d['diversity'][f'pass@{K}']:.3f} | - |
| Format compliance | {b['in_domain']['format_compliance']:.3f} | {d['in_domain']['format_compliance']:.3f} | - |
| Mean length (tokens) | {b['in_domain']['mean_length']:.0f} | {d['in_domain']['mean_length']:.0f} | - |

Single seed, {N_GREEDY} problems. Differences under ~5 points are not meaningful
at this sample size -- say so rather than claiming them.
""")
