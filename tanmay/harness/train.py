"""G5 tiny harness — loads config_v0.yaml, trains a tiny GPT-2 + Kronecker embedding
(+ minimal MoE when enabled), logs V4 metrics. Seed 1234 => identical loss file.

Determinism contract (Sat-31-Oct gate): two runs with seed 1234 produce
byte-identical losses.json (empty diff), loss decreases, no NaN.

Outputs (V4-style split):
  <out>/losses.json   ONLY seed, loss_ntp, loss, lr  (bit-deterministic; the gate diffs this)
  <out>/eval.json     held-out XNLI-validation token CE at step 0, every eval.every_steps, and the end
                      (deterministic; byte-identical on same-setup repeats; separate from losses.json)
  <out>/manifest.json run command, config hash, git commit, versions, CUDA/GPUs, world size, seed
                      (also never diffed)
  <out>/metrics.jsonl one JSON object per step: step, tokens_per_second, gpu_alloc_gb
                      (wall-clock / allocator dependent; NOT part of the gate)

    # needs Python >= 3.10 (kronecker-embeddings requires it); use `python -m pip` of that interpreter
    python -m pip install torch transformers kronecker-embeddings pyyaml pandas pyarrow datasets
    python -m pip install deepspeed   # optional, only for --deepspeed
    python data/make_mixed_sample.py  # builds data/mixed_en_hi_xnli.txt (never commit)

    # single process (CPU, or cuda:0 if visible)
    python harness/train.py --config configs/config_v0.yaml --out runs/gate1
    python harness/train.py --config configs/config_v0.yaml --out runs/gate2  # same seed
    diff runs/gate1/losses.json runs/gate2/losses.json && echo DETERMINISTIC

    # multi-GPU DDP (global batch from config is split across ranks; batch % world_size == 0)
    python -m torch.distributed.run --nproc_per_node=4 harness/train.py --config <cfg with batch: 4> --out runs/ddp1
    # (the `torchrun` script on this box belongs to another Python and fails; use the -m form)
    # optional DeepSpeed ZeRO-3 path (needs `pip install deepspeed`, not part of the gate)
    python -m torch.distributed.run --nproc_per_node=4 harness/train.py --config <cfg> --out runs/ds --deepspeed

Data: cfg data.train_sample (mixed EN+HI sample, one text per line). Batch selection is a
pure function of the step (no RNG): global row i of step s is line (s*batch + i) % n_lines;
rank r takes rows r, r+world, ... Falls back to seeded synthetic ids (INVALID for our
purposes) ONLY if --allow_fallback is passed; otherwise a missing tokenizer/Kronecker/data
exits with code 4.

Kill rule (plan_v0): NaN, or after step 100 the mean of the last 20 steps above the mean of the
first 20 steps -> stop, exit code 3 (partial trace in losses_partial.json).
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml


# ---------------------------------------------------------------- determinism
def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


# ---------------------------------------------------------------- model
class TinyMoEMLP(nn.Module):
    """Top-k MoE replacing one GPT-2 MLP. Experts are GPT2MLP-shaped FFNs."""

    def __init__(self, d_model, n_inner, n_experts, top_k, dropout=0.0):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, n_inner),
                    nn.GELU(),
                    nn.Linear(n_inner, d_model),
                    nn.Dropout(dropout),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, x):
        scores = torch.softmax(self.router(x), dim=-1)
        top = torch.topk(scores, self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = top.indices[..., k]
            w = top.values[..., k].unsqueeze(-1)
            for e, expert in enumerate(self.experts):
                mask = (idx == e).unsqueeze(-1)
                # every expert runs on every rank every step (no data-dependent skip): ZeRO-3 must
                # all-gather parameters in the same order on all ranks or the collectives deadlock.
                out = out + torch.where(mask, w * expert(x), torch.zeros_like(x))
        return out


def build_model(cfg, allow_fallback=False):
    from transformers import GPT2Config, GPT2LMHeadModel

    m = cfg["model"]
    gpt = GPT2Config(
        vocab_size=50257,
        n_positions=512,
        n_layer=m["n_layer"],
        n_embd=m["n_embd"],
        n_head=m["n_head"],
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        # determinism: eager attention (SDPA mem-efficient backward is non-deterministic)
        attn_implementation="eager",
        # explicit loss type (silences "loss_type=None" notice; loss itself is computed below)
        loss_type="ForCausalLM",
    )
    model = GPT2LMHeadModel(gpt)
    emb_cfg = m["embedding"]
    assert emb_cfg["type"] == "kronecker", "G5 rule: Kronecker only, never standard"
    try:
        from kronecker_embeddings import KroneckerEmbedding
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        kron = KroneckerEmbedding(
            vocab_size=tok.vocab_size,
            d_model=m["n_embd"],
            tokenizer=tok,
            pos_dim=emb_cfg["pos_dim"],
        )
        model.transformer.wte = kron  # drop-in: same forward signature
        tokenizer = tok
    except Exception as e:
        if not allow_fallback:  # an invalid run must never look like a passing one
            print(f"ERROR: Kronecker/tokenizer path unavailable ({e!r}); run would be INVALID. "
                  "Fix network/tokenizer access, or pass --allow_fallback to debug only.", file=sys.stderr)
            sys.exit(4)
        print(f"WARNING: Kronecker/HF path unavailable ({e}); standard emb fallback")
        tokenizer = None
    if m["moe"]["enabled"]:
        for i in range(m["n_layer"]):
            blk = model.transformer.h[i]
            blk.mlp = TinyMoEMLP(
                m["n_embd"], 4 * m["n_embd"], m["moe"]["n_experts"], m["moe"]["top_k"]
            )
    return model, tokenizer


class Batcher:
    """Pure-function-of-step batching. No RNG, no shuffling, identical on every rank."""

    def __init__(self, cfg, tokenizer, allow_fallback=False):
        self.seq = int(cfg["training"].get("seq_len", 32))
        self.batch = int(cfg["training"].get("batch", 2))
        self.tok = tokenizer
        self.lines = []
        path = cfg["data"].get("train_sample", "")
        if tokenizer is not None and path and os.path.exists(path):
            with open(path) as f:
                self.lines = [l.strip() for l in f if l.strip()]
        if not self.lines and not allow_fallback:
            print(f"ERROR: training data '{path}' missing/empty (run data/make_mixed_sample.py); "
                  "synthetic ids are INVALID. Pass --allow_fallback to debug only.", file=sys.stderr)
            sys.exit(4)
        self.cache = {}
        # one line per row (truncated/cycled to seq+1): rank splits then see exactly the single-GPU rows

    def _row(self, line_idx):
        if line_idx not in self.cache:
            ids = self.tok(self.lines[line_idx], add_special_tokens=False).input_ids
            if not ids:
                ids = [self.tok.eos_token_id]
            need = self.seq + 1
            self.cache[line_idx] = (ids * (need // len(ids) + 1))[:need]
        return self.cache[line_idx]

    def get(self, step, device, rank=0, world=1):
        rows_idx = list(range(rank, self.batch, world))  # global rows owned by this rank
        if self.lines:
            n = len(self.lines)
            start = (step * self.batch) % n
            rows = [self._row((start + i) % n) for i in rows_idx]
            t = torch.tensor(rows, dtype=torch.long)
        else:  # seeded synthetic fallback, still a pure function of step
            g = torch.Generator().manual_seed(1234 + step)
            full = torch.randint(0, 50257, (self.batch, self.seq + 1), generator=g)
            t = full[rows_idx]
        t = t.reshape(len(rows_idx), self.seq + 1)
        return t[:, :-1].to(device), t[:, 1:].to(device)


def load_eval_rows(cfg, tok, seq, allow_fallback=False):
    """Held-out eval rows: XNLI validation en+hi, file order, NOT tiled/repeated (truncated to seq+1)."""
    ev = cfg.get("eval") or {}
    path = ev.get("data", "")
    if tok is None:
        return []
    if not path or not os.path.exists(path):
        if allow_fallback:
            print(f"WARNING: eval data '{path}' missing; eval skipped (--allow_fallback)")
            return []
        print(f"ERROR: eval data '{path}' missing (run data/make_mixed_sample.py --split validation)", file=sys.stderr)
        sys.exit(4)
    rows = []
    with open(path) as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            ids = tok(l, add_special_tokens=False).input_ids[: seq + 1]
            if len(ids) >= 2:
                rows.append(ids)
    return rows


@torch.no_grad()
def run_eval(fwd, rows, device, bs=128):
    """Mean token cross-entropy over all rows (fixed order, fixed batches, fp64 accumulation).
    Every rank evaluates the full set so ZeRO-3 collectives stay aligned."""
    fwd.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(rows), bs):
        chunk = rows[i : i + bs]
        L = max(len(r) for r in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        mask = torch.zeros(len(chunk), L - 1, dtype=torch.bool)
        for j, r in enumerate(chunk):  # right-pad: causal attention keeps valid positions unaffected
            ids[j, : len(r)] = torch.tensor(r)
            mask[j, : len(r) - 1] = True
        ids, mask = ids.to(device), mask.to(device)
        logits = fwd(input_ids=ids[:, :-1]).logits
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction="none"
        ).double()
        m = mask.reshape(-1)
        tot += (ce * m).sum().item()
        cnt += int(m.sum().item())
    fwd.train()
    return tot / cnt, cnt


def write_manifest(out, args, seed, world, status, use_cuda):
    """manifest.json: run command, environment, repeatability evidence. Never part of the gate diff."""
    import hashlib
    import platform
    import subprocess
    from importlib import metadata

    def ver(pkg):
        try:
            return metadata.version(pkg)
        except Exception:
            return None

    def sha(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None

    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", here], cwd=here, capture_output=True, text=True, timeout=10)
        git = {"commit": commit.stdout.strip() or None, "uncommitted_changes": bool(dirty.stdout.strip()) if commit.returncode == 0 else None}
        if commit.returncode != 0:
            git["note"] = "not a git checkout (git rev-parse failed)"
    except Exception as e:
        git = {"commit": None, "note": f"git unavailable: {e!r}"}
    manifest = {
        "status": status,  # completed | killed | failed_gate
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "config_file": os.path.abspath(args.config),
        "config_sha256": sha(args.config),
        "git": git,
        "seed": seed,
        "world_size": world,
        "deepspeed_path": bool(args.deepspeed),
        "allow_fallback": bool(args.allow_fallback),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": ver("transformers"),
            "kronecker-embeddings": ver("kronecker-embeddings"),
            "deepspeed": ver("deepspeed"),
        },
        "cuda_version": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if use_cuda else [],
        "repeatability": {
            "losses_json_sha256": sha(os.path.join(out, "losses.json")),
            "note": "same-setup repeats must have identical losses_json_sha256; cross-setup within 1e-5/step (plan_v0)",
        },
    }
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--allow_fallback", action="store_true", help="debug only: permit standard-emb/synthetic fallback (run is INVALID)")
    ap.add_argument("--no_find_unused", action="store_true", help="DDP find_unused_parameters=False (Task F test)")
    ap.add_argument("--deepspeed", action="store_true", help="route through deepspeed.initialize")
    ap.add_argument("--ds_config", default=os.path.join(os.path.dirname(__file__), "..", "deepspeed", "zero3_4gpu.json"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    tr = cfg["training"]
    seed = a.seed if a.seed is not None else tr["seed"]
    seed_all(seed)

    # ---- distributed setup (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK)
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    if world > 1:
        if use_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl" if use_cuda else "gloo")
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    is_main = rank == 0

    batch = int(tr.get("batch", 2))
    if batch % world != 0:
        raise SystemExit(
            f"ERROR: global batch {batch} (training.batch) is not divisible by world_size {world}; "
            f"set batch to a multiple of {world}"
        )

    model, tok = build_model(cfg, a.allow_fallback)
    model.to(device).train()
    base_lr = float(tr["lr"])
    grad_clip = float(tr["grad_clip"])  # V4: gradient_clipping 1.0 on every path (never DeepSpeed's default)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(float(tr["adam_beta1"]), float(tr["adam_beta2"])),
        eps=float(tr["adam_eps"]),
        weight_decay=float(tr["weight_decay"]),
    )
    engine = None
    if a.deepspeed:
        import deepspeed  # lazy: not required for the gate

        ds_cfg = json.load(open(a.ds_config))
        ds_cfg["train_batch_size"] = batch
        ds_cfg["train_micro_batch_size_per_gpu"] = batch // world
        if ds_cfg.get("gradient_clipping") != grad_clip:  # config is the single source of truth
            print(f"NOTE: deepspeed json gradient_clipping={ds_cfg.get('gradient_clipping')} overridden by "
                  f"training.grad_clip={grad_clip}", flush=True)
        ds_cfg["gradient_clipping"] = grad_clip
        engine, opt, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=ds_cfg)
        fwd = engine
    elif world > 1:
        # find_unused_parameters=True is kept as the MoE top-1 safeguard. Verified load-bearing when experts
        # were skipped data-dependently (off => ranks desync: "mismatch between collectives" / hang);
        # with every expert always executed (TinyMoEMLP) off also ran clean for 500 steps.
        # KroneckerEmbedding keeps int16 lookup buffers (built from the tokenizer, identical on
        # every rank); NCCL cannot broadcast them, so exclude non-float buffers from DDP sync.
        skip = [n for n, b in model.named_buffers() if not b.is_floating_point()]
        nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, skip)
        fwd = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if use_cuda else None,
            find_unused_parameters=not a.no_find_unused,
        )
    else:
        fwd = model

    steps = int(tr["max_steps"])
    warmup = int(tr["warmup_steps"])
    min_lr = float(tr.get("lr_min", 1.0e-5))

    def lr_at(s):  # V4: linear warmup then cosine decay to lr_min
        if s < warmup:
            return base_lr * (s + 1) / max(1, warmup)
        p = (s - warmup) / max(1, steps - warmup)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(p, 1.0)))

    batcher = Batcher(cfg, tok, a.allow_fallback)
    eval_rows = load_eval_rows(cfg, tok, batcher.seq, a.allow_fallback)
    eval_every = int((cfg.get("eval") or {}).get("every_steps", 0))
    evals, eval_tokens = [], 0

    def do_eval(done):  # done = completed optimizer updates
        nonlocal eval_tokens
        if eval_rows:
            v_, eval_tokens = run_eval(fwd, eval_rows, device)
            evals.append({"step": done, "val_loss_ntp": v_})
            if is_main:
                print(f"eval @ step {done}: val_loss_ntp={v_:.4f} ({len(eval_rows)} rows, {eval_tokens} tokens)", flush=True)

    do_eval(0)
    if is_main:
        os.makedirs(a.out, exist_ok=True)
        mf = open(os.path.join(a.out, "metrics.jsonl"), "w")
    losses, lrs = [], []
    killed = None
    for s in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(s)
        xb, yb = batcher.get(s, device, rank, world)
        if use_cuda:
            torch.cuda.synchronize(device)
        t0 = time.time()
        logits = fwd(input_ids=xb).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), yb.reshape(-1)
        )
        if engine is not None:
            engine.backward(loss)
            engine.step()
        else:
            opt.zero_grad()
            loss.backward()  # DDP averages grads across ranks (deterministic allreduce)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        gl = loss.detach().clone()
        if world > 1:  # global mean loss (equal local batch sizes)
            dist.all_reduce(gl)
            gl /= world
        v = gl.item()  # syncs
        dt = time.time() - t0
        if not math.isfinite(v):
            killed = f"NaN/Inf loss at step {s}"
            break
        losses.append(v)
        lrs.append(lr_at(s))
        tps = round(batch * xb.size(1) / max(dt, 1e-6), 1)
        if is_main:
            gb = round(torch.cuda.memory_allocated(device) / 1e9, 3) if use_cuda else 0.0
            mf.write(json.dumps({"step": s, "tokens_per_second": tps, "gpu_alloc_gb": gb}) + "\n")
            print(f"step {s}: loss_ntp={v:.4f} lr={lrs[-1]:.2e} tokens_per_second={tps}", flush=True)
        if eval_every > 0 and ((s + 1) % eval_every == 0 or s + 1 == steps):
            do_eval(s + 1)
        if len(losses) > 100:  # kill rule (plan_v0): after step 100, mean of last 20 > mean of first 20
            first20 = sum(losses[:20]) / 20
            last20 = sum(losses[-20:]) / 20
            if last20 > first20:
                killed = (f"mean of last 20 steps {last20:.4f} above mean of first 20 steps {first20:.4f} "
                          f"at step {s} (>100 steps)")
                break

    if is_main:
        mf.close()
        name = "losses.json" if killed is None else "losses_partial.json"
        json.dump(
            {"seed": seed, "loss_ntp": losses, "loss": losses, "lr": lrs},
            open(os.path.join(a.out, name), "w"),
            indent=1,
        )
        if eval_rows:
            json.dump(
                {"seed": seed, "every_steps": eval_every, "n_rows": len(eval_rows), "n_tokens": eval_tokens,
                 "evals": evals},
                open(os.path.join(a.out, "eval.json"), "w"),
                indent=1,
            )
        if losses:
            print(f"wrote {a.out}/{name}  start={losses[0]:.4f} end={losses[-1]:.4f}")
    if is_main:
        status = "killed" if killed else ("completed" if losses and losses[-1] < losses[0] else "failed_gate")
        write_manifest(a.out, a, seed, world, status, use_cuda)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if killed:
        print(f"KILLED: {killed}", file=sys.stderr)
        sys.exit(3)
    if not losses[-1] < losses[0]:
        raise SystemExit("gate requires decreasing loss")


if __name__ == "__main__":
    main()
