"""Check a run's metrics.jsonl against the six names G8 integrates against.

G8 was handed six scalar names on Tuesday: step plus

    tokens_per_second, loss_ntp, loss_mtp, loss, lr, gpu_alloc_gb

Every member logs more than this and names things differently, so when G8
wires its logger in on Thu 29 Oct it has to know, per member, (a) which of the
six are present, (b) what each is called if renamed, and (c) one semantic that
is NOT settled by the name alone: whether `loss` is the optimized total or just
the next-token component. Five of six folders log `loss` = total; one logs
`loss` = loss_ntp. A plot of "the loss" is a different curve in each case.

This tool reads metrics.jsonl only. It never imports or runs a harness - same
rule as verify/gate1.py. Standard library only.

    python verify/metrics_contract.py ../shivman
    python verify/metrics_contract.py --all ..
    python verify/metrics_contract.py ../Udit/evidence/operational_gates/full_a/metrics.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The contract. Order is the handoff order. `required` names must map to
# something in every run; `loss_mtp` is conditional (absent/None when MTP is
# off) and `gpu_alloc_gb` is informational (0 on CPU), so a miss on those two
# is a note, not a failure.
CANONICAL = ["step", "loss_ntp", "loss_mtp", "loss", "lr",
             "tokens_per_second", "gpu_alloc_gb"]
REQUIRED = {"step", "loss_ntp", "loss", "lr"}

# Aliases seen across the six folders. Matched after stripping any
# "namespace/" prefix, so train/ce_loss resolves by its leaf ce_loss.
ALIASES = {
    "loss_ntp":          ("ce_loss", "ce", "ntp", "loss_ce"),
    "loss_mtp":          ("mtp", "mtp_loss"),
    "loss":              ("total_loss", "total"),
    "lr":                ("adamw_lr", "learning_rate", "lr_adamw"),
    "tokens_per_second": ("tokens_per_sec", "tps", "step_tokens_per_second"),
    "gpu_alloc_gb":      ("gpu_alloc", "gpu_mem_gb", "cuda_alloc_gb",
                          "gpu_memory_gb", "peak_gpu_gb"),
}
# A companion learning rate under a hybrid Muon+AdamW optimizer. Its presence
# is why `lr` needs a stated meaning (we take `lr` = the AdamW group).
MUON_LR = ("muon_lr", "lr_muon")
# Names that, if present and nonzero, make `loss` == loss_ntp provably NOT the
# optimized total: there is an extra term in the objective.
EXTRA_TERMS = ("loss_mtp", "mtp", "z_loss", "router_z", "loss/router_z",
               "moe_aux", "moe_aux_loss", "loss/moe_aux", "logit_z")


def read_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def leafmap(keys):
    """leaf name -> full key, for keys present in every row."""
    return {k.split("/")[-1]: k for k in keys}


def resolve(want, keys, leaves):
    if want in keys:
        return want, "exact"
    if want in leaves:
        return leaves[want], "renamed"
    for a in ALIASES.get(want, ()):
        if a in leaves:
            return leaves[a], "renamed"
    return None, "absent"


def num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def check_file(path: Path):
    """Return (coverage list, loss_verdict, detail) for one metrics.jsonl."""
    rows = read_rows(path)
    if not rows:
        return None
    common = set(rows[0]).intersection(*[set(r) for r in rows[1:]]) if len(rows) > 1 else set(rows[0])
    leaves = leafmap(common)

    coverage = []
    mapping = {}
    for want in CANONICAL:
        hit, how = resolve(want, common, leaves)
        coverage.append((want, how, hit))
        mapping[want] = hit

    # The loss-semantics check. If `loss` and `loss_ntp` are present and equal
    # on every row, yet some extra objective term is present and nonzero
    # somewhere, then `loss` is the NTP component, not the total.
    verdict, detail = "unknown", ""
    lk, nk = mapping.get("loss"), mapping.get("loss_ntp")
    if lk and nk:
        pairs = [(num(r.get(lk)), num(r.get(nk))) for r in rows]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if pairs:
            eq = sum(1 for a, b in pairs if abs(a - b) < 1e-9)
            extras = [e for e in EXTRA_TERMS
                      if (e in common or e.split("/")[-1] in leaves)]
            extra_nonzero = False
            for e in extras:
                full = e if e in common else leaves.get(e.split("/")[-1])
                if full and any((num(r.get(full)) or 0) not in (0, None) for r in rows):
                    extra_nonzero = True
                    break
            if eq == len(pairs) and extra_nonzero:
                verdict = "ntp"
                detail = (f"loss == loss_ntp in all {len(pairs)} rows, but "
                          f"{', '.join(extras)} present and nonzero -> `loss` is the "
                          f"NTP component, not the optimized total")
            elif eq == len(pairs):
                verdict = "ntp?"
                detail = (f"loss == loss_ntp in all {len(pairs)} rows; no extra "
                          f"term logged, so total and NTP coincide here")
            else:
                maxd = max(abs(a - b) for a, b in pairs)
                verdict = "total"
                detail = (f"loss != loss_ntp (max |diff| {maxd:.4f}) -> `loss` is "
                          f"the optimized total, as the contract requires")
    muon = next((m for m in MUON_LR if m in leaves), None)
    return coverage, verdict, detail, muon, len(rows)


def report_folder(root: Path):
    files = sorted(p for p in root.rglob("metrics.jsonl"))
    print(f"\n=== {root.name} " + "=" * max(0, 60 - len(root.name)))
    if not files:
        print("  no metrics.jsonl - harness may log CSV, or no runs yet")
        return "no-metrics"
    # One representative schema is enough; report the first, note the count.
    path = files[0]
    res = check_file(path)
    if res is None:
        print(f"  {path.relative_to(root)}: empty")
        return "empty"
    coverage, verdict, detail, muon, nrows = res
    print(f"  {path.relative_to(root)}  ({nrows} rows"
          + (f", +{len(files)-1} more file(s)" if len(files) > 1 else "") + ")")
    missing_required = []
    for want, how, hit in coverage:
        tag = {"exact": "ok      ", "renamed": "renamed ", "absent": "ABSENT  "}[how]
        shown = f"-> {hit}" if how == "renamed" else ("" if how == "exact" else "")
        print(f"    [{tag}] {want:18} {shown}")
        if how == "absent" and want in REQUIRED:
            missing_required.append(want)
    if muon:
        print(f"    [note    ] hybrid optimizer: `{muon}` also logged; `lr` = the AdamW group")
    print(f"    loss semantics: {verdict.upper():6} - {detail}")
    status = "ok"
    if missing_required:
        print(f"  -> FAIL: required metric(s) not found: {', '.join(missing_required)}")
        status = "fail"
    elif verdict == "ntp":
        print("  -> FAIL: `loss` logs the NTP component; G8 and 5/6 folders expect the total")
        status = "fail"
    else:
        print("  -> OK for G8")
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path, help="a member folder, a dir of folders with --all, or a metrics.jsonl")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    a.target = a.target.resolve()

    if a.target.is_file():
        res = check_file(a.target)
        if not res:
            sys.exit("empty file")
        coverage, verdict, detail, muon, nrows = res
        for want, how, hit in coverage:
            print(f"  {how:8} {want:18} {('-> '+hit) if how=='renamed' else ''}")
        if muon:
            print(f"  note     hybrid lr: {muon}; lr = AdamW group")
        print(f"  loss: {verdict} - {detail}")
        return 0

    if a.all:
        members = [p for p in sorted(a.target.iterdir())
                   if p.is_dir() and not p.name.startswith((".", "_"))]
        res = {m.name: report_folder(m) for m in members}
        print("\n" + "=" * 68)
        for k, v in res.items():
            print(f"  {v:11}  {k}")
        return 0
    return 0 if report_folder(a.target) == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
