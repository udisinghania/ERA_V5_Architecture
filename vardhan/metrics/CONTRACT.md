# G8 metric contract — the six names, reconciled across all six folders

For G8's logger integration on **Thu 29 Oct**. G8 was handed six scalar names
on Tuesday. Every member logs more than six and names them differently, and one
of the six — `loss` — is defined two incompatible ways across the team. This
file is the single mapping G8 needs, plus the one decision the names do not
settle. It reads only the committed `metrics.jsonl` files; `verify/metrics_contract.py`
reproduces every claim here.

## The six (plus `step`)

| Name | Meaning | Unit | Notes |
|---|---|---|---|
| `step` | optimizer update index | int | starts at 1 (Udit) or 0 (others); a cosmetic offset, not a gate item |
| `loss_ntp` | next-token cross-entropy on the train batch | nats/token | the quantity the Saturday "loss falls" gate reads |
| `loss_mtp` | multi-token-prediction head CE | nats/token | `null`/absent when MTP is off; **not** part of `loss` unless the harness weights it in |
| `loss` | **the optimized total** = `loss_ntp` + weighted `loss_mtp` + MoE aux + router/output z | nats | see the decision below — this is the contested one |
| `lr` | learning rate of the **AdamW** parameter group | — | under the hybrid Muon+AdamW optimizer, Muon's rate is a separate `muon_lr` |
| `tokens_per_second` | throughput | tokens/s | informational; **not reproducible**, differs run to run even when losses are byte-identical |
| `gpu_alloc_gb` | peak CUDA memory, decimal GB | GB | `0` on CPU |

## Decision 1 — `loss` is the optimized total, not the NTP component

The name alone does not fix this, and the team is split today:

| Folder | What `loss` equals | Evidence |
|---|---|---|
| deepjyotisaha | total (ntp + mtp + aux) | `loss - loss_ntp` up to 2.71; 0/200 rows equal |
| revantg | total (ntp + z + aux) | `loss - loss_ntp` = 0.014 (= z_loss); 0/1000 rows equal |
| Udit | total (`train/total_loss`) | `loss - loss_ntp` up to 0.029; distinct from `train/ce_loss` |
| Vedang | total (harness doc, line 61) | `loss = loss_ntp + mtp_coef*mtp + balance + z` |
| Dharshan | total (README §8) | `loss = ntp + z + aux + weighted mtp` |
| **shivman** | **NTP component** | `loss == loss_ntp` in **40/40 rows**, though `loss_mtp` is present and nonzero |

Five of six log the total. `SYNTHESIS.md` already records the agreement —
"loss means actual total, loss_ntp means CE". **shivman's folder is the file in
lab-repo PR #67**, and it logs `loss = loss_ntp`. If that merges unchanged, G8
plots a curve that is a component in one folder and the full objective in five,
and sees two identical columns (`loss`, `loss_ntp`) that look like a duplicated
field.

**Resolution, cheapest first:** one line in the PR #67 harness — write the
summed objective to `loss` instead of the NTP value (it already computes
`(ntp + mtp).backward()`); *or*, if the team wants `loss` to stay NTP there,
state it in `result.txt` and G8 reads `loss_ntp + loss_mtp` for that folder.
Either is fine; silence is not, because the two give different training curves.

## Decision 2 — `lr` under the hybrid optimizer

Muon on 2-D matrices + AdamW on the rest means two learning rates. The contract
is: `lr` = the AdamW group's rate; the Muon rate, when logged, is `muon_lr`
(Udit logs `optimizer/muon_lr`; Dharshan `lr_muon`). G8 should treat `lr` as a
single series and ignore `muon_lr` unless it wants the second curve.

## Per-folder mapping G8 wires against

Output of `python verify/metrics_contract.py --all ..` on 28 Oct:

| Folder | maps to the six? | renames to apply |
|---|---|---|
| deepjyotisaha | all 6 exact | — |
| revantg | all 6 exact | — |
| shivman | all 6 exact | **but `loss` = NTP (Decision 1)** |
| Udit | all via rename | `loss_ntp→train/ce_loss`, `loss_mtp→loss/mtp`, `loss→train/total_loss`, `lr→optimizer/adamw_lr`, `tokens_per_second→perf/tokens_per_second`; **`gpu_alloc_gb` not logged** |
| Vedang | CSV, no `metrics.jsonl` yet | splits reproducible metrics from `tokens_per_second`/`gpu_alloc_gb` (the `.system.csv`) |
| Dharshan | CSV, no runs yet | `metrics.csv`, same six names |

Two gaps that are not name problems: **Udit does not log `gpu_alloc_gb` at all**,
and **Vedang/Dharshan log CSV, not JSONL** — G8 needs either a JSONL row per
step from them or a reader for their `metrics.csv`.

## Run it

```bash
python verify/metrics_contract.py --all ..          # every folder, summary table
python verify/metrics_contract.py ../shivman        # one folder
python verify/metrics_contract.py ../Udit/evidence/operational_gates/full_a/metrics.jsonl
```

Exits non-zero if a required name (`step`, `loss_ntp`, `loss`, `lr`) cannot be
mapped, or if `loss` is logged as the NTP component. Reads the files only —
never imports or runs a harness.
