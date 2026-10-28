# G5 experiment plan v0 (needs TSAI approval by Sat 31 Oct)

Question: best tiny-model configuration to freeze 14 Nov. Every arm below runs the
week-1 harness (`harness/train.py` on `configs/config_v0.yaml`) on a single GPU
(Colab/Kaggle T4 class) unless noted. Kronecker embeddings in ALL arms — standard
embeddings are out of scope (V4: results do not carry over).

## Arms

| Arm | Varies | Metric | Pass bar (proposal for review) | GPU h (proposal for review) |
|-----|--------|--------|----------|-------|
| LR-1/2/3/4 | lr = 1e-5 / 1e-4 (base) / 2e-4 / 1e-3, cosine to 1e-5, warmup 10. V4's 3B MoE bakeoff (lr_schedule.yaml stages BAKEOFF_3B, BAKEOFF_3B_10X) used flat 1e-5 and flat 1e-4; our arms use a cosine decay, so they are not a re-run of it | `loss_ntp`, mean of final 20 steps (single-step values are noisy), plus held-out val loss at the final step | best arm beats others by >= 5% relative, no NaN; winner repeated on 3 seeds | 4 x 2 |
| K-8/16 | kron `pos_dim` 8 (D=2048) vs 16 (D=4096) | `loss_ntp` @ step 500 + input-side param count | pick cheaper D unless loss gap > 3% | 2 x 2 |
| M-4/8 | MoE experts 4/top-1 vs 8/top-2 | `loss_ntp` @ step 500 + `tokens_per_second` | winner keeps >= 80% of dense throughput | 2 x 3 |
| D-ord | data order: abrupt EN->HI switch vs gradual EN/HI blend | `loss_ntp` spike (max - pre-switch) | blend wins: spike < 50% of abrupt spike (V4 collapse lesson) | 2 x 2 |
| C-ckpt | checkpoint resume: loss before/after resume at step 250 | `loss` continuity | post-resume loss within 1% of uninterrupted run | 1 x 3 |

LR arms: early signal only (tiny model). Arms ran concurrently on a shared box, so their `tokens_per_second` is contaminated and not comparable across arms.

Total: ~25 GPU hours on T4-class (proposal for review; the sum of the table rows. The earlier '~21' did not match the table). Owners assigned at 26 Oct sync.

## Rules for comparing arms (all proposals for review)

- **Parity:** arms being compared use the same tokenizer, held-out data, training-token budget, batch policy and precision, and each reports its parameter count (total and input-side). Optimizer settings are held fixed across arms unless the arm varies them: gradient clipping 1.0 (zero-1-1b.json / zero-1-120b-tqlora.json `gradient_clipping`), AdamW betas 0.9/0.95 and eps 1e-10 (zero-1-1b.json `optimizer.params`), weight decay 0.0 (lr_schedule.yaml stage 3B_MOE_PRODUCTION). The MoE assumption (V5 expected to be a MoE) is an assumption, not a team decision.
- **Seeds:** single-seed results are preliminary. The LR winner needs repeats on 3 seeds before it counts.
- **GPU-hour cap:** each arm's GPU-hour figure in the table is its hard cap. Stop at the cap and report where the arm got to; do not extend silently.
- **Metric definitions:** a *step* is one completed optimizer update. *Train loss* (`loss_ntp`, `loss`) is the mean token cross-entropy from that step's forward pass, measured before the update.

## Gates

- Determinism gate first: seed 1234 twice => empty `diff` on `losses.json` (holds only seed, loss_ntp, loss, lr), decreasing, no NaN. No arm starts before this passes. Throughput/memory go to `metrics.jsonl` and are never diffed.
  - Same-setup repeats (same hardware, config, launch: gate1 vs gate2, ddp1 vs ddp2, ds1 vs ds2) must be byte-identical.
  - Cross-setup matches (1 vs 4 GPU, GPU vs CPU, DDP vs DeepSpeed) must agree within 1e-5 on every step of `loss_ntp` (lr and seed exact). fp32 gradient summation order differs across parallelism, so exact equality is impossible there.
- DDP payoff starts at real batch sizes: at the tiny gate size (one row per rank) 4 GPUs are slower than 1 because allreduce dominates. This is expected; do not "optimize" it.
- Each arm logs the G8 metric names: `tokens_per_second, loss_ntp, loss, lr, gpu_alloc_gb` (`loss_mtp` joins when MTP plumbing lands).
- Kill rule (proposal for review): any NaN, or, after step 100, the mean of the last 20 steps above the mean of the first 20 steps => stop arm, report, do not retry silently. (Replaces the single-step comparison against step 0, which killed a run whose validation loss was still falling.)

## Handoffs (proposal for review)

| Item | Direction | Date | Action |
|------|-----------|------|--------|
| KE2 output shape | G4 -> G5 | 27 Oct | G5 plugs it into `data.ke2_shape_tbd` / the embedding selection point |
| Harness + small configs | G5 -> G6 | 27 Oct | G6 reuses the harness and tiny configs |
| Config v0 | G5 -> G7 | 27 Oct | G7 reads `configs/config_v0.yaml` (blocked on this file) |
| Metric names | G5 -> G8 | 27 Oct | G8 wires `tokens_per_second, loss_ntp, loss, lr, gpu_alloc_gb` |
| Trainer logger + alarms | G8 -> G5 | 29 Oct | G5 integrates into `harness/train.py` |
