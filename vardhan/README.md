# vardhan

Vardhan Walavalkar. Week-1 folder in the working repo.

I am not adding a seventh `config_v0.yaml`. Six of us have written one, each
with its own schema, its own checker and its own evidence layout, and
`SYNTHESIS.md` has to cherry-pick between them. What was missing was a way to
check all six **the same way**. That is what this folder is.

## `verify/gate1.py` — one Gate-1 check for every submission

The Saturday gate is:

> config loads, `check_config.py` gives 0 errors, two seed-1234 runs match,
> the loss goes down, no NaN, TSAI approves the plan.

```bash
python verify/gate1.py ../shivman      # one member
python verify/gate1.py --all ..        # everyone, with a summary table
```

Needs `pyyaml`. Nothing else — standard library only.

### Two rules it follows

**1. It never imports or executes code from the folder it checks.** The source
audit re-reads the YAML itself; the run checks read the evidence files. A
submission cannot pass by shipping a checker that prints `0 errors`. This also
means it is safe to point at a branch you have not read.

**2. It does not assume your layout.** Both config shapes in use are handled —
the flat `settings:` record list, and the nested mapping with trailing
`# source:` comments — and so are the five loss-file formats in this repo:
bare floats, `step` + C99 hex floats, zero-padded `000001 12.3`, CSV with a
header, and tab-separated. Run pairs are discovered automatically
(`run_a`/`run_b`, `run1`/`run2`, `full_a`/`full_b`, `R1a`/`R1b`).

### What counts as a failure, and what does not

Only the gate items above can fail. Things that are a **different convention
rather than a defect** are reported as `[note]`, never as a failure:

- sources kept in a separate `SOURCES.md` or `README.md` instead of per-line
  comments — the tool says so and checks values only;
- a `null` whose source explains what it is waiting for — that is a declared
  open size, not a missing value;
- metric names that differ from the six G8 was promised — reported as a
  mapping, because the names were a Tuesday handoff, not a Saturday gate item;
- a folder with no config **and** no run evidence at all — that is a folder
  holding something other than a submission, so `--all` marks it `-` rather
  than `INCOMPLETE`. This folder is one of them. A folder that has a config but
  no runs is still `INCOMPLETE`, because that one really is an unmet gate.

This matters. The first version of this tool failed Vedang and Revant for
"no source" when their sources were simply in another file, and told Udit all
six metrics were missing when five were present under other names. A verifier
that misreports teammates' work is worse than no verifier.

## What the 27 Oct sweep found

Five of six folders pass the gate checks. See `findings/gate1-sweep-27oct.md`
for the detail. The three things worth acting on:

1. **`gpu_alloc_gb` is not logged anywhere in Udit's runs**, and the other five
   G8 names are renamed (`train/ce_loss`, `loss/mtp`, `train/total_loss`,
   `optimizer/adamw_lr`). G8 integrates on Thursday against the six exact names
   we sent on Tuesday.
2. **`config_v0.yaml` leaves no setting open** in two folders, though the rule
   is that sizes stay empty until TSAI fills them on 31 Oct.
3. **One folder has no run evidence at all** — no loss files anywhere, so the
   matching-seed and falling-loss claims cannot be checked.

## `metrics/CONTRACT.md` — the six G8 names, reconciled

`verify/metrics_contract.py` does for the metric handoff what `gate1.py` does
for the Saturday gate: it reads every folder's `metrics.jsonl` and reports, per
member, which of the six names G8 was promised are present, what each is
renamed to, and one semantic the names do not settle — whether `loss` is the
optimized total or the next-token component. Five folders log the total; the
skeleton in PR #67 logs the NTP component. `metrics/CONTRACT.md` is the single
mapping G8 wires against on 29 Oct, with that decision called out.

```bash
python verify/metrics_contract.py --all ..
```

## Diagnosis

My pack is `ind-mr-fw2`. Rows live in the lab repo, not here:
`diagnosis/submissions/ind-mr-fw2/vpw/` (PR 43).
