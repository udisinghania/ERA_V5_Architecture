# kronecker-moe-gate

Tiny GPT-2 with a Kronecker embedding and a 4-expert top-1 MoE, trained on English+Hindi (XNLI). Seeded and deterministic; runs on 1 GPU, CPU, 4-GPU DDP and 4-GPU DeepSpeed ZeRO-3.

## Layout

- `configs/config_v0.yaml`: settings, each with a source comment
- `harness/train.py`: trainer
- `harness/check_config.py`: config checker
- `data/make_mixed_sample.py`: builds the en+hi text
- `deepspeed/zero3_4gpu.json`: ZeRO-3 config
- `experiments/plan_v0.md`: draft experiment plan

## Setup

Python 3.10+.

```
python -m pip install torch transformers kronecker-embeddings pyyaml pandas pyarrow datasets
python -m pip install deepspeed   # only for --deepspeed
python data/make_mixed_sample.py
python data/make_mixed_sample.py --split validation
```

## Run

From the repo root:

```
python harness/check_config.py configs/config_v0.yaml --allow-tbd ke2_shape_tbd size_tbd_1 size_tbd_2

python harness/train.py --config configs/config_v0.yaml --out runs/gate1
python harness/train.py --config configs/config_v0.yaml --out runs/gate2
diff runs/gate1/losses.json runs/gate2/losses.json

# 4 GPUs (training.batch must be a multiple of 4)
python -m torch.distributed.run --nproc_per_node=4 harness/train.py --config <cfg> --out runs/ddp
python -m torch.distributed.run --nproc_per_node=4 harness/train.py --config <cfg> --out runs/ds --deepspeed
```
