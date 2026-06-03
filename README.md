# PTQ for Spike-driven Transformer

This repository contains the implementation and experimental code for post-training quantization (PTQ) and structured pruning on Spike-driven Transformer models.

The code is based on Spike-driven Transformer classification models and extends them with:

* Post-training weight quantization for spiking neural networks
* ReScaW-style weight rescaling for PTQ
* Mixed-precision bit allocation
* Fisher-based sensitivity estimation
* TG-SRS structured pruning
* Fine-tuning scripts for pruned models

## 1. Overview

Spiking Neural Networks (SNNs) are promising for energy-efficient inference due to their sparse spike-driven computation. However, large-scale SNN Transformer models still contain considerable parameter and computation redundancy.

This repository explores two compression directions:

1. **Post-Training Quantization (PTQ)**
   The full-precision model is first trained normally. Then, the pretrained weights are quantized using calibration data without full retraining.

2. **Structured Channel Pruning**
   A task-guided spatiotemporal rank-based pruning strategy is used to estimate channel importance and prune redundant channels.

The final goal is to obtain a compressed Spike-driven Transformer with reduced memory cost and improved hardware friendliness.

## 2. Repository Structure

```text
.
├── main_finetune.py              # Training, evaluation, PTQ, and pruning entry
├── qp_quant.py                   # Quantization wrapper and PTQ implementation
├── tg_srs_prune.py               # TG-SRS structured pruning implementation
├── engine_finetune.py            # Training and evaluation loops
├── models/                       # Model definitions
├── util/                         # Utility functions
├── scripts/                      # Running scripts
└── README.md
```

The exact structure may vary depending on the local project version.

## 3. Environment

The experiments were conducted with Python and PyTorch.

Example environment:

```bash
conda create -n SpikeZoo python=3.8 -y
conda activate SpikeZoo
```

Install dependencies according to your local project requirements, for example:

```bash
pip install torch torchvision timm numpy scipy
```

If SpikingJelly is used:

```bash
pip install spikingjelly
```

## 4. Dataset

ImageNet is used for classification experiments.

Please organize the dataset as:

```text
/path/to/ImageNet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

In the following scripts, replace:

```bash
/data/dataset/ImageNet
```

with your actual ImageNet path.

## 5. Full-Precision Training

Before PTQ or pruning, train or prepare a full-precision checkpoint.

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_finetune.py \
  --model spikformer_8_15M_CAFormer \
  --data_path /data/dataset/ImageNet \
  --output_dir ./output_dir/15M_FP32 \
  --batch_size 128 \
  --epochs 300 \
  --blr 6e-4 \
  --weight_decay 0.05 \
  --warmup_epochs 15 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --aa rand-m9-mstd0.5-inc1 \
  --reprob 0.25
```

After training, the best checkpoint is saved as:

```text
./output_dir/15M_FP32/checkpoint-best.pth
```

In our experiment, the full-precision model achieved approximately:

```text
Top-1 Accuracy: 78.7%
Top-5 Accuracy: 94.4%
```

## 6. Full-Precision Evaluation

To evaluate the full-precision checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
  --model spikformer_8_15M_CAFormer \
  --data_path /data/dataset/ImageNet \
  --output_dir ./output_dir/15M_FP32_eval \
  --finetune ./output_dir/15M_FP32/checkpoint-best.pth \
  --eval \
  --batch_size 128
```

## 7. PTQ Evaluation

The PTQ pipeline loads a trained full-precision checkpoint, calibrates quantization parameters, assigns bit-widths, and evaluates the quantized model.

### 7.1 W8 Sanity Check

Before low-bit PTQ, first verify that 8-bit PTQ does not significantly degrade accuracy.

```bash
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
  --model spikformer_8_15M_CAFormer \
  --data_path /data/dataset/ImageNet \
  --output_dir ./output_dir/15M_PTQ_W8 \
  --finetune ./output_dir/15M_FP32/checkpoint-best.pth \
  --eval \
  --batch_size 128 \
  --mixup 0 \
  --cutmix 0 \
  --reprob 0 \
  --aa "" \
  --ptq_enable \
  --ptq_calibrate \
  --ptq_candidate_bits 2 3 4 6 8 \
  --ptq_target_avg_bit 8.0 \
  --ptq_fisher_batches 32 \
  --ptq_rescaw_gamma_mode channel_maxabs \
  --ptq_rescaw_clip_value 1.0 \
  --ptq_grid_multipliers 1.0 \
  --ptq_save_mapping ./output_dir/15M_PTQ_W8/ptq_mapping.pth
```

The expected result should be close to the FP32 accuracy.

### 7.2 Mixed-Precision PTQ

Example for average 4-bit PTQ:

```bash
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
  --model spikformer_8_15M_CAFormer \
  --data_path /data/dataset/ImageNet \
  --output_dir ./output_dir/15M_PTQ_AVG4 \
  --finetune ./output_dir/15M_FP32/checkpoint-best.pth \
  --eval \
  --batch_size 128 \
  --mixup 0 \
  --cutmix 0 \
  --reprob 0 \
  --aa "" \
  --ptq_enable \
  --ptq_calibrate \
  --ptq_candidate_bits 2 3 4 6 8 \
  --ptq_target_avg_bit 4.0 \
  --ptq_fisher_batches 32 \
  --ptq_rescaw_gamma_mode channel_maxabs \
  --ptq_rescaw_clip_value 1.0 \
  --ptq_grid_multipliers 1.0 \
  --ptq_save_mapping ./output_dir/15M_PTQ_AVG4/ptq_mapping.pth
```

Note: Low-bit pure PTQ can be sensitive for SNN Transformers. If average 4-bit PTQ causes large accuracy degradation, quantization-aware recovery training may be required.

## 8. TG-SRS Structured Pruning

TG-SRS estimates channel importance using spatiotemporal activation information and task-gradient information.

The pruning score includes:

* Singular-value-based structural score
* Effective-rank score
* Temporal dynamic score
* Activation-gradient task score
* Activity-cost penalty

Example pruning and fine-tuning command:

```bash
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
  --model spikformer_8_15M_CAFormer \
  --data_path /data/dataset/ImageNet \
  --output_dir ./output_dir/15M_TGSRS_P20 \
  --finetune ./output_dir/15M_FP32/checkpoint-best.pth \
  --batch_size 128 \
  --epochs 30 \
  --blr 1e-4 \
  --min_lr 1e-6 \
  --warmup_epochs 3 \
  --mixup 0.2 \
  --cutmix 0.0 \
  --reprob 0.0 \
  --aa "" \
  --auto_prune_tgsrs \
  --tgsrs_rank_batches 8 \
  --tgsrs_hook_group_size 4 \
  --tgsrs_rank_dir ./rank_tgsrs_p20 \
  --tgsrs_compress_rate "[0.20]*999" \
  --tgsrs_bit_protect_omega 0.0
```

For debugging, use fewer batches:

```bash
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
  --model spikformer_8_15M_CAFormer \
  --data_path /data/dataset/ImageNet \
  --output_dir ./output_dir/15M_TGSRS_DEBUG \
  --finetune ./output_dir/15M_FP32/checkpoint-best.pth \
  --batch_size 64 \
  --epochs 1 \
  --blr 1e-4 \
  --mixup 0.0 \
  --cutmix 0.0 \
  --reprob 0.0 \
  --aa "" \
  --auto_prune_tgsrs \
  --tgsrs_rank_batches 1 \
  --tgsrs_hook_group_size 1 \
  --tgsrs_rank_dir ./rank_tgsrs_debug \
  --tgsrs_compress_rate "[0.20]*999" \
  --tgsrs_bit_protect_omega 0.0
```

## 9. Important Notes

### 9.1 Do Not Upload Large Files

The following files should not be uploaded to GitHub:

```text
*.pth
*.pt
*.ckpt
output_dir/
log_dir/
rank_tgsrs*/
rank_conv*/
datasets/
ImageNet/
```

Please make sure these paths are included in `.gitignore`.

### 9.2 PTQ and Pruning Should Be Tested Separately

It is recommended to test the compression pipeline in the following order:

```text
FP32 training
↓
FP32 evaluation
↓
PTQ W8 sanity check
↓
Mixed-precision PTQ
```

or:

```text
FP32 training
↓
FP32 evaluation
↓
TG-SRS pruning
↓
Pruned model fine-tuning
```

Do not combine PTQ and pruning before each module is individually verified.

## 10. Current Experimental Results

| Model                        |     Method | Avg Bit |    Top-1 Acc. |    Top-5 Acc. |
| ---------------------------- | ---------: | ------: | ------------: | ------------: |
| Spike-driven Transformer 15M |       FP32 |      32 |          78.7 |          94.4 |
| Spike-driven Transformer 15M |     PTQ W8 |       8 |          78.7 |          94.4 |
| Spike-driven Transformer 15M |   PTQ Avg4 |       4 | To be updated | To be updated |
| Spike-driven Transformer 15M | TG-SRS P20 |    FP32 | To be updated | To be updated |

Please update this table with final reproduced results.

## 11. Citation

If you use this code or find it helpful, please cite the related works:

```bibtex
@article{spikedriventransformer,
  title={Spike-driven Transformer},
  author={},
  journal={},
  year={}
}
```

```bibtex
@article{qpsnn,
  title={QP-SNN: Quantized and Pruned Spiking Neural Networks},
  author={},
  journal={},
  year={}
}
```

The BibTeX information should be completed according to the final referenced papers.

## 12. License

This repository is released for academic research purposes. Please check the licenses of the original Spike-driven Transformer code and related third-party dependencies before redistribution.
# PTQ
