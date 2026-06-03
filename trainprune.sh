OUT_DIR=./output_prune/15M_TGSRS_P20
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  main_finetune.py \
  --model spikformer_8_15M_CAFormer \
  --data_path /data/dataset/ImageNet \
  --output_dir ${OUT_DIR} \
  --log_dir ${OUT_DIR} \
  --finetune /data1/users/zhengw/QSD-Transformer/classification/output_dir/15M_FP32/checkpoint-best.pth \
  --batch_size 64 \
  --epochs 30 \
  --blr 1e-4 \
  --min_lr 1e-6 \
  --warmup_epochs 3 \
  --mixup 0.0 \
  --cutmix 0.0 \
  --reprob 0.0 \
  --aa "" \
  --auto_prune_tgsrs \
  --tgsrs_rank_batches 8 \
  --tgsrs_hook_group_size 4 \
  --tgsrs_rank_dir ./rank_tgsrs_p20 \
  --tgsrs_compress_rate "[0.20]*999" \
  --tgsrs_bit_protect_omega 0.0 \
  2>&1 | tee -a ${OUT_DIR}/prune.log