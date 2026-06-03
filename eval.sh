OUT_DIR=./output_dir/15M_PTQ_AVG4

CUDA_VISIBLE_DEVICES=1 python main_finetune.py \
  --model spikformer_8_15M_CAFormer \
  --data_path /data/dataset/ImageNet \
  --output_dir ${OUT_DIR} \
  --finetune /data1/users/zhengw/QSD-Transformer/classification/output_dir/15M_FP32/checkpoint-best.pth \
  --eval \
  --batch_size 32 \
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
  --ptq_use_logit_sensitivity \
  --ptq_sensitivity_batches 8 \
  --ptq_sensitivity_temperature 2.0 \
  --ptq_save_mapping ${OUT_DIR}/ptq_mapping.pth \
  2>&1 | tee -a ${OUT_DIR}/eval.log