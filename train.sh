CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
  main_finetune.py \
  --batch_size 128 \
  --blr 6e-4 \
  --warmup_epochs 15 \
  --epochs 30 \
  --model spikformer_8_15M_CAFormer \
  --data_set IMNET \
  --nb_classes 1000 \
  --data_path /data/dataset/ImageNet \
  --output_dir /data1/users/zhengw/QSD-Transformer/classification/output_dir/15M_FP32 \
  --log_dir /data1/users/zhengw/QSD-Transformer/classification/log_dir/15M_FP32 \
  --finetune /data1/users/zhengw/QSD-Transformer/classification/output_dir/15M_FP32/checkpoint-best.pth \
  --dist_eval \
  --num_workers 4 \
  --mixup 0 \
  --cutmix 0 \
  --smoothing 0.1 \
  2>&1 | tee -a /data1/users/zhengw/QSD-Transformer/classification/log_dir/15M_FP32/train.log
