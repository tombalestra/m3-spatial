CUDA_VISIBLE_DEVICES=2 python3 -m xy_utils.save_test_camera \
--source_path /data/xueyanz/data/3dgs/playroom \
--model_path /data/xueyanz/output/mmm/ckpt/train/train_bsz8_gpu8_embTrue_clipTrue_sigTrue_dinoTrue_seemTrue_llaTrue_llvTrue_dim160_temp0.05_baseline/run_0000 \
--preload_dataset_to_gpu_threshold 0 \
--local_sampling \
--skip_train \
--render