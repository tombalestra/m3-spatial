CUDA_VISIBLE_DEVICES=7 python -m lmm.clip.eval_grounding \
-s /data/xueyanz/data/3dgs/train \
--model_path /data/xueyanz/output/mmm/ckpt/train/train_bsz8_gpu8_embTrue_clipTrue_sigTrue_dinoTrue_seemTrue_llaTrue_llvTrue_dim160_temp0.05_baseline/run_0000 \
--preload_dataset_to_gpu_threshold 0 \
--local_sampling \
--annot_name vlm_info_semsaml2_test.json \
--skip_train \
--render \
--use_emb