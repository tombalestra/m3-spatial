CUDA_VISIBLE_DEVICES=7 python -m lmm.seem.eval_visual_grounding \
-s /data/xueyanz/data/3dgs/playroom \
--model_path /data/xueyanz/output/mmm/ckpt/playroom/playroom_bsz7_gpu7_embTrue_clipTrue_sigTrue_dinoTrue_seemTrue_llaTrue_llvTrue_dim160_temp0.05_baseline/run_0002 \
--preload_dataset_to_gpu_threshold 0 \
--local_sampling \
--annot_name seem_info.json \
--skip_train \
--render \
--use_emb