CUDA_VISIBLE_DEVICES=2 python -m lmm.llama.infer_grounding \
-s /disk1/data/m3/data_v2/garden \
--model_path /disk1/checkpoint/mmm/garden_bsz1_gpu1_embTrue_clipFalse_sigFalse_dinoFalse_seemFalse_llaTrue_llvFalse_dim32_temp0.05_debug/run_0001 \
--preload_dataset_to_gpu_threshold 0 \
--local_sampling \
--render \
--use_emb \
--skip_train \
--text "wooden table top" \
--index 10