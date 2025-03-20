CUDA_VISIBLE_DEVICES=7 python -m lmm.siglip.eval_retrieval \
-s /data/xueyanz/data/3dgs/train \
--model_path /data/xueyanz/output/mmm/ckpt/train/train_bsz7_gpu7_embTrue_clipTrue_sigTrue_dinoTrue_seemTrue_llaTrue_llvTrue_dim160_temp0.05_siglip_image/run_0000 \
--preload_dataset_to_gpu_threshold 0 \
--local_sampling \
--annot_name cap_info_semsaml2_test.json \
--coco_info /data/xueyanz/data/3dgs/coco/siglip_info.json \
--skip_train \
--render \
--use_emb