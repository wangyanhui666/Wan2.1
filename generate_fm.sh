CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    generate.py \
    --task t2v-1.3B \
    --size 832*480 \
    --ulysses_size 4 \
    --ckpt_dir /mnt/shangcephfs/wangyanhui/distillation/Wan2.1-T2V-1.3B \
    --dit_path "/mnt/shangcephfs/wangyanhui/distillation/Wan2.1-T2V-1.3B" \
    --sample_solver 'flow_matching' \
    --offload_model False \
    --sample_shift 5.0 \
    --sample_guide_scale 6 \
    --base_seed 42 \
    --prompt "Waves crash against a rugged shoreline, creating a rhythmic and soothing sound as they break against the rocks. The rocks are jagged and uneven, with a rich brown color, partially covered in white foam from the waves. The water is a deep blue-green color, reflecting the sunlight and creating a sparkling effect on the surface. The camera is positioned at a distance, providing a wide view of the shoreline and the vast expanse of the ocean." \
    --save_dir /mnt/shangcephfs/wangyanhui/distillation/results/0612_Wan2.1-T2V-1.3B_shift5_original_codebase_debuginference \
    --sample_steps 100 \