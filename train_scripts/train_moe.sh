wandb login WANDB_TOKEN
hf auth login --token HF_TOKEN
torchrun --nproc_per_node=2 train_scripts/train_moe.py --epochs 10 \
 --wandb_run_name moe_multiple_datasets_includes_social_media_tuned_equalized \
 --num_experts 6 --top_k 2 \
 --train_datasets avlips,celebdf \
 --batch_size 2 --num_workers 10 --dropout_modalities 0. \
 --ckpt_av_tf /home/galadriel/projects/BiodeepDetection/model_checkpoints/cross_transformer/avlips_mavos_social_media_equalized/epoch_17.pth \
 --ckpt_avff /home/galadriel/projects/BiodeepDetection/model_checkpoints/avff_social_media.pth