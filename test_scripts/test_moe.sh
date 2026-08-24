wandb login WANDB_TOKEN
hf auth login --token HF_TOKEN

torchrun --nproc_per_node=2 test_scripts/test_moe.py --wandb_run_name moe_multiple_datasets_includes_social_media_tuned_equalized_test \
 --checkpoint_path /home/galadriel/projects/BiodeepDetection/model_checkpoints/last_moe_multiple_datasets_includes_social_media_tuned_equalized.pt \
  --num_experts 6 --top_k 2 --num_workers 12 \
   --save_path  predictions/moe_multiple_datasets_includes_social_media_tuned_equalized_test \
    --path_features_video /mnt/data/datasets/features_social_media_test/video \
     --path_features_audio /mnt/data/datasets/features_social_media_test/audio --batch_size 2