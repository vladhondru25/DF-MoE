#!/bin/bash

# Define the directory to search (default to current directory if not provided)
SEARCH_DIR=${1:-"."}
echo $SEARCH_DIR
CHECKPOINT="/home/galadriel/projects/BiodeepDetection/model_checkpoints/last_moe_multiple_datasets_includes_social_media_tuned_equalized.pt"

# Find all .mp4 files (case-insensitive) and loop through them
# -print0 and IFS= are used to safely handle filenames with spaces
find "$SEARCH_DIR" -type f -iname "*.mp4" -print0 | while IFS= read -r -d '' video; do
    echo "------------------------------------------------"
    echo "Processing: $video"
    
    # Run your inference command
    python inference_video.py --video_path "$video" --checkpoint "$CHECKPOINT"
    
    # Optional: Check if the command failed and exit if it did
    if [ $? -ne 0 ]; then
        echo "Error encountered on $video. Skipping to next..."
    fi
done

echo "Done!"