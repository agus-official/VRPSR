accelerate launch --num_processes=1 --gpu_ids="0" --main_process_port 29300 src/inference_post.py \
    --de_net_path="checkpoints/de_net.pth" \
    --pretrained_sr_path="checkpoints/pre_sr.pkl" \
    --pretrained_post_path="checkpoints/post_sr.pkl" \
    --output_dir="result_post" \
    --output_combined_dir="result_post_collage" \
    --ref_path="datasets/kodak" \
    --fixed_codec_type="libx264" \
    --fixed_qp="33"