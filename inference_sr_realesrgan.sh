accelerate launch --num_processes=1 --gpu_ids="0" --main_process_port 29300 src/inference_sr.py \
    --de_net_path="checkpoints/de_net.pth" \
    --ref_path="datasets/kodak" \
    --fixed_codec_type="libx264" \
    --fixed_qp="32" \
    --pretrained_sr_path="checkpoints/VRPSR_realesrgan.pkl" \
    --model="realesrgan" \
    --output_dir="result_sr_realesrgan" \
    --output_combined_dir="result_sr_realesrgan_collage"