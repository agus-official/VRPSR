accelerate launch --num_processes=1 --gpu_ids="0" --main_process_port 29300 src/inference_sim.py \
    --de_net_path="checkpoints/de_net.pth" \
    --pretrained_path="checkpoints/simulator.pkl" \
    --output_dir="result_sim/sim" \
    --rec_save_dir="result_sim/real" \
    --ref_path="datasets/kodak" \
    --fixed_codec_type="libx264" \
    --fixed_qp="33"
