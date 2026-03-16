export CUDA_VISIBLE_DEVICES=2


export PYTHONPATH="/mnt/data/x2robot_v2/yangping/github/jerry_git/wall-x:${PYTHONPATH}"

# ckpt=/x2robot_v2/share/yangping/ckpt/benchmark/parcel_sorting/pos-xpred-0119/5_520000
# ckpt=/mnt/data/x2robot_v2/yangping/ckpt/benchmark/parcel_sorting-pos-xpred-1230/9_520000

# ckpt=/x2robot_v2/share/yangping/ckpt/robochallenge_github/benchmark/pick_up_cups/bus2507_plus_mot-0128/40
ckpt=/x2robot_v2/share/bus2602/pretrain_vq_delta_6d_eef_xloss_448_0214/ckpt_ddp_1/0_220000


python -m wall_x.serving.launch_serving \
  --env ALOHA \
  --port 32157 \
  model-config:model-config \
  --model-config.model-path $ckpt \
  --model-config.action-tokenizer-path /x2robot_v2/Models/fast \
  --model-config.train-config-path ${ckpt}/config.yml \
  --model-config.action-dim 26 \
  --model-config.state-dim 26 \
  --model-config.pred-horizon 32 \
  --model-config.camera-key face_view left_wrist_view right_wrist_view