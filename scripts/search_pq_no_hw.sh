#!/usr/bin/env bash

nohup python -u ../tools/search_policy.py \
  --model yolov8n \
  --ckpt_load_path /export/home/yliu/galen-main/scripts/results/checkpoints/yolov8n.pt \
  --log_dir ./logs \
  --agent pruning-quantization-agent \
  --episodes 100 \
  --add_search_identifier exp_01c_pq \
  --alg_config num_workers=20 reward=r6 mixed_reference_bits=6 reward_target_cost_ratio=0.5 enable_latency_eval=False reward_episode_cost_key=BOPs data_set=tick\
  --wandb_project 'model_cpression' --wandb_entity 'pg_search'  >log.out 2>&1 &

job_pid=$!

echo "PID of the job: $job_pid"
