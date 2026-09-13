#!/bin/bash
# =============================================================================
# Phase A — single-suite GRPO + LoRA expert (libero_10)
# =============================================================================
# Goal: train ONE LoRA expert with GRPO on libero_10, warm-started from the
# trajall SFT checkpoint. Produces a `lora_adapter/` (the atom for the MoE plan).
#
# Based on examples/run_openvla_oft_rl_libero.sh (Apache-2.0, PRIME-RL/SimpleVLA-RL),
# modified: LoRA enabled, target_modules restricted to LLaMA control layers,
# single-node/low-GPU friendly, wandb off, KL off (SimpleVLA recipe).
#
# EDIT the 3 paths marked  <<< EDIT  before running on the pod.
# Run from /opt/SimpleVLA-RL:  bash examples/run_phaseA_lora.sh
# =============================================================================
set -x

export NCCL_DEBUG=WARN
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export MUJOCO_GL=egl
export NVIDIA_DRIVER_CAPABILITIES=all
unset PYOPENGL_PLATFORM
export ROBOT_PLATFORM=LIBERO

PROJECT_NAME='VLA-MoE-LoRA'
EXPERIMENT_NAME='phaseA_libero10_lora_r32_grpo'

SFT_MODEL_PATH="/workspace/ckpt_libero10_trajall"          # <<< EDIT (warm-start ckpt)
CKPT_PATH="/workspace/phaseA_out"                          # <<< EDIT (where to save)
ALIGN_PATH="/opt/SimpleVLA-RL/align.json"                  # <<< EDIT if different

DATASET_NAME="libero_10"
VLA_NAME="openvla-oft"

# ---- GPU scale -------------------------------------------------------------
# LoRA is light; start with 1-2 GPUs. Set to the # of visible GPUs.
NUM_GPUS=1
NUM_NODES=1
# ppo_micro_batch_size is PER-GPU; with 1 GPU keep it small to fit memory.
PPO_MINI_BS=16
PPO_MICRO_BS=1
# batch sizes: keep small for a first single-GPU run (raise once it fits)
TRAIN_BS=8
VAL_BS=100          # 10 tasks x 10 val trials (see num_trials_per_task below)
VAL_MICRO_BS=10     # must divide VAL_BS (avoids the chunk() assertion)

bash examples/overwrite_vla_ckpt_utils.sh $SFT_MODEL_PATH

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=10 \
    data.n_samples=8 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=$TRAIN_BS \
    data.val_batch_size=$VAL_BS \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BS \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=$VAL_MICRO_BS \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    trainer.val_only=False \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=offline \
    trainer.val_before_train=True
