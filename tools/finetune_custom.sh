#!/bin/bash

export NCCL_P2P_LEVEL=NVL
 
MASTER_PORT=14156
NUM_GPUS=1

CLS_COEFF=5
COO_COEFF=20
SEM_COEFF=1
SEQ_LEN=512
NUM_BINS=32
CONVERTER=v3
BATCH_SIZE=8
EPOCHS=500

DATA=/path/to/data
JOB=job_name_1
PRETRAIN=/path/to/pretrain/checkpoint.pth
OUTPUT_DIR=/output/dir/

WANDB_MODE=online torchrun --nproc_per_node=1 main_ddp.py --dataset_name=cubicasa \
		       --dataset_root=${DATA} \
		       --semantic_classes=12 \
		       --job_name=${JOB} \
		       --batch_size ${BATCH_SIZE} \
		       --input_channels=3 \
		       --output_dir ${OUTPUT_DIR} \
		       --poly2seq \
		       --seq_len ${SEQ_LEN} \
		       --num_bins ${NUM_BINS} \
		       --disable_eval \
		       --label_smoothing 0.1 \
		       --epochs ${EPOCHS} \
		       --lr_drop '' \
		       --cls_loss_coef ${CLS_COEFF} \
		       --coords_loss_coef ${COO_COEFF} \
		       --room_cls_loss_coef ${SEM_COEFF} \
		       --resume ${OUTPUT_DIR}/${JOB}/checkpoint.pth \
		       --ema4eval \
		       --disable_poly_refine \
		       --dec_attn_concat_src \
		       --per_token_sem_loss \
		       --jointly_train \
		       --converter_version ${CONVERTER} \
		       --use_anchor \
		       --start_from_checkpoint ${PRETRAIN}
