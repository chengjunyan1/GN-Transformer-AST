#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

SRC_DIR=../..
DATA_DIR=${SRC_DIR}/data
MODEL_DIR=${SRC_DIR}/models

make_dir $MODEL_DIR

DATASET=python
CODE_EXTENSION=original_subtoken
JAVADOC_EXTENSION=original
ADDDATA_EXTENSION=adddata


function train () {

echo "============TRAINING============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/train.py \
--data_workers 5 \
--random_seed 9970 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--train_src train/code.${CODE_EXTENSION} \
--train_tgt train/javadoc.${JAVADOC_EXTENSION} \
--train_ast train/syntax.${ADDDATA_EXTENSION} \
--dev_src dev/code.${CODE_EXTENSION} \
--dev_tgt dev/javadoc.${JAVADOC_EXTENSION} \
--dev_ast dev/syntax.${ADDDATA_EXTENSION} \
--uncase True \
--max_src_len 400 \
--max_tgt_len 30 \
--emsize 512 \
--fix_embeddings False \
--src_vocab_size 50000 \
--tgt_vocab_size 30000 \
--share_decoder_embeddings True \
--max_examples -1 \
--batch_size 30 \
--test_batch_size 60 \
--num_epochs 200 \
--num_head 8 \
--d_k 64 \
--d_v 64 \
--d_ff 2048 \
--src_pos_emb False \
--tgt_pos_emb True \
--nlayers 6 \
--trans_drop 0.2 \
--dropout_emb 0.2 \
--dropout 0.2 \
--early_stop 20 \
--warmup_steps 0 \
--optimizer adam \
--learning_rate 0.0001 \
--lr_decay 0.99 \
--valid_metric bleu \
--checkpoint True \
--use_word_type False \
--use_word_fc False \
--use_dense_connection False \
--add_top_down_edges True \
--add_bottom_up_edges True \
--use_rpe False \
--rpe_size 512 \
--rpe_m 6 \
--rpe_c 0.4 \
--rpe_layer 2 \
--rpe_approx 0 \
--rpe_share_emb False \
--rpe_mode concat \
--rpe_all False \
--copy_attn False \

}


function test () {

echo "============TESTING============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/train.py \
--only_test True \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--dev_src test/code.${CODE_EXTENSION} \
--dev_tgt test/javadoc.${JAVADOC_EXTENSION} \
--dev_ast test/syntax.${ADDDATA_EXTENSION} \
--uncase True \
--max_src_len 400 \
--max_tgt_len 30 \
--max_examples -1 \
--test_batch_size 64 \

}


function beam_search () {

echo "============Beam Search TESTING============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/test.py \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--dev_src test/code.${CODE_EXTENSION} \
--dev_tgt test/javadoc.${JAVADOC_EXTENSION} \
--dev_ast test/syntax.${ADDDATA_EXTENSION} \
--uncase True \
--max_examples -1 \
--max_src_len 400 \
--max_tgt_len 30 \
--test_batch_size 32 \
--beam_size 4 \
--n_best 1 \
--block_ngram_repeat 3 \
--stepwise_penalty False \
--coverage_penalty none \
--length_penalty none \
--beta 0 \
--gamma 0 \
--replace_unk \

}

train $1 $2
test $1 $2
beam_search $1 $2