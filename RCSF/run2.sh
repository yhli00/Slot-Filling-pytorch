#!/bin/bash
#$ -S /bin/bash
#here you'd best to change testjob as username
#$ -N lyh
#cwd define the work environment,files(username.o) will generate here
#$ -cwd
#$ -l h=gpu05
# merge stdo and stde to one file
#$ -j y

export HF_HOME=/Work21/2021/liyuhang/huggingface_cache

echo "job start time: `date`"

# tgt_domains="AddToPlaylist RateBook PlayMusic BookRestaurant SearchScreeningEvent GetWeather SearchCreativeWork"
tgt_domains="AddToPlaylist"
n_samples=(0)

for tgt_domain in ${tgt_domains[@]}
do
    for n in ${n_samples[@]}
    do
        CUDA_VISIBLE_DEVICES=2 /Work21/2021/liyuhang/envs/py3.7/BERT_TAGGER/bin/python main.py \
        --do_train \
        --do_test \
        --batch_size 16 \
        --num_epochs 64 \
        --target_domain $tgt_domain \
        --pretrained_model bert-large-uncased \
        --n_samples $n \
        --num_workers 0 \
        --model_dir ../model_dir/RCSF \
        --log_dir ../log_dir/RCSF \
        --context_max_len 32 \
        --query_max_len 32 \
        --lr 1e-5 \
        --early_stop 5
    done
done

echo "job end time:`date`"
