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
        CUDA_VISIBLE_DEVICES=0 /Work21/2021/liyuhang/envs/py3.7/BERT_TAGGER/bin/python main.py \
        --do_train \
        --do_test \
        --batch_size 8 \
        --num_epochs 300 \
        --target_domain $tgt_domain \
        --pretrained_model bert-large-uncased \
        --n_samples $n \
        --num_workers 0 \
        --model_dir model_dir2 \
        --log_dir log_dir2 \
        --max_len 128 \
        --early_stop 150
    done
done

echo "job end time:`date`"
