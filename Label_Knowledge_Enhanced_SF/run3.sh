#!/bin/bash
#$ -S /bin/bash
#here you'd best to change testjob as username
#$ -N lyh
#cwd define the work environment,files(username.o) will generate here
#$ -cwd
#$ -l h=gpu05
# merge stdo and stde to one file
#$ -j y
# export LD_LIBRARY_PATH=/Work21/2021/liyuhang/system
# export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64
export HF_HOME=/Work21/2021/liyuhang/huggingface_cache

echo "job start time: `date`"

# tgt_domains="RateBook PlayMusic BookRestaurant"
# tgt_domains="SearchScreeningEvent GetWeather SearchCreativeWork"
# tgt_domains="AddToPlaylist RateBook PlayMusic BookRestaurant SearchScreeningEvent GetWeather SearchCreativeWork"
# tgt_domains="PlayMusic BookRestaurant SearchScreeningEvent GetWeather SearchCreativeWork"
# tgt_domains="BookRestaurant SearchScreeningEvent GetWeather SearchCreativeWork"
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
        --num_epochs 24 \
        --use_gpu \
        --target_domain $tgt_domain \
        --pretrained_model bert-large-uncased \
        --n_samples $n \
        --num_workers 4 \
        --context_max_len 32 \
        --label_max_len 16 \
        --warmup_rate 0.0 \
        --early_stop 20 \
        --lr 1e-5 \
        --model_dir model_dir3 \
        --log_dir log_dir3
    done
done

echo "job end time:`date`"