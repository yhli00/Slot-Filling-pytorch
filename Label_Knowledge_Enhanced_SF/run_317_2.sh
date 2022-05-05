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
export HF_HOME=/Work18/2021/liyuhang/huggingface_cache

echo "job start time: `date`"
# tgt_domains="RateBook PlayMusic BookRestaurant"
# tgt_domains="SearchScreeningEvent GetWeather SearchCreativeWork"
# tgt_domains="AddToPlaylist RateBook PlayMusic BookRestaurant SearchScreeningEvent GetWeather SearchCreativeWork"
# tgt_domains="PlayMusic BookRestaurant SearchScreeningEvent GetWeather SearchCreativeWork"
# tgt_domains="AddToPlaylist RateBook PlayMusic"
tgt_domains="BookRestaurant"
n_samples=(0)
message="模型所有dropout都设成0.2,2层self-attention,attention_mask=attention_mask.unsqueeze(-2).repeat(1,1,L1+L2,1)的情况"
for tgt_domain in ${tgt_domains[@]}
do
    for n in ${n_samples[@]}
    do
        CUDA_VISIBLE_DEVICES=1 /Work18/2021/liyuhang/envs/py3.7/BERT_TAGGER/bin/python main.py \
        --do_train \
        --do_test \
        --batch_size 4 \
        --num_epochs 64 \
        --use_gpu \
        --target_domain $tgt_domain \
        --pretrained_model bert-large-uncased \
        --n_samples $n \
        --num_workers 4 \
        --context_max_len 64 \
        --label_max_len 32 \
        --warmup_rate 0.0 \
        --early_stop 20 \
        --lr 1e-5 \
        --model_dir ../model_dir_317/Label_Knowledge_Enhanced_SF \
        --log_dir ../log_dir_317/Label_Knowledge_Enhanced_SF \
        --log_message $message \
        --dropout_rate 0.2
    done
done

echo "job end time:`date`"