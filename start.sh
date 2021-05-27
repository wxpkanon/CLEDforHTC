#!/bin/sh
train='train_file'
dev='dev_file'
test='test_file'

train_file="${train}"                                                
dev_file="${dev}"                                      
test_file="${test}"

c1_kb_prebuilt_label_embedding_file="c1_kb_prebuilt_label_embedding_file"
c2_kb_prebuilt_label_embedding_file="c2_kb_prebuilt_label_embedding_file"
c3_kb_prebuilt_label_embedding_file="c3_kb_prebuilt_label_embedding_file"
init_lr=1e-3
dropkeep=0.5
batch_size=512
cluster_num=30
keytop="1k"
lastdim=300
step_group=1000
lr_decay_step=1000
c1_keywords_embeddings_file="c1_keywords_embeddings_file"
c2_keywords_embeddings_file="c2_keywords_embeddings_file"
numof_keywords_percat1=${cluster_num}

embedding_dim=300
let cell_dim="${embedding_dim}/2"
let last_dim_theta="${lastdim}/${cell_dim}"

output_dir="output_dir"

if [ ! -d ${output_dir}/log  ];then
  mkdir -p ${output_dir}/log
else
  echo dir exist: ${output_dir}/log
fi

python3 train.py \
    --epoch_num 200 \
    --filter_num 128 \
    --sentence_len 512 \
    --embedding_dim 300 \
    --cell_dim ${cell_dim} \
    --batch_size ${batch_size}\
    --dropout_keep ${dropkeep} \
    --c1_labels_file c1.label \
    --c2_labels_file c2.label \
    --c3_labels_file c3.label \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --test_file ${test_file} \
    --vocab_file vocab_file \
    --embedding_file embedding_file \
    --c1_kb_label_embeddings_file ${c1_kb_prebuilt_label_embedding_file} \
    --c2_kb_label_embeddings_file ${c2_kb_prebuilt_label_embedding_file} \
    --c3_kb_label_embeddings_file ${c3_kb_prebuilt_label_embedding_file} \
    --output_dir ${output_dir} \
    --eval_every_steps ${step_group} \
    --lr_decay_step ${lr_decay_step} \
    --early_stop_times 1000 \
    --init_lr ${init_lr} \
    --numof_keywords_percat1 ${numof_keywords_percat1} \
    --c1_keywords_embeddings_file ${c1_keywords_embeddings_file} \
    --c2_keywords_embeddings_file ${c2_keywords_embeddings_file} \
    --last_dim_theta ${last_dim_theta} \
    --cluster_num ${cluster_num}\
    > ${output_dir}/log.log 2>&1
