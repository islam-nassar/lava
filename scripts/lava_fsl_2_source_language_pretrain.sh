num_gpus=2;
data=data=/home/data/imagenet_metadataset_train/;
output=/home/lava_output/imagenet_metadataset_train_language_pretraining_vit_s_16;

python -m torch.distributed.launch --nproc_per_node=${num_gpus} main_lava.py \
    --arch vit_small \
    --epochs 20 \
    --batch_size_per_gpu 128 \
    --pretrained_weights /home/lava_output/imagenet_metadataset_train_self_pretraining_vit_s_16/checkpoint.pth \
    --load_backbone_only true \
    --freeze_student_backbone true \
    --data_path ${data} \
    --output_dir ${output} \
    --start_pl_epoch -1 \
    --lam_dino 1 \
    --lam_pl 0 \
    --lam_sup 0 \
    --lam_sem 1 \
    --use_sentence_transformer true \
    --transformer_language_model 'paraphrase-mpnet-base-v2' \
    --teacher_temp 0.07 \
    --warmup_teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 20 \
    --momentum_teacher 0.996 \
    --eval_linear true \
    --eval_freq 5
