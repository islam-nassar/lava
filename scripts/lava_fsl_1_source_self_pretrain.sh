num_gpus=8;
data=/home/data/imagenet_metadataset_train/;
output=/home/lava_output/imagenet_metadataset_train_self_pretraining_vit_s_16;

python -m torch.distributed.launch --nproc_per_node=${num_gpus} main_lava.py \
    --arch vit_small \
    --epochs 800 \
    --batch_size_per_gpu 128 \
    --norm_last_layer true \
    --freeze_last_layer 1 \
    --data_path ${data} \
    --output_dir ${output} \
    --start_pl_epoch -1 \
    --lam_dino 1 \
    --lam_pl 0 \
    --lam_sup 0 \
    --lam_sem 0 \
    --teacher_temp 0.07 \
    --warmup_teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --local_crops_number 10 \
    --momentum_teacher 0.996 \
    --eval_linear false \
    --eval_freq 5
