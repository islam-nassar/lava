num_gpus=2;
data=/home/data/clipart/;
model=/home/lava_output/imagenet_self_pretraining_vit_s_16/checkpoint.pth;
output=/home/lava_output/clipart_self_finetuning_from_imagenet;

python -m torch.distributed.launch --nproc_per_node=${num_gpus} main_lava.py \
    --arch vit_small \
    --epochs 50 \
    --batch_size_per_gpu 128 \
    --pretrained_weights ${model} \
    --load_backbone_only False \
    --data_path ${data} \
    --output_dir ${output} \
    --start_pl_epoch -1 \
    --lam_dino 1 \
    --lam_pl 0 \
    --lam_sup 0 \
    --lam_sem 0 \
    --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 15 \
    --momentum_teacher 0.95 \
    --eval_linear false \
    --eval_freq 5
