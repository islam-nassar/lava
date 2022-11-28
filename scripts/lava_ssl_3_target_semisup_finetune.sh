num_gpus=2;
data=/home/data/clipart_4shot/;
output=/home/lava_output/clipart_sup_finetuning_4shot;

python -m torch.distributed.launch --nproc_per_node=${num_gpus} main_lava.py \
    --arch vit_small \
    --epochs 20 \
    --batch_size_per_gpu 128 \
    --pretrained_weights /home/lava_output/clipart_self_finetuning_from_imagenet/checkpoint.pth \
    --load_backbone_only True \
    --data_path ${data} \
    --output_dir ${output} \
    --start_pl_epoch 0 \
    --lam_dino 1 \
    --lam_pl 1 \
    --lam_sup 1 \
    --lam_sem 9 \
    --lr_scaler 2 \
    --use_sentence_transformer true \
    --transformer_language_model 'paraphrase-mpnet-base-v2' \
    --teacher_temp 0.07 \
    --warmup_teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 20 \
    --momentum_teacher 0.996 \
    --eval_linear true \
    --eval_freq 5
