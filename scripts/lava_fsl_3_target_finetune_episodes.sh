gpu=0;
output=/home/fsl_experiments/mscoco;

CUDA_VISIBLE_DEVICES=${gpu} \
python -m fewshot_runner \
    --dataset_name mscoco \
    --meta_model_weights /home/lava_output/imagenet_metadataset_train_language_pretraining_vit_s_16/checkpoint.pth \
    --meta_output_dir ${output} \
    --num_episodes 600