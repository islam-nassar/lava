gpu=0;
dataset=mscoco
model=/home/lava_output/imagenet_metadataset_train_language_pretraining_vit_s_16/checkpoint.pth
output=/home/fsl_experiments/mscoco;

CUDA_VISIBLE_DEVICES=${gpu} \
python -m fewshot_runner \
    --dataset_name ${dataset} \
    --meta_model_weights ${model} \
    --meta_output_dir ${output} \
    --num_episodes 600