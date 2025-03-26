export nnUNet_N_proc_DA=36
export nnUNet_raw="/data/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0003.datasets-nnunetv2-raw/nnUNet_raw_data"
export nnUNet_preprocessed="/data/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0004.datasets-nnunetv2-preprocessed"
export nnUNet_results="./runsv2"

nnUNetv2_plan_and_preprocess -d 1013 -npfp 64 -np 64 -c 3d_fullres

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1013 3d_fullres all --use_compressed