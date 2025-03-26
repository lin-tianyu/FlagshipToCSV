
# export nnUNet_raw="/projects/bodymaps/Tianyu/nnunet/Dataset_raw"
# export nnUNet_preprocessed="/projects/bodymaps/Tianyu/nnunet/Dataset_preprocessed"
# export nnUNet_results="/projects/bodymaps/Tianyu/nnunet/Dataset_results"

DATA_PATH="/projects/bodymaps/Tianyu/nnunet/Dataset_raw/Dataset802_UCSF/imagesTs"
SAVE_PATH="/projects/bodymaps/Tianyu/nnunet/Dataset_raw/Dataset802_UCSF/flagshipEval"
CKPT_PATH="/projects/bodymaps/Tianyu/FlagshipToCSV/FlagshipModelCKPT"


INPUT_CSV_PATH="/projects/bodymaps/Tianyu/train/input_csv/UCSF-Test-Normal-AllOrgans.csv"
OUTPUT_CSV_PATH="/projects/bodymaps/Tianyu/train/output_csv/UCSF-Test-Normal-AllOrgans.csv"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict_from_modelfolder \
    -i $DATA_PATH \
    -o $SAVE_PATH \
    -m $CKPT_PATH \
    -f all \
    --input_csv $INPUT_CSV_PATH \
    --output_csv $OUTPUT_CSV_PATH \
    --continue_prediction \
    --save_probabilities \
    -npp 6 \
    -nps 6 \
	-num_parts 1 \
	-part_id 0
    