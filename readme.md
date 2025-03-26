# ePAI

## 0. Create a virtual environment

<details>
<summary style="margin-left: 25px;">[Optional] Install Anaconda on Linux</summary>
<div style="margin-left: 25px;">

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p ./anaconda3
./anaconda3/bin/conda init
source ~/.bashrc
```

</div>
</details>


```bash
conda create -n ePAI python=3.10 -y
```

## 1. Installation

```bash
source activate ePAI
pip install --upgrade setuptools packaging
pip install nnunetv2
pip install -e .
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

## 2. Train ePAI

#### 2.1 Generate json file in the path of raw data

```bash
python -W ignore generate_json.py --raw_dir /mnt/bodymaps/ePAI/nnUNet/raw --dataset_name Dataset1013_ePAI_3MM 
```

Modify `vi nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`

```python
### Some hyperparameters for you to fiddle with
self.initial_lr = 8e-2
self.weight_decay = 3e-5
self.oversample_foreground_percent = 0.5
self.probabilistic_oversampling = False
self.num_iterations_per_epoch = 40
self.num_val_iterations_per_epoch = 50
self.num_epochs = 1000
self.current_epoch = 0
self.enable_deep_supervision = True
```

Modify `Dataset1013_ePAI_3MM/nnUNetPlans.json`

```python
"batch_size": 16, # 47,276 MiB
```

```bash
bash nnunetv2_25cls.sh
```

Note: readme about the script.

```bash
export nnUNet_N_proc_DA=36 # number of CPU cores
export nnUNet_raw="/mnt/bodymaps/ePAI/nnUNet/raw"
export nnUNet_preprocessed="/mnt/bodymaps/ePAI/nnUNet/preprocessed"
export nnUNet_results="/mnt/realccvl15/zzhou82/project/OncoKit/ePAI/train/runsv2"

nnUNetv2_plan_and_preprocess -d 1013 -npfp 64 -np 64 -c 3d_fullres
# npfp and np are CPU cores used for preprocessing

CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train 1013 3d_fullres all -num_gpus 4
```

## 3. Inference ePAI

#### 3.1 Organize Dataset from BDMAP format to nnUNet format

Assume the original CT data is under BDMAP format:

```bash
path_to_bdmap_format_data/
│── BDMAP_0000001/
│    └── ct.nii.gz
│── BDMAP_0000002/
│    └── ct.nii.gz
└── ...
```

Then, to convert it into a nnUNet input folder, run

```bash
python softlink2nnUNet.py \
  --input_bdmap_path="path_to_bdmap_format_data/" \
  --outout_nnunet_path="path_to_nnunet_format_data/"
```

This will create softlinks of the BDMAP format data into nnUNet format:

```bash
path_to_nnunet_format_data/
│── BDMAP_0000001_0000.nii.gz
│── BDMAP_0000002_0000.nii.gz
└── ...
```

Please use this nnUNet format data folder for the inference process below.

#### 3.2 Setup Inference Configurations

Modify 5 input arguments in `inference.sh`

1. DATA_PATH: an nnUNet format CT input path (the result folder from step 3.1 above)
2. SAVE_PATH: the path to save segmentation predictions
3. CKPT_PATH: path to the Flagship Model checkpoint
4. INPUT_CSV_PATH: path to input csv with `Original ID` and `BDMAP ID` columns.
5. OUTPUT_CSV_PATH: path to output csv file

#### 3.3 Run the inference process

```bash
sh inference.sh
```

The output would contains two types:

1. The segmentation predictions under SAVE_PATH
2. The CSV output under OUTPUT_CSV_PATH
