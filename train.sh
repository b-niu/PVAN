# Activate the nnunet environment
conda activate nnunet

# Set the root directory for the project
export root_dir="/home/user/experiments/learning/"

# Set the directory for raw nnUNet data
export nnUNet_raw="${root_dir}/nnUNet_raw"

# Set the directory for preprocessed nnUNet data
export nnUNet_preprocessed="${root_dir}/nnUNet_preprocessed"

# Set the directory for nnUNet results
export nnUNet_results="${root_dir}/nnUNet_results"

# Assume the project folder name is Dataset201_VAN

# Plan and preprocess the dataset with ID 201, verify dataset integrity, and use 3D full resolution
nnUNetv2_plan_and_preprocess -d 201 --verify_dataset_integrity -c 3d_fullres

# Modify the splits_final.json file under ${nnUNet_preprocessed}/Dataset201_VAN to fix the 5-fold cross-validation split

# Modify the nnUNetPlans.json file under ${nnUNet_preprocessed}/Dataset201_VAN to change the patch size parameter

# Training
# Use GPU 0 to train with 3D full resolution and cube size 128
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 201 3d_fullres_cube128 3

# Use GPU 1 to train with 3D full resolution and cube size 64
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 201 3d_fullres_cube64 3
