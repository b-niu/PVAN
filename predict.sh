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

# Use GPU 3 to predict with 3D full resolution and cube size 128, fold 3
CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -d 201 -c 3d_fullres_cube128 -f 3 \
    -i /home/user/experiments/learning/nnUNet_raw/Dataset201_VAN/imagesTs/ \
    -o /home/user/experiments/learning/for_test/predictions/testset_by_ds201_3dfull_128_fold3/

# Use GPU 3 to predict with 3D full resolution and cube size 64, fold 3
CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -d 201 -c 3d_fullres_cube64 -f 3 \
    -i /home/user/experiments/learning/nnUNet_raw/Dataset201_VAN/imagesTs/ \
    -o /home/user/experiments/learning/for_test/predictions/testset_by_ds201_3dfull_64_fold3/
