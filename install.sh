# Create a new Conda environment named nnunet
conda create -n nnunet

# Activate the nnunet environment
conda activate nnunet

# Install PyTorch compatible with the CUDA version
conda install \
    -y \
    --strict-channel-priority \
    --solver libmamba \
    -c pytorch \
    -c nvidia \
    -c conda-forge \
    "python>=3.11,<3.12" \
    pytorch \
    pytorch-cuda=11.8 \
    torchvision

# Install other necessary libraries
conda install \
    -y \
    -c https://mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge \
    --solver libmamba \
    "SimpleITK<2.4.0" \
    cython \
    matplotlib \
    numpy \
    opencv \
    tqdm

# Install nnunetv2 using pip
pip install nnunetv2
