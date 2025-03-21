# M3: 3D-Spatial Multimodal Memory

## TODO
- [x] Installation
- [ ] Dataset
- [ ] Demo
- [ ] Checkpoint
- [ ] Training
- [ ] Inference

## Installation
* Prepare Conda Environment
```sh
conda create --name gs python=3.10
conda activate gs
conda install -c conda-forge cudatoolkit=11.7
conda install -c nvidia/label/cuda-11.7.0 cuda-toolkit
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge gxx_linux-64=11.2.0
conda install -c conda-forge libxcrypt
pip install plyfile tqdm psutil setuptools mkl pandas
pip install --force-reinstall numpy==1.23.5
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
* Download M3 and install submodules
```sh
git clone https://github.com/MaureenZOU/m3-spatial.git
cd submodules/diff-gaussian-rasterization && pip install -e .
cd submodules/diff-gaussian-rasterization2 && pip install -e .
```

* Download Grendel-GS in a separate folder and install submodules
```sh
git clone git@github.com:nyu-systems/Grendel-GS.git --recursive
cd submodules/gsplat && pip install -e .
cd submodules/simple-knn && pip install -e .
```

## Dataset
