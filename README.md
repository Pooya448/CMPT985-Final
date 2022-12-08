# ARCH
This is our reimplementation of ARCH. 

Our work is inspired by the ARCH paper itself, and we have also taken advantage of some of the following existing components:
1. [Stacked Hourglass Network](https://medium.com/@monadsblog/stacked-hourglass-networks-14bee8c35678)
2. [LEAP](https://github.com/neuralbodies/leap)
3. [SDF](https://github.com/sxyu/sdf)

## Models

All the models have been implemented in the `models` folder in the root directory. This includes:

- Stacked Hourglass Network (`/models/SHGNet.py`)
- 3 Subnetworks
- -  Color Network `models/ColorNet.py`
- -  Occupany Network  `models/OccNet.py`
- -  Normal Network `models/NormNet.py`
- UNet (`models/UNet.py`)
- Differentiable Renderer (`/models/DR.py`)

## Train

To run the training job, run the following:

```
python trainer_arch.py -c configs/arch.yaml
```

## OpenPose

In order to obtain the OpenPose data, first follow [this guide](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#windows-portable-demo) to set it up with Windows. (or can alternatively build from source)

After this is setup, run the following command to get both the video and json output on the image directory you want:


```
./build/examples/openpose/openpose.bin --image_dir <IMAGE_DIR> --write_json {OUTPUT_JSON_PATH}
```

## LEAP

In order to install LEAP, simply follow the instructions in their [README](https://github.com/neuralbodies/leap#2-installation)

For easier access, we have provided this below (Run this in root of this directory):

```
git clone https://github.com/neuralbodies/leap.git
cd ./leap
python setup.py build_ext --inplace
pip install -e .

```

## Resources:

Stacked Hourglass: https://medium.com/@monadsblog/stacked-hourglass-networks-14bee8c35678

LEAP: https://github.com/neuralbodies/leap

SDF Library: https://github.com/sxyu/sdf