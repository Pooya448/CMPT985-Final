# ARCH
This is our reimplementation of ARCH. 

Our work is inspired by the ARCH paper itself, and we have also taken advantage of some of the following existing components:
1. [Stacked Hourglass Network](https://medium.com/@monadsblog/stacked-hourglass-networks-14bee8c35678)
2. [LEAP](https://github.com/neuralbodies/leap)

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


Resources:

Stacked Hourglass: https://medium.com/@monadsblog/stacked-hourglass-networks-14bee8c35678

LEAP: https://github.com/neuralbodies/leap