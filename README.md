# StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation
### [Project Page](https://stylesdf.github.io/) | [Paper](https://arxiv.org/pdf/2112.11427.pdf)

[![Explore in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/royorel/StyleSDF/blob/main/StyleSDF_demo.ipynb)<br>

[Roy Or-El](https://homes.cs.washington.edu/~royorel/)<sup>1</sup> ,
[Xuan Luo](https://roxanneluo.github.io/)<sup>1</sup>,
[Mengyi Shan](https://shanmy.github.io/)<sup>1</sup>,
[Eli Shechtman](https://research.adobe.com/person/eli-shechtman/)<sup>2</sup>,
[Jeong Joon Park](https://jjparkcv.github.io/)<sup>3</sup>,
[Ira Kemelmacher-Shlizerman](https://www.irakemelmacher.com/)<sup>1</sup><br>
<sup>1</sup>University of Washington, <sup>2</sup>Adobe Research, <sup>3</sup>Stanford University

<div align="center">
<img src=./assets/teaser.png>
</div>

## Updates
3/4/2022: Testing code and Colab demo were released. **Training files will be released soon.**

## Overview
StyleSDF is a 3D-aware GAN, aimed at solving two main challenges:
1. High-resolution, view-consistent generation of the RGB images.
2. Generating detailed 3D shapes.

StyleSDF is trained only on single-view RGB data. The 3D geometry is learned implicitly with an SDF-based volume renderer.<br>

This code is the official PyTorch implementation of the paper:
> **StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation**<br>
> Roy Or-El, Xuan Luo, Mengyi Shan, Eli Shechtman, Jeong Joon Park, Ira Kemelmacher-Shlizerman<br>
> CVPR 2022<br>
> https://arxiv.org/pdf/2112.11427.pdf

## Abstract
We introduce a high resolution, 3D-consistent image and
shape generation technique which we call StyleSDF. Our
method is trained on single-view RGB data only, and stands
on the shoulders of StyleGAN2 for image generation, while
solving two main challenges in 3D-aware GANs: 1) high-resolution,
view-consistent generation of the RGB images,
and 2) detailed 3D shape. We achieve this by merging a
SDF-based 3D representation with a style-based 2D generator.
Our 3D implicit network renders low-resolution feature
maps, from which the style-based network generates
view-consistent, 1024×1024 images. Notably, our SDFbased
3D modeling defines detailed 3D surfaces, leading
to consistent volume rendering. Our method shows higher
quality results compared to state of the art in terms of visual
and geometric quality.

## Pre-Requisits
You must have a **GPU with CUDA support** in order to run the code.

This code requires **PyTorch**, **PyTorch3D** and **torchvision** to be installed, please go to [PyTorch.org](https://pytorch.org/) and [PyTorch3d.org](https://pytorch3d.org/) for installation info.<br>
We tested our code on Python 3.8.5, PyTorch 1.9.0, PyTorch3D 0.6.1 and torchvision 0.10.0.

The following packages should also be installed:
1. lmdb
2. numpy
3. ninja
4. pillow
5. requests
6. tqdm
7. scipy
8. skimage
9. skvideo
10. trimesh[easy]
11. configargparse
12. munch
13. wandb (optional)

If any of these packages are not installed on your computer, you can install them using the supplied `requirements.txt` file:<br>
```pip install -r requirements.txt```

## Quick Demo
You can explore our method in:
1. Google Colab [![Explore in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/royorel/StyleSDF/blob/main/StyleSDF_demo.ipynb).

Alternatively, you can download the pretrained models by running:<br>
`python download_models.py`

To generate human faces from the model pre-trained on FFHQ, run:<br>
`python generate_shapes_and_images.py --expname ffhq1024x1024 --size 1024 --identities NUMBER_OF_FACES`

To generate animal faces from the model pre-trained on AFHQ, run:<br>
`python generate_shapes_and_images.py --expname afhq512x512 --size 512 --identities NUMBER_OF_FACES`

## Generating images and meshes
To generate images and meshes from a trained model, run:
`python generate_shapes_and_images.py --expname NAME_OF_TRAINED_MODEL --size MODEL_OUTPUT_SIZE --identities NUMBER_OF_FACES`

The script will generate an RGB image, a mesh generated from depth map, and the mesh extracted with Marching cubes.

### Optional flags for image and mesh generation
```
  --no_surface_renderings          When true, only RGB outputs will be generated. otherwise, both RGB and depth videos/renderings will be generated. (default: false)
  --fixed_camera_angles            When true, the generator will render indentities from a fixed set of camera angles. (default: false)
```

## Generating Videos
To generate videos from a trained model, run: <br>
`python render_video.py --expname NAME_OF_TRAINED_MODEL --size MODEL_OUTPUT_SIZE --identities NUMBER_OF_FACES`.

This script will generate RGB video as well as depth map video for each identity. The average processing time per video is ~5-10 minutes on an RTX2080 Ti GPU.

### Optional flags for video rendering
```
  --no_surface_videos          When true, only RGB video will be generated when running render_video.py. otherwise, both RGB and depth videos will be generated. this cuts the processing time per video. (default: false)
  --azim_video                 When true, the camera trajectory will travel along the azimuth direction. Otherwise, the camera will travel along an ellipsoid trajectory. (default: ellipsoid)
  --project_noise              When true, use geometry-aware noise projection to reduce flickering effects (see supplementary section C.1 in the paper). Warning: processing time significantly increases with this flag to ~20 minutes per video. (default: false)
```

## Training (training files will be released soon...)
### Preparing your Dataset
If you wish to train a model from scratch, first you need to convert your dataset to an lmdb format. Run:<br>
`python prepare_data.py --out_path OUTPUT_LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... INPUT_DATASET_PATH`

### Training the volume renderer
#### Training scripts
To train the volume renderer on FFHQ run: `bash ./scripts/train_ffhq_vol_renderer.sh`. <br>
To train the volume renderer on AFHQ run: `bash ./scripts/train_afhq_vol_renderer.sh`.

* The scripts above use distributed training. To train the models on a single GPU (not recommended) remove `-m torch.distributed.launch --nproc_per_node NUM_GPUS` from the script.

#### Training on a new dataset
To train the volume renderer on a new dataset, run:<br>
`python train_volume_renderer.py --batch BATCH_SIZE --chunk CHUNK_SIZE --expname EXPERIMENT_NAME --dataset_path DATASET_PATH`

Ideally, `CHUNK_SIZE` should be the same as `BATCH_SIZE`, but on most GPUs it will likely cause an out of memory error. In such case, reduce `CHUNK_SIZE` to perform gradient accumulation.

**Important note**: The best way to monitor the SDF convergence is to look at "Beta value" graph on wandb. Convergence is successful once beta reaches values below approx. 3*10<sup>-3</sup>. If the SDF is not converging, increase the R1 regularization weight. Another helpful option (to a lesser degree) is to decrease the weight of the minimal surface regularization.  

#### Distributed training
If you have multiple GPUs you can train your model on multiple instances by running:<br>
`python -m torch.distributed.launch --nproc_per_node NUM_GPUS train_volume_renderer.py --batch BATCH_SIZE --chunk CHUNK_SIZE --expname EXPERIMENT_NAME --dataset_path DATASET_PATH`

### Training the full_pipeline
#### Training scripts
To train the volume renderer on FFHQ run: `bash ./scripts/train_ffhq_full_pipeline_1024x1024.sh`. <br>
To train the volume renderer on AFHQ run: `bash ./scripts/train_afhq_full_pipeline_512x512.sh`.

* The scripts above assume that the volume renderer model was already trained. **Do not run them from scratch.**
* The scripts above use distributed training. To train the models on a single GPU (not recommended) remove `-m torch.distributed.launch --nproc_per_node NUM_GPUS` from the script.

#### Training on a new dataset
To train the full pipeline on a new dataset, **first train the volume renderer separately**. <br>
After the volume renderer training is finished, run:<br>
`python train_full_pipeline.py --batch BATCH_SIZE --chunk CHUNK_SIZE --expname EXPERIMENT_NAME --size OUTPUT_SIZE`

Ideally, `CHUNK_SIZE` should be the same as `BATCH_SIZE`, but on most GPUs it will likely cause an out of memory error. In such case, reduce `CHUNK_SIZE` to perform gradient accumulation.

#### Distributed training
If you have multiple GPUs you can train your model on multiple instances by running:<br>
`python -m torch.distributed.launch --nproc_per_node NUM_GPUS train_full_pipeline.py --batch BATCH_SIZE --chunk CHUNK_SIZE --expname EXPERIMENT_NAME --size OUTPUT_SIZE`

Here, **BATCH_SIZE represents the batch per GPU**, not the overall batch size.

### Optional training flags
```
Training regime options:
  --iter                 Total number of training iterations. (default: 300,000)
  --wandb                Use use weights and biases logging. (default: False)
  --r1                   Weight of the r1 regularization. (default: 10.0)
  --view_lambda          Weight of the viewpoint regularization. (Equation 6, default: 15)
  --eikonal_lambda       Weight of the eikonal regularization. (Equation 7, default: 0.1)
  --min_surf_lambda      Weight of the minimal surface regularization. (Equation 8, default: 0.05)

Camera options:
  --uniform              When true, the camera position is sampled from uniform distribution. (default: gaussian)
  --azim                 Camera azimuth angle std (guassian)/range (uniform) in Radians. (default: 0.3 Rad.)
  --elev                 Camera elevation angle std (guassian)/range (uniform) in Radians. (default: 0.15 Rad.)
  --fov                  Camera field of view half angle in **Degrees**. (default: 6 Deg.)
  --dist_radius          Radius of points sampling distance from the origin. Determines the near and far fields. (default: 0.12)
```

## Citation
If you use this code for your research, please cite our paper.
```
@article{orel2021stylesdf,
  title={Style{SDF}: {H}igh-{R}esolution {3D}-{C}onsistent {I}mage and {G}eometry {G}eneration},
  author={Or-El, Roy and
          Luo, Xuan and
          Shan, Mengyi and
          Shechtman, Eli and
          Park, Jeong Joon and
          Kemelmacher-Shlizerman, Ira},
  journal={arXiv preprint arXiv:2112.11427},
  year={2021}
}
```

## Acknowledgments
This code is inspired by rosinality's [StyleGAN2-PyTorch](https://github.com/rosinality/stylegan2-pytorch) and Yen-Chen Lin's [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch).
