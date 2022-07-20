""" Program to run 3D-aware GAN through Cog/Replicate"""
import os
import subprocess
import tempfile
import zipfile
from pdb import set_trace as st

import numpy as np
import skvideo.io
import torch
import torchvision
import trimesh
from cog import BaseModel, BasePredictor, File, Input, Path
from munch import *
from PIL import Image
from pytorch3d.structures import Meshes
from scipy.spatial import Delaunay
from skimage.measure import marching_cubes
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

from model import Generator
from options import BaseOptions
from utils import (
    add_textures,
    align_volume,
    create_cameras,
    create_mesh_renderer,
    extract_mesh_with_marching_cubes,
    generate_camera_params,
    xyz2mesh,
)

""" Multi-output support for Cog"""


class Output(BaseModel):
    output1: Path
    output2: Path


class OptGroup:
    def __init__(self):
        pass


class Opt:
    def __init__(self):
        self.dataset = OptGroup()
        self.experiment = OptGroup()
        self.training = OptGroup()
        self.inference = OptGroup()
        self.camera = OptGroup()
        self.model = OptGroup()
        self.rendering = OptGroup()


def set_default_options():
    """Sets default option settings"""
    opt = Opt()
    # dataset
    opt.dataset.dataset_path = "./datasets/FFHQ"

    # experiment
    opt.experiment.config = None
    opt.experiment.expname = "debug"
    opt.experiment.ckpt = 300000
    opt.experiment.continue_training = False

    # training
    opt.training.checkpoints_dir = "./checkpoint"
    opt.training.iter = 300000
    opt.training.batch = 4
    opt.training.chunk = 4
    opt.training.val_n_sample = 8
    opt.training.d_reg_every = 16
    opt.training.g_reg_every = 4
    opt.training.local_rank = 0
    opt.training.mixing = 0.9
    opt.training.lr = 0.002
    opt.training.r1 = 10
    opt.training.view_lambda = 15
    opt.training.eikonal_lambda = 0.1
    opt.training.min_surf_lambda = 0.05
    opt.training.min_surf_beta = 100.0
    opt.training.path_regularize = 2
    opt.training.path_batch_shrink = 2
    opt.training.wandb = False
    opt.training.no_sphere_init = False

    # inference
    opt.inference.results_dir = "./evaluations"
    opt.inference.truncation_ratio = 0.5
    opt.inference.truncation_mean = 10000
    opt.inference.identities = 16
    opt.inference.num_views_per_id = 3
    opt.inference.no_surface_renderings = False
    opt.inference.fixed_camera_angles = False
    opt.inference.azim_video = False

    # model
    opt.model.size = 256
    opt.model.style_dim = 256
    opt.model.channel_multiplier = 2
    opt.model.n_mlp = 8
    opt.model.lr_mapping = 0.01
    opt.model.renderer_spatial_output_dim = 64
    opt.model.project_noise = False

    # camera
    opt.camera.uniform = False
    opt.camera.azim = 0.3
    opt.camera.elev = 0.15
    opt.camera.fov = 6
    opt.camera.dist_radius = 0.12

    # rendering
    opt.rendering.depth = 8
    opt.rendering.width = 256
    opt.rendering.no_sdf = False
    opt.rendering.no_z_normalize = False
    opt.rendering.static_viewdirs = False
    opt.rendering.N_samples = 24
    opt.rendering.no_offset_sampling = False
    opt.rendering.perturb = 1.0
    opt.rendering.raw_noise_std = 0.0
    opt.rendering.force_background = False
    opt.rendering.return_xyz = False
    opt.rendering.return_sdf = False

    return opt


def parse_options(opt):
    """Manually parse options"""
    opt = Munch()
    args = set_default_options()
    for group in [
        "dataset",
        "experiment",
        "training",
        "inference",
        "model",
        "camera",
        "rendering",
    ]:
        opt[group] = Munch()
        for key in getattr(args, group).__dict__.keys():
            opt[group][key] = getattr(args, group).__dict__[key]
    return opt


""" Main Predictor Class for Cog"""


class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(
        self,
        image_type: str = Input(
            description="Choose between animal or human face to generate",
            choices=["animal", "human"],
        ),
        identities: int = Input(
            description="Number of different faces to generate", ge=1, le=3, default=1
        ),
        num_views_per_id: int = Input(
            description="Number of output views per identity", ge=1, le=3, default=1
        ),
        generate_video: bool = Input(
            description="Whether to generate a live depth video. If True, please select identities=1.",
            default=False,
        ),
    ) -> Output:
        # modified: manually set options
        args = set_default_options()
        opt = parse_options(args)

        # human: trained on ffhq, image size:1024
        if image_type == "human":
            opt.experiment.expname = "ffhq1024x1024"
            opt.model.size = 1024
        else:  # animal: trained on afhq, image size: 512
            opt.experiment.expname = "afhq512x512"
            opt.model.size = 512

        opt.inference.identities = 1
        opt.model.is_test = True
        opt.model.freeze_renderer = False
        opt.rendering.offset_sampling = True
        opt.rendering.static_viewdirs = True
        opt.rendering.force_background = True
        opt.rendering.perturb = 0
        opt.inference.size = opt.model.size
        opt.inference.camera = opt.camera
        opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
        opt.inference.style_dim = opt.model.style_dim
        opt.inference.project_noise = opt.model.project_noise
        opt.inference.return_xyz = opt.rendering.return_xyz
        opt.inference.num_views_per_id = num_views_per_id
        opt.inference.identities = identities

        if generate_video:
            opt.rendering.perturb = 0
            opt.rendering.force_background = True
            opt.rendering.static_viewdirs = True
            opt.rendering.return_sdf = True
            opt.rendering.N_samples = 64

        #############   DEFINE MODEL/CHECKPOINT DIR #############
        # find checkpoint directory
        # check if there's a fully trained model
        print("Creating model and checkpoint directories......")
        checkpoints_dir = "full_models"
        checkpoint_path = os.path.join(checkpoints_dir, opt.experiment.expname + ".pt")
        if os.path.isfile(checkpoint_path):
            # define results directory name
            result_model_dir = "final_model"
        else:
            checkpoints_dir = os.path.join("checkpoint", opt.experiment.expname)
            checkpoint_path = os.path.join(
                checkpoints_dir, "models_{}.pt".format(opt.experiment.ckpt.zfill(7))
            )
            # define results directory name
            result_model_dir = "iter_{}".format(opt.experiment.ckpt.zfill(7))

        ################# DEFINE RESULTS DIRECTORIES ############
        print("Creating results directories......")

        if generate_video:
            results_dir_basename = os.path.join(
                opt.inference.results_dir, opt.experiment.expname
            )
            opt.inference.results_dst_dir = os.path.join(
                results_dir_basename, result_model_dir, "videos"
            )
            if opt.model.project_noise:
                opt.inference.results_dst_dir = os.path.join(
                    opt.inference.results_dst_dir, "with_noise_projection"
                )

            os.makedirs(opt.inference.results_dst_dir, exist_ok=True)
        else:
            results_dir_basename = os.path.join(
                opt.inference.results_dir, opt.experiment.expname
            )
            opt.inference.results_dst_dir = os.path.join(
                results_dir_basename, result_model_dir
            )
            if opt.inference.fixed_camera_angles:
                opt.inference.results_dst_dir = os.path.join(
                    opt.inference.results_dst_dir, "fixed_angles"
                )
            else:
                opt.inference.results_dst_dir = os.path.join(
                    opt.inference.results_dst_dir, "random_angles"
                )
            os.makedirs(opt.inference.results_dst_dir, exist_ok=True)
            os.makedirs(
                os.path.join(opt.inference.results_dst_dir, "images"), exist_ok=True
            )

            if not opt.inference.no_surface_renderings:
                os.makedirs(
                    os.path.join(opt.inference.results_dst_dir, "depth_map_meshes"),
                    exist_ok=True,
                )
                os.makedirs(
                    os.path.join(
                        opt.inference.results_dst_dir, "marching_cubes_meshes"
                    ),
                    exist_ok=True,
                )

        print("Loading model checkpoint......")

        # load saved model
        checkpoint = torch.load(checkpoint_path)

        print("Loading Generator......")
        # load image generation model
        g_ema = Generator(opt.model, opt.rendering).to(self.device)
        pretrained_weights_dict = checkpoint["g_ema"]
        model_dict = g_ema.state_dict()
        for k, v in pretrained_weights_dict.items():
            if v.size() == model_dict[k].size():
                model_dict[k] = v

        g_ema.load_state_dict(model_dict)

        # load a second volume renderer that extracts surfaces at 128x128x128 (or higher) for better surface resolution
        if not opt.inference.no_surface_renderings:
            opt["surf_extraction"] = Munch()
            opt.surf_extraction.rendering = opt.rendering
            opt.surf_extraction.model = opt.model.copy()
            opt.surf_extraction.model.renderer_spatial_output_dim = 128
            opt.surf_extraction.rendering.N_samples = (
                opt.surf_extraction.model.renderer_spatial_output_dim
            )
            opt.surf_extraction.rendering.return_xyz = True
            opt.surf_extraction.rendering.return_sdf = True
            surface_g_ema = Generator(
                opt.surf_extraction.model,
                opt.surf_extraction.rendering,
                full_pipeline=False,
            ).to(self.device)

            # Load weights to surface extractor
            surface_extractor_dict = surface_g_ema.state_dict()
            for k, v in pretrained_weights_dict.items():
                if (
                    k in surface_extractor_dict.keys()
                    and v.size() == surface_extractor_dict[k].size()
                ):
                    surface_extractor_dict[k] = v

            surface_g_ema.load_state_dict(surface_extractor_dict)
        else:
            surface_g_ema = None

        # get the mean latent vector for g_ema
        if opt.inference.truncation_ratio < 1:
            with torch.no_grad():
                mean_latent = g_ema.mean_latent(
                    opt.inference.truncation_mean, self.device
                )
        else:
            surface_mean_latent = None

        # get the mean latent vector for surface_g_ema
        if not opt.inference.no_surface_renderings:
            surface_mean_latent = mean_latent[0]
        else:
            surface_mean_latent = None

        if generate_video:
            print("Rendering video..... this may take a few minutes.")

            video_filenames, depth_video_filenames, mesh_filenames = render_video(
                opt.inference,
                g_ema,
                surface_g_ema,
                self.device,
                mean_latent,
                surface_mean_latent,
            )
            print("Saved videos ", video_filenames[0], depth_video_filenames[0])
            return Output(output1=Path(video_filenames[0]), output2=Path(depth_video_filenames[0]))
        else:
            print("Creating images and meshes...")

            rgb_path, depth_mesh_list, marching_cubes_mesh_list = generate(
                opt.inference,
                g_ema,
                surface_g_ema,
                self.device,
                mean_latent,
                surface_mean_latent,
            )

            # zip mesh file
            mesh_zip_path = Path(tempfile.mkdtemp()) / "out.zip"

            print("Zipping depth mesh and marching cubes mesh files...")

            with zipfile.ZipFile(mesh_zip_path, "w") as zip_obj:
                # Add multiple files to the zip
                for f in depth_mesh_list + marching_cubes_mesh_list:
                    zip_obj.write(f)

            return Output(output1=rgb_path, output2=mesh_zip_path)


def generate(opt, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent):
    g_ema.eval()

    rgb_images_list = []
    depth_mesh_list = []
    marching_cubes_mesh_list = []

    if not opt.no_surface_renderings:
        surface_g_ema.eval()

    # set camera angles
    if opt.fixed_camera_angles:
        # These can be changed to any other specific viewpoints.
        # You can add or remove viewpoints as you wish
        locations = torch.tensor(
            [
                [0, 0],
                [-1.5 * opt.camera.azim, 0],
                [-1 * opt.camera.azim, 0],
                [-0.5 * opt.camera.azim, 0],
                [0.5 * opt.camera.azim, 0],
                [1 * opt.camera.azim, 0],
                [1.5 * opt.camera.azim, 0],
                [0, -1.5 * opt.camera.elev],
                [0, -1 * opt.camera.elev],
                [0, -0.5 * opt.camera.elev],
                [0, 0.5 * opt.camera.elev],
                [0, 1 * opt.camera.elev],
                [0, 1.5 * opt.camera.elev],
            ],
            device=self.device,
        )
        # For zooming in/out change the values of fov
        # (This can be defined for each view separately via a custom tensor
        # like the locations tensor above. Tensor shape should be [locations.shape[0],1])
        # reasonable values are [0.75 * opt.camera.fov, 1.25 * opt.camera.fov]
        fov = opt.camera.fov * torch.ones((locations.shape[0], 1), device=self.device)
        num_viewdirs = locations.shape[0]
    else:  # draw random camera angles
        locations = None
        # fov = None
        fov = opt.camera.fov
        num_viewdirs = opt.num_views_per_id

    # generate images
    for i in tqdm(range(opt.identities)):
        with torch.no_grad():
            chunk = 8
            sample_z = torch.randn(1, opt.style_dim, device=device).repeat(
                num_viewdirs, 1
            )
            (
                sample_cam_extrinsics,
                sample_focals,
                sample_near,
                sample_far,
                sample_locations,
            ) = generate_camera_params(
                opt.renderer_output_size,
                device,
                batch=num_viewdirs,
                locations=locations,  # input_fov=fov,
                uniform=opt.camera.uniform,
                azim_range=opt.camera.azim,
                elev_range=opt.camera.elev,
                fov_ang=fov,
                dist_radius=opt.camera.dist_radius,
            )
            rgb_images = torch.Tensor(0, 3, opt.size, opt.size)
            rgb_images_thumbs = torch.Tensor(
                0, 3, opt.renderer_output_size, opt.renderer_output_size
            )
            for j in range(0, num_viewdirs, chunk):
                out = g_ema(
                    [sample_z[j : j + chunk]],
                    sample_cam_extrinsics[j : j + chunk],
                    sample_focals[j : j + chunk],
                    sample_near[j : j + chunk],
                    sample_far[j : j + chunk],
                    truncation=opt.truncation_ratio,
                    truncation_latent=mean_latent,
                )

                rgb_images = torch.cat([rgb_images, out[0].cpu()], 0)
                rgb_images_thumbs = torch.cat([rgb_images_thumbs, out[1].cpu()], 0)

            rgb_images_filename = os.path.join(
                opt.results_dst_dir, "images", "{}.png".format(str(i).zfill(7))
            )
            utils.save_image(
                rgb_images,
                rgb_images_filename,
                nrow=num_viewdirs,
                normalize=True,
                padding=0,
                range=(-1, 1),
            )

            rgb_images_list.append(rgb_images.cpu())

            # this is done to fit to RTX2080 RAM size (11GB)
            del out
            torch.cuda.empty_cache()

            if not opt.no_surface_renderings:
                surface_chunk = 1
                scale = surface_g_ema.renderer.out_im_res / g_ema.renderer.out_im_res
                surface_sample_focals = sample_focals * scale
                for j in range(0, num_viewdirs, surface_chunk):
                    surface_out = surface_g_ema(
                        [sample_z[j : j + surface_chunk]],
                        sample_cam_extrinsics[j : j + surface_chunk],
                        surface_sample_focals[j : j + surface_chunk],
                        sample_near[j : j + surface_chunk],
                        sample_far[j : j + surface_chunk],
                        truncation=opt.truncation_ratio,
                        truncation_latent=surface_mean_latent,
                        return_sdf=True,
                        return_xyz=True,
                    )

                    xyz = surface_out[2].cpu()
                    sdf = surface_out[3].cpu()

                    # this is done to fit to RTX2080 RAM size (11GB)
                    del surface_out
                    torch.cuda.empty_cache()

                    # mesh extractions are done one at a time
                    for k in range(surface_chunk):
                        curr_locations = sample_locations[j : j + surface_chunk]
                        loc_str = "_azim{}_elev{}".format(
                            int(curr_locations[k, 0] * 180 / np.pi),
                            int(curr_locations[k, 1] * 180 / np.pi),
                        )

                        # Save depth outputs as meshes
                        depth_mesh_filename = os.path.join(
                            opt.results_dst_dir,
                            "depth_map_meshes",
                            "sample_{}_depth_mesh{}.obj".format(i, loc_str),
                        )
                        depth_mesh = xyz2mesh(xyz[k : k + surface_chunk])
                        if depth_mesh != None:
                            with open(depth_mesh_filename, "w") as f:
                                depth_mesh.export(f, file_type="obj")

                                depth_mesh_list.append(depth_mesh_filename)

                        # extract full geometry with marching cubes
                        if j == 0:
                            try:
                                frostum_aligned_sdf = align_volume(sdf)
                                marching_cubes_mesh = extract_mesh_with_marching_cubes(
                                    frostum_aligned_sdf[k : k + surface_chunk]
                                )
                            except ValueError:
                                marching_cubes_mesh = None
                                print("Marching cubes extraction failed.")
                                print(
                                    "Please check whether the SDF values are all larger (or all smaller) than 0."
                                )

                            if marching_cubes_mesh != None:
                                marching_cubes_mesh_filename = os.path.join(
                                    opt.results_dst_dir,
                                    "marching_cubes_meshes",
                                    "sample_{}_marching_cubes_mesh{}.obj".format(
                                        i, loc_str
                                    ),
                                )
                                with open(marching_cubes_mesh_filename, "w") as f:
                                    marching_cubes_mesh.export(f, file_type="obj")

                                    marching_cubes_mesh_list.append(
                                        marching_cubes_mesh_filename
                                    )

    # Create a grid where each row represents one identity
    all_rgb_images = None
    for im in rgb_images_list:
        n, c, w, h = im.shape
        all_rgb_images = (
            torch.vstack((all_rgb_images, im)) if all_rgb_images is not None else im
        )
    grid = torchvision.utils.make_grid(
        all_rgb_images,
        range=(-1, 1),
        normalize=True,
        padding=0,
        nrow=opt.num_views_per_id,
    )

    rgb_path = Path(tempfile.mkdtemp()) / "out.jpg"
    torchvision.utils.save_image(grid, rgb_path)
    return rgb_path, depth_mesh_list, marching_cubes_mesh_list


def render_video(opt, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent):

    video_filenames = []
    depth_video_filenames = []
    mesh_filenames = []

    g_ema.eval()
    if not opt.no_surface_renderings or opt.project_noise:
        surface_g_ema.eval()

    images = torch.Tensor(0, 3, opt.size, opt.size)
    num_frames = 250
    # Generate video trajectory
    trajectory = np.zeros((num_frames, 3), dtype=np.float32)

    # set camera trajectory
    # sweep azimuth angles (4 seconds)
    if opt.azim_video:
        t = np.linspace(0, 1, num_frames)
        elev = 0
        fov = opt.camera.fov
        if opt.camera.uniform:
            azim = opt.camera.azim * np.cos(t * 2 * np.pi)
        else:
            azim = 1.5 * opt.camera.azim * np.cos(t * 2 * np.pi)

        trajectory[:num_frames, 0] = azim
        trajectory[:num_frames, 1] = elev
        trajectory[:num_frames, 2] = fov

    # elipsoid sweep (4 seconds)
    else:
        t = np.linspace(0, 1, num_frames)
        fov = opt.camera.fov  # + 1 * np.sin(t * 2 * np.pi)
        if opt.camera.uniform:
            elev = opt.camera.elev / 2 + opt.camera.elev / 2 * np.sin(t * 2 * np.pi)
            azim = opt.camera.azim * np.cos(t * 2 * np.pi)
        else:
            elev = 1.5 * opt.camera.elev * np.sin(t * 2 * np.pi)
            azim = 1.5 * opt.camera.azim * np.cos(t * 2 * np.pi)

        trajectory[:num_frames, 0] = azim
        trajectory[:num_frames, 1] = elev
        trajectory[:num_frames, 2] = fov

    trajectory = torch.from_numpy(trajectory).to(device)

    # generate input parameters for the camera trajectory
    # sample_cam_poses, sample_focals, sample_near, sample_far = \
    # generate_camera_params(trajectory, opt.renderer_output_size, device, dist_radius=opt.camera.dist_radius)
    (
        sample_cam_extrinsics,
        sample_focals,
        sample_near,
        sample_far,
        _,
    ) = generate_camera_params(
        opt.renderer_output_size,
        device,
        locations=trajectory[:, :2],
        fov_ang=trajectory[:, 2:],
        dist_radius=opt.camera.dist_radius,
    )

    # In case of noise projection, generate input parameters for the frontal position.
    # The reference mesh for the noise projection is extracted from the frontal position.
    # For more details see section C.1 in the supplementary material.
    if opt.project_noise:
        frontal_pose = torch.tensor([[0.0, 0.0, opt.camera.fov]]).to(device)
        # frontal_cam_pose, frontal_focals, frontal_near, frontal_far = \
        # generate_camera_params(frontal_pose, opt.surf_extraction_output_size, device, dist_radius=opt.camera.dist_radius)
        (
            frontal_cam_pose,
            frontal_focals,
            frontal_near,
            frontal_far,
            _,
        ) = generate_camera_params(
            opt.surf_extraction_output_size,
            device,
            location=frontal_pose[:, :2],
            fov_ang=frontal_pose[:, 2:],
            dist_radius=opt.camera.dist_radius,
        )

    # create geometry renderer (renders the depth maps)
    cameras = create_cameras(
        azim=np.rad2deg(trajectory[0, 0].cpu().numpy()),
        elev=np.rad2deg(trajectory[0, 1].cpu().numpy()),
        dist=1,
        device=device,
    )
    renderer = create_mesh_renderer(
        cameras,
        image_size=512,
        specular_color=((0, 0, 0),),
        ambient_color=((0.1, 0.1, 0.1),),
        diffuse_color=((0.75, 0.75, 0.75),),
        device=device,
    )

    suffix = "_azim" if opt.azim_video else "_elipsoid"

    # generate videos
    for i in range(opt.identities):
        print("Processing identity {}/{}...".format(i + 1, opt.identities))
        chunk = 1
        sample_z = torch.randn(1, opt.style_dim, device=device).repeat(chunk, 1)
        video_filename = "sample_video_{}{}.mp4".format(i, suffix)

        print(f"Writing to {os.path.join(opt.results_dst_dir, video_filename),}")

        video_filenames.append(
            os.path.join(opt.results_dst_dir, video_filename),
        )

        writer = skvideo.io.FFmpegWriter(
            os.path.join(opt.results_dst_dir, video_filename),
            outputdict={"-pix_fmt": "yuv420p", "-crf": "10"},
        )
        if not opt.no_surface_renderings:
            depth_video_filename = "sample_depth_video_{}{}.mp4".format(i, suffix)

            print(
                f"Writing to {os.path.join(opt.results_dst_dir, depth_video_filename),}"
            )

            depth_video_filenames.append(
                os.path.join(opt.results_dst_dir, depth_video_filename)
            )
            depth_writer = skvideo.io.FFmpegWriter(
                os.path.join(opt.results_dst_dir, depth_video_filename),
                outputdict={"-pix_fmt": "yuv420p", "-crf": "1"},
            )

        ###################### Extract initial surface mesh from the frontal viewpoint #############
        # For more details see section C.1 in the supplementary material.
        if opt.project_noise:
            with torch.no_grad():
                frontal_surface_out = surface_g_ema(
                    [sample_z],
                    frontal_cam_pose,
                    frontal_focals,
                    frontal_near,
                    frontal_far,
                    truncation=opt.truncation_ratio,
                    truncation_latent=surface_mean_latent,
                    return_sdf=True,
                )
                frontal_sdf = frontal_surface_out[2].cpu()

            print(
                "Extracting Identity {} Frontal view Marching Cubes for consistent video rendering".format(
                    i
                )
            )

            frostum_aligned_frontal_sdf = align_volume(frontal_sdf)
            del frontal_sdf

            try:
                frontal_marching_cubes_mesh = extract_mesh_with_marching_cubes(
                    frostum_aligned_frontal_sdf
                )
            except ValueError:
                frontal_marching_cubes_mesh = None

            if frontal_marching_cubes_mesh != None:
                print('Marching cubes mesh')
                frontal_marching_cubes_mesh_filename = os.path.join(
                    opt.results_dst_dir,
                    "sample_{}_frontal_marching_cubes_mesh{}.obj".format(i, suffix),
                )
                with open(frontal_marching_cubes_mesh_filename, "w") as f:
                    frontal_marching_cubes_mesh.export(f, file_type="obj")

                mesh_filenames.append(frontal_marching_cubes_mesh_filename)

            del frontal_surface_out
            torch.cuda.empty_cache()
        #############################################################################################

        for j in tqdm(range(0, num_frames, chunk)):
            with torch.no_grad():
                out = g_ema(
                    [sample_z],
                    sample_cam_extrinsics[j : j + chunk],
                    sample_focals[j : j + chunk],
                    sample_near[j : j + chunk],
                    sample_far[j : j + chunk],
                    truncation=opt.truncation_ratio,
                    truncation_latent=mean_latent,
                    randomize_noise=False,
                    project_noise=opt.project_noise,
                    mesh_path=frontal_marching_cubes_mesh_filename
                    if opt.project_noise
                    else None,
                )

                rgb = out[0].cpu()

                # this is done to fit to RTX2080 RAM size (11GB)
                del out
                torch.cuda.empty_cache()

                # Convert RGB from [-1, 1] to [0,255]
                rgb = 127.5 * (rgb.clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy() + 1)

                # Add RGB, frame to video
                for k in range(chunk):
                    writer.writeFrame(rgb[k])

                ########## Extract surface ##########
                if not opt.no_surface_renderings:
                    scale = (
                        surface_g_ema.renderer.out_im_res / g_ema.renderer.out_im_res
                    )
                    surface_sample_focals = sample_focals * scale
                    surface_out = surface_g_ema(
                        [sample_z],
                        sample_cam_extrinsics[j : j + chunk],
                        surface_sample_focals[j : j + chunk],
                        sample_near[j : j + chunk],
                        sample_far[j : j + chunk],
                        truncation=opt.truncation_ratio,
                        truncation_latent=surface_mean_latent,
                        return_xyz=True,
                    )
                    xyz = surface_out[2].cpu()

                    # this is done to fit to RTX2080 RAM size (11GB)
                    del surface_out
                    torch.cuda.empty_cache()

                    # Render mesh for video
                    depth_mesh = xyz2mesh(xyz)
                    mesh = Meshes(
                        verts=[
                            torch.from_numpy(np.asarray(depth_mesh.vertices))
                            .to(torch.float32)
                            .to(device)
                        ],
                        faces=[
                            torch.from_numpy(np.asarray(depth_mesh.faces))
                            .to(torch.float32)
                            .to(device)
                        ],
                        textures=None,
                        verts_normals=[
                            torch.from_numpy(
                                np.copy(np.asarray(depth_mesh.vertex_normals))
                            )
                            .to(torch.float32)
                            .to(device)
                        ],
                    )
                    mesh = add_textures(mesh)
                    cameras = create_cameras(
                        azim=np.rad2deg(trajectory[j, 0].cpu().numpy()),
                        elev=np.rad2deg(trajectory[j, 1].cpu().numpy()),
                        fov=2 * trajectory[j, 2].cpu().numpy(),
                        dist=1,
                        device=device,
                    )
                    renderer = create_mesh_renderer(
                        cameras,
                        image_size=512,
                        light_location=((0.0, 1.0, 5.0),),
                        specular_color=((0.2, 0.2, 0.2),),
                        ambient_color=((0.1, 0.1, 0.1),),
                        diffuse_color=((0.65, 0.65, 0.65),),
                        device=device,
                    )

                    mesh_image = 255 * renderer(mesh).cpu().numpy()
                    mesh_image = mesh_image[..., :3]

                    # Add depth frame to video
                    for k in range(chunk):
                        depth_writer.writeFrame(mesh_image[k])

        # Close video writers
        writer.close()
        if not opt.no_surface_renderings:
            depth_writer.close()

        return video_filenames, depth_video_filenames, mesh_filenames
