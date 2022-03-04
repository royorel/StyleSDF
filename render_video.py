import os
import torch
import trimesh
import numpy as np
import skvideo.io
from munch import *
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from torchvision import transforms
from options import BaseOptions
from model import Generator
from utils import (
    generate_camera_params, align_volume, extract_mesh_with_marching_cubes,
    xyz2mesh, create_cameras, create_mesh_renderer, add_textures,
    )
from pytorch3d.structures import Meshes

from pdb import set_trace as st

torch.random.manual_seed(1234)


def render_video(opt, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent):
    g_ema.eval()
    surface_g_ema.eval()

    images = torch.Tensor(0, 3, opt.size, opt.size)
    num_frames = 250
    # Generate video trajectory
    trajectory = np.zeros((num_frames,3), dtype=np.float32)

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

        trajectory[:num_frames,0] = azim
        trajectory[:num_frames,1] = elev
        trajectory[:num_frames,2] = fov

    # elipsoid sweep (4 seconds)
    else:
        t = np.linspace(0, 1, num_frames)
        fov = opt.camera.fov #+ 1 * np.sin(t * 2 * np.pi)
        if opt.camera.uniform:
            elev = opt.camera.elev / 2 + opt.camera.elev / 2  * np.sin(t * 2 * np.pi)
            azim = opt.camera.azim  * np.cos(t * 2 * np.pi)
        else:
            elev = 1.5 * opt.camera.elev * np.sin(t * 2 * np.pi)
            azim = 1.5 * opt.camera.azim * np.cos(t * 2 * np.pi)

        trajectory[:num_frames,0] = azim
        trajectory[:num_frames,1] = elev
        trajectory[:num_frames,2] = fov

    trajectory = torch.from_numpy(trajectory).to(device)

    # generate input parameters for the camera trajectory
    # sample_cam_poses, sample_focals, sample_near, sample_far = \
    # generate_camera_params(trajectory, opt.renderer_output_size, device, dist_radius=opt.camera.dist_radius)
    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = \
    generate_camera_params(opt.renderer_output_size, device, locations=trajectory[:,:2],
                           fov_ang=trajectory[:,2:], dist_radius=opt.camera.dist_radius)

    # In case of noise projection, generate input parameters for the frontal position.
    # The reference mesh for the noise projection is extracted from the frontal position.
    # For more details see section C.1 in the supplementary material.
    if opt.project_noise:
        frontal_pose = torch.tensor([[0.0,0.0,opt.camera.fov]]).to(device)
        # frontal_cam_pose, frontal_focals, frontal_near, frontal_far = \
        # generate_camera_params(frontal_pose, opt.surf_extraction_output_size, device, dist_radius=opt.camera.dist_radius)
        frontal_cam_pose, frontal_focals, frontal_near, frontal_far, _ = \
        generate_camera_params(opt.surf_extraction_output_size, device, location=frontal_pose[:,:2],
                               fov_ang=frontal_pose[:,2:], dist_radius=opt.camera.dist_radius)

    # create geometry renderer (renders the depth maps)
    cameras = create_cameras(azim=np.rad2deg(trajectory[0,0].cpu().numpy()),
                             elev=np.rad2deg(trajectory[0,1].cpu().numpy()),
                             dist=1, device=device)
    renderer = create_mesh_renderer(cameras, image_size=512, specular_color=((0,0,0),),
                    ambient_color=((0.1,.1,.1),), diffuse_color=((0.75,.75,.75),),
                    device=device)

    suffix = '_azim' if opt.azim_video else '_elipsoid'

    # generate videos
    for i in range(opt.identities):
        print('Processing identity {}/{}...'.format(i+1, opt.identities))
        chunk = 1
        sample_z = torch.randn(1, opt.style_dim, device=device).repeat(chunk,1)
        video_filename = 'sample_video_{}{}.mp4'.format(i,suffix)
        writer = skvideo.io.FFmpegWriter(os.path.join(opt.results_dst_dir, video_filename),
                                         outputdict={'-pix_fmt': 'yuv420p', '-crf': '10'})
        if not opt.no_surface_renderings:
            depth_video_filename = 'sample_depth_video_{}{}.mp4'.format(i,suffix)
            depth_writer = skvideo.io.FFmpegWriter(os.path.join(opt.results_dst_dir, depth_video_filename),
                                             outputdict={'-pix_fmt': 'yuv420p', '-crf': '1'})


        ####################### Extract initial surface mesh from the frontal viewpoint #############
        # For more details see section C.1 in the supplementary material.
        if opt.project_noise:
            with torch.no_grad():
                frontal_surface_out = surface_g_ema([sample_z],
                                                    frontal_cam_pose,
                                                    frontal_focals,
                                                    frontal_near,
                                                    frontal_far,
                                                    truncation=opt.truncation_ratio,
                                                    truncation_latent=surface_mean_latent,
                                                    return_sdf=True)
                frontal_sdf = frontal_surface_out[2].cpu()

            print('Extracting Identity {} Frontal view Marching Cubes for consistent video rendering'.format(i))

            frostum_aligned_frontal_sdf = align_volume(frontal_sdf)
            del frontal_sdf

            try:
                frontal_marching_cubes_mesh = extract_mesh_with_marching_cubes(frostum_aligned_frontal_sdf)
            except ValueError:
                frontal_marching_cubes_mesh = None

            if frontal_marching_cubes_mesh != None:
                frontal_marching_cubes_mesh_filename = os.path.join(opt.results_dst_dir,'sample_{}_frontal_marching_cubes_mesh{}.obj'.format(i,suffix))
                with open(frontal_marching_cubes_mesh_filename, 'w') as f:
                    frontal_marching_cubes_mesh.export(f,file_type='obj')

            del frontal_surface_out
            torch.cuda.empty_cache()
        #############################################################################################

        for j in tqdm(range(0, num_frames, chunk)):
            with torch.no_grad():
                out = g_ema([sample_z],
                            sample_cam_extrinsics[j:j+chunk],
                            sample_focals[j:j+chunk],
                            sample_near[j:j+chunk],
                            sample_far[j:j+chunk],
                            truncation=opt.truncation_ratio,
                            truncation_latent=mean_latent,
                            randomize_noise=False,
                            project_noise=opt.project_noise,
                            mesh_path=frontal_marching_cubes_mesh_filename if opt.project_noise else None)

                rgb = out[0].cpu()

                # this is done to fit to RTX2080 RAM size (11GB)
                del out
                torch.cuda.empty_cache()

                # Convert RGB from [-1, 1] to [0,255]
                rgb = 127.5 * (rgb.clamp(-1,1).permute(0,2,3,1).cpu().numpy() + 1)

                # Add RGB, frame to video
                for k in range(chunk):
                    writer.writeFrame(rgb[k])

                ########## Extract surface ##########
                if not opt.no_surface_renderings:
                    scale = surface_g_ema.renderer.out_im_res / g_ema.renderer.out_im_res
                    surface_sample_focals = sample_focals * scale
                    surface_out = surface_g_ema([sample_z],
                                                sample_cam_extrinsics[j:j+chunk],
                                                surface_sample_focals[j:j+chunk],
                                                sample_near[j:j+chunk],
                                                sample_far[j:j+chunk],
                                                truncation=opt.truncation_ratio,
                                                truncation_latent=surface_mean_latent,
                                                return_xyz=True)
                    xyz = surface_out[2].cpu()

                    # this is done to fit to RTX2080 RAM size (11GB)
                    del surface_out
                    torch.cuda.empty_cache()

                    # Render mesh for video
                    depth_mesh = xyz2mesh(xyz)
                    mesh = Meshes(
                        verts=[torch.from_numpy(np.asarray(depth_mesh.vertices)).to(torch.float32).to(device)],
                        faces = [torch.from_numpy(np.asarray(depth_mesh.faces)).to(torch.float32).to(device)],
                        textures=None,
                        verts_normals=[torch.from_numpy(np.copy(np.asarray(depth_mesh.vertex_normals))).to(torch.float32).to(device)],
                    )
                    mesh = add_textures(mesh)
                    cameras = create_cameras(azim=np.rad2deg(trajectory[j,0].cpu().numpy()),
                                             elev=np.rad2deg(trajectory[j,1].cpu().numpy()),
                                             fov=2*trajectory[j,2].cpu().numpy(),
                                             dist=1, device=device)
                    renderer = create_mesh_renderer(cameras, image_size=512,
                                                    light_location=((0.0,1.0,5.0),), specular_color=((0.2,0.2,0.2),),
                                                    ambient_color=((0.1,0.1,0.1),), diffuse_color=((0.65,.65,.65),),
                                                    device=device)

                    mesh_image = 255 * renderer(mesh).cpu().numpy()
                    mesh_image = mesh_image[...,:3]

                    # Add depth frame to video
                    for k in range(chunk):
                        depth_writer.writeFrame(mesh_image[k])

        # Close video writers
        writer.close()
        if not opt.no_surface_renderings:
            depth_writer.close()


if __name__ == "__main__":
    device = "cuda"
    opt = BaseOptions().parse()
    opt.model.is_test = True
    opt.model.style_dim = 256
    opt.model.freeze_renderer = False
    opt.inference.size = opt.model.size
    opt.inference.camera = opt.camera
    opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.inference.style_dim = opt.model.style_dim
    opt.inference.project_noise = opt.model.project_noise
    opt.rendering.perturb = 0
    opt.rendering.force_background = True
    opt.rendering.static_viewdirs = True
    opt.rendering.return_sdf = True
    opt.rendering.N_samples = 64

    # find checkpoint directory
    # check if there's a fully trained model
    checkpoints_dir = 'full_models'
    checkpoint_path = os.path.join(checkpoints_dir, opt.experiment.expname + '.pt')
    if os.path.isfile(checkpoint_path):
        # define results directory name
        result_model_dir = 'final_model'
    else:
        checkpoints_dir = os.path.join('checkpoint', opt.experiment.expname)
        checkpoint_path = os.path.join(checkpoints_dir,
                                       'models_{}.pt'.format(opt.experiment.ckpt.zfill(7)))
        # define results directory name
        result_model_dir = 'iter_{}'.format(opt.experiment.ckpt.zfill(7))

    results_dir_basename = os.path.join(opt.inference.results_dir, opt.experiment.expname)
    opt.inference.results_dst_dir = os.path.join(results_dir_basename, result_model_dir, 'videos')
    if opt.model.project_noise:
        opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'with_noise_projection')

    os.makedirs(opt.inference.results_dst_dir, exist_ok=True)

    # load saved model
    checkpoint = torch.load(checkpoint_path)

    # load image generation model
    g_ema = Generator(opt.model, opt.rendering).to(device)

    # temp fix because of wrong noise sizes
    pretrained_weights_dict = checkpoint["g_ema"]
    model_dict = g_ema.state_dict()
    for k, v in pretrained_weights_dict.items():
        if v.size() == model_dict[k].size():
            model_dict[k] = v

    g_ema.load_state_dict(model_dict)

    # load a the volume renderee to a second that extracts surfaces at 128x128x128
    if not opt.inference.no_surface_renderings or opt.model.project_noise:
        opt['surf_extraction'] = Munch()
        opt.surf_extraction.rendering = opt.rendering
        opt.surf_extraction.model = opt.model.copy()
        opt.surf_extraction.model.renderer_spatial_output_dim = 128
        opt.surf_extraction.rendering.N_samples = opt.surf_extraction.model.renderer_spatial_output_dim
        opt.surf_extraction.rendering.return_xyz = True
        opt.surf_extraction.rendering.return_sdf = True
        opt.inference.surf_extraction_output_size = opt.surf_extraction.model.renderer_spatial_output_dim
        surface_g_ema = Generator(opt.surf_extraction.model, opt.surf_extraction.rendering, full_pipeline=False).to(device)

        # Load weights to surface extractor
        surface_extractor_dict = surface_g_ema.state_dict()
        for k, v in pretrained_weights_dict.items():
            if k in surface_extractor_dict.keys() and v.size() == surface_extractor_dict[k].size():
                surface_extractor_dict[k] = v

        surface_g_ema.load_state_dict(surface_extractor_dict)
    else:
        surface_g_ema = None

    # get the mean latent vector for g_ema
    if opt.inference.truncation_ratio < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(opt.inference.truncation_mean, device)
    else:
        mean_latent = None

    # get the mean latent vector for surface_g_ema
    if not opt.inference.no_surface_renderings or opt.model.project_noise:
        surface_mean_latent = mean_latent[0]
    else:
        surface_mean_latent = None

    render_video(opt.inference, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent)
