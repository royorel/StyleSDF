import argparse
import math
import random
import os
import yaml
import numpy as np
import torch
import torch.distributed as dist
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
from PIL import Image
from losses import *
from options import BaseOptions
from model import Generator, VolumeRenderDiscriminator
from dataset import MultiResolutionDataset
from utils import data_sampler, requires_grad, accumulate, sample_data, make_noise, mixing_noise, generate_camera_params
from distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size

try:
    import wandb
except ImportError:
    wandb = None


def train(opt, experiment_opt, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    d_view_loss = torch.tensor(0.0, device=device)
    g_view_loss = torch.tensor(0.0, device=device)
    g_eikonal = torch.tensor(0.0, device=device)
    g_minimal_surface = torch.tensor(0.0, device=device)

    g_loss_val = 0
    loss_dict = {}

    viewpoint_condition = opt.view_lambda > 0

    if opt.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = [torch.randn(opt.val_n_sample, opt.style_dim, device=device).repeat_interleave(8,dim=0)]
    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = generate_camera_params(opt.renderer_output_size, device, batch=opt.val_n_sample, sweep=True,
                                                                                         uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                                                                         elev_range=opt.camera.elev, fov_ang=opt.camera.fov,
                                                                                         dist_radius=opt.camera.dist_radius)

    if opt.with_sdf and opt.sphere_init and opt.start_iter == 0:
        init_pbar = range(10000)
        if get_rank() == 0:
            init_pbar = tqdm(init_pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

        generator.zero_grad()
        for idx in init_pbar:
            noise = mixing_noise(3, opt.style_dim, opt.mixing, device)
            cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(opt.renderer_output_size, device, batch=3,
                                                                                uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                                                                elev_range=opt.camera.elev, fov_ang=opt.camera.fov,
                                                                                dist_radius=opt.camera.dist_radius)
            sdf, target_values = g_module.init_forward(noise, cam_extrinsics, focal, near, far)
            loss = F.l1_loss(sdf, target_values)
            loss.backward()
            g_optim.step()
            generator.zero_grad()
            if get_rank() == 0:
                init_pbar.set_description((f"MLP init to sphere procedure - Loss: {loss.item():.4f}"))

        accumulate(g_ema, g_module, 0)
        torch.save(
            {
                "g": g_module.state_dict(),
                "d": d_module.state_dict(),
                "g_ema": g_ema.state_dict(),
            },
            os.path.join(opt.checkpoints_dir, experiment_opt.expname, f"sdf_init_models_{str(0).zfill(7)}.pt")
        )
        print('Successfully saved checkpoint for SDF initialized MLP.')

    pbar = range(opt.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=opt.start_iter, dynamic_ncols=True, smoothing=0.01)

    for idx in pbar:
        i = idx + opt.start_iter

        if i > opt.iter:
            print("Done!")

            break

        requires_grad(generator, False)
        requires_grad(discriminator, True)
        discriminator.zero_grad()
        _, real_imgs = next(loader)
        real_imgs = real_imgs.to(device)
        noise = mixing_noise(opt.batch, opt.style_dim, opt.mixing, device)
        cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(opt.renderer_output_size, device, batch=opt.batch,
                                                                            uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                                                            elev_range=opt.camera.elev, fov_ang=opt.camera.fov,
                                                                            dist_radius=opt.camera.dist_radius)
        gen_imgs = []
        for j in range(0, opt.batch, opt.chunk):
            curr_noise = [n[j:j+opt.chunk] for n in noise]
            _, fake_img = generator(curr_noise,
                                    cam_extrinsics[j:j+opt.chunk],
                                    focal[j:j+opt.chunk],
                                    near[j:j+opt.chunk],
                                    far[j:j+opt.chunk])

            gen_imgs += [fake_img]

        gen_imgs = torch.cat(gen_imgs, 0)
        fake_pred, fake_viewpoint_pred = discriminator(gen_imgs.detach())
        if viewpoint_condition:
            d_view_loss = opt.view_lambda * viewpoints_loss(fake_viewpoint_pred, gt_viewpoints)

        real_imgs.requires_grad = True
        real_pred, _ = discriminator(real_imgs)
        d_gan_loss = d_logistic_loss(real_pred, fake_pred)
        grad_penalty = d_r1_loss(real_pred, real_imgs)
        r1_loss = opt.r1 * 0.5 * grad_penalty
        d_loss = d_gan_loss + r1_loss + d_view_loss
        d_loss.backward()
        d_optim.step()

        loss_dict["d"] = d_gan_loss
        loss_dict["r1"] = r1_loss
        loss_dict["d_view"] = d_view_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        for j in range(0, opt.batch, opt.chunk):
            noise = mixing_noise(opt.chunk, opt.style_dim, opt.mixing, device)
            cam_extrinsics, focal, near, far, curr_gt_viewpoints = generate_camera_params(opt.renderer_output_size, device, batch=opt.chunk,
                                                                                     uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                                                                     elev_range=opt.camera.elev, fov_ang=opt.camera.fov,
                                                                                     dist_radius=opt.camera.dist_radius)

            out = generator(noise, cam_extrinsics, focal, near, far,
                            return_sdf=opt.min_surf_lambda > 0,
                            return_eikonal=opt.eikonal_lambda > 0)
            fake_img  = out[1]
            if opt.min_surf_lambda > 0:
                sdf = out[2]
            if opt.eikonal_lambda > 0:
                eikonal_term = out[3]

            fake_pred, fake_viewpoint_pred = discriminator(fake_img)
            if viewpoint_condition:
                g_view_loss = opt.view_lambda * viewpoints_loss(fake_viewpoint_pred, curr_gt_viewpoints)

            if opt.with_sdf and opt.eikonal_lambda > 0:
                g_eikonal, g_minimal_surface = eikonal_loss(eikonal_term, sdf=sdf if opt.min_surf_lambda > 0 else None,
                                                            beta=opt.min_surf_beta)
                g_eikonal = opt.eikonal_lambda * g_eikonal
                if opt.min_surf_lambda > 0:
                    g_minimal_surface = opt.min_surf_lambda * g_minimal_surface

            g_gan_loss = g_nonsaturating_loss(fake_pred)
            g_loss = g_gan_loss + g_view_loss + g_eikonal + g_minimal_surface
            g_loss.backward()

        g_optim.step()
        generator.zero_grad()
        loss_dict["g"] = g_gan_loss
        loss_dict["g_view"] = g_view_loss
        loss_dict["g_eikonal"] = g_eikonal
        loss_dict["g_minimal_surface"] = g_minimal_surface

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        d_view_val = loss_reduced["d_view"].mean().item()
        g_view_val = loss_reduced["g_view"].mean().item()
        g_eikonal_loss = loss_reduced["g_eikonal"].mean().item()
        g_minimal_surface_loss = loss_reduced["g_minimal_surface"].mean().item()
        g_beta_val = g_module.renderer.sigmoid_beta.item() if opt.with_sdf else 0

        if get_rank() == 0:
            pbar.set_description(
                (f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; viewpoint: {d_view_val+g_view_val:.4f}; eikonal: {g_eikonal_loss:.4f}; surf: {g_minimal_surface_loss:.4f}")
            )

            if i % 1000 == 0:
                with torch.no_grad():
                    samples = torch.Tensor(0, 3, opt.renderer_output_size, opt.renderer_output_size)
                    step_size = 4
                    mean_latent = g_module.mean_latent(10000, device)
                    for k in range(0, opt.val_n_sample * 8, step_size):
                        _, curr_samples = g_ema([sample_z[0][k:k+step_size]],
                                                sample_cam_extrinsics[k:k+step_size],
                                                sample_focals[k:k+step_size],
                                                sample_near[k:k+step_size],
                                                sample_far[k:k+step_size],
                                                truncation=0.7,
                                                truncation_latent=mean_latent,)
                        samples = torch.cat([samples, curr_samples.cpu()], 0)

                    if i % 10000 == 0:
                        utils.save_image(samples,
                            os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'volume_renderer', f"samples/{str(i).zfill(7)}.png"),
                            nrow=int(opt.val_n_sample),
                            normalize=True,
                            value_range=(-1, 1),)

            if wandb and opt.wandb:
                wandb_log_dict = {"Generator": g_loss_val,
                                  "Discriminator": d_loss_val,
                                  "R1": r1_val,
                                  "Real Score": real_score_val,
                                  "Fake Score": fake_score_val,
                                  "D viewpoint": d_view_val,
                                  "G viewpoint": g_view_val,
                                  "G eikonal loss": g_eikonal_loss,
                                  "G minimal surface loss": g_minimal_surface_loss,
                                  }
                if opt.with_sdf:
                    wandb_log_dict.update({"Beta value": g_beta_val})

                if i % 1000 == 0:
                    wandb_grid = utils.make_grid(samples, nrow=int(opt.val_n_sample),
                                                   normalize=True, value_range=(-1, 1))
                    wandb_ndarr = (255 * wandb_grid.permute(1, 2, 0).numpy()).astype(np.uint8)
                    wandb_images = Image.fromarray(wandb_ndarr)
                    wandb_log_dict.update({"examples": [wandb.Image(wandb_images,
                                            caption="Generated samples for azimuth angles of: -35, -25, -15, -5, 5, 15, 25, 35 degrees.")]})

                wandb.log(wandb_log_dict)

            if i % 10000 == 0 or (i < 10000 and i % 1000 == 0):
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                    },
                    os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'volume_renderer', f"models_{str(i).zfill(7)}.pt")
                )
                print('Successfully saved checkpoint for iteration {}.'.format(i))

    if get_rank() == 0:
        # create final model directory
        final_model_path = 'pretrained_renderer'
        os.makedirs(final_model_path, exist_ok=True)
        torch.save(
            {
                "g": g_module.state_dict(),
                "d": d_module.state_dict(),
                "g_ema": g_ema.state_dict(),
            },
            os.path.join(final_model_path, experiment_opt.expname + '_vol_renderer.pt')
        )
        print('Successfully saved final model.')


if __name__ == "__main__":
    device = "cuda"
    opt = BaseOptions().parse()
    opt.model.freeze_renderer = False
    opt.model.no_viewpoint_loss = opt.training.view_lambda == 0.0
    opt.training.camera = opt.camera
    opt.training.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.training.style_dim = opt.model.style_dim
    opt.training.with_sdf = not opt.rendering.no_sdf
    if opt.training.with_sdf and opt.training.min_surf_lambda > 0:
        opt.rendering.return_sdf = True
    opt.training.iter = 200001
    opt.rendering.no_features_output = True

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.training.distributed = n_gpu > 1

    if opt.training.distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # create checkpoints directories
    os.makedirs(os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'volume_renderer'), exist_ok=True)
    os.makedirs(os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'volume_renderer', 'samples'), exist_ok=True)

    discriminator = VolumeRenderDiscriminator(opt.model).to(device)
    generator = Generator(opt.model, opt.rendering, full_pipeline=False).to(device)
    g_ema = Generator(opt.model, opt.rendering, ema=True, full_pipeline=False).to(device)

    g_ema.eval()
    accumulate(g_ema, generator, 0)
    g_optim = optim.Adam(generator.parameters(), lr=2e-5, betas=(0, 0.9))
    d_optim = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0, 0.9))

    opt.training.start_iter = 0

    if opt.experiment.continue_training and opt.experiment.ckpt is not None:
        if get_rank() == 0:
            print("load model:", opt.experiment.ckpt)
        ckpt_path = os.path.join(opt.training.checkpoints_dir,
                                 opt.experiment.expname,
                                 'models_{}.pt'.format(opt.experiment.ckpt.zfill(7)))
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        try:
            opt.training.start_iter = int(opt.experiment.ckpt) + 1

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        if "g_optim" in ckpt.keys():
            g_optim.load_state_dict(ckpt["g_optim"])
            d_optim.load_state_dict(ckpt["d_optim"])

    sphere_init_path = './pretrained_renderer/sphere_init.pt'
    if opt.training.no_sphere_init:
        opt.training.sphere_init = False
    elif not opt.experiment.continue_training and opt.training.with_sdf and os.path.isfile(sphere_init_path):
        if get_rank() == 0:
            print("loading sphere inititialized model")
        ckpt = torch.load(sphere_init_path, map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        opt.training.sphere_init = False
    else:
        opt.training.sphere_init = True

    if opt.training.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[opt.training.local_rank],
            output_device=opt.training.local_rank,
            broadcast_buffers=True,
            find_unused_parameters=True,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[opt.training.local_rank],
            output_device=opt.training.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])

    dataset = MultiResolutionDataset(opt.dataset.dataset_path, transform, opt.model.size,
                                     opt.model.renderer_spatial_output_dim)
    loader = data.DataLoader(
        dataset,
        batch_size=opt.training.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=opt.training.distributed),
        drop_last=True,
    )
    opt.training.dataset_name = opt.dataset.dataset_path.lower()

    # save options
    opt_path = os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'volume_renderer', f"opt.yaml")
    with open(opt_path,'w') as f:
        yaml.safe_dump(opt, f)

    # set wandb environment
    if get_rank() == 0 and wandb is not None and opt.training.wandb:
        wandb.init(project="StyleSDF")
        wandb.run.name = opt.experiment.expname
        wandb.config.dataset = os.path.basename(opt.dataset.dataset_path)
        wandb.config.update(opt.training)
        wandb.config.update(opt.model)
        wandb.config.update(opt.rendering)

    train(opt.training, opt.experiment, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
