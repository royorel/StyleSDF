import argparse
import math
import random
import os
import yaml
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from PIL import Image
from losses import *
from options import BaseOptions
from model import Generator, Discriminator
from dataset import MultiResolutionDataset
from utils import data_sampler, requires_grad, accumulate, sample_data, make_noise, mixing_noise, generate_camera_params
from distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size

try:
    import wandb
except ImportError:
    wandb = None


def train(opt, experiment_opt, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    pbar = range(opt.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=opt.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    g_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_gan_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if opt.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = [torch.randn(opt.val_n_sample, opt.style_dim, device=device).repeat_interleave(8, dim=0)]
    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = generate_camera_params(opt.renderer_output_size, device, batch=opt.val_n_sample, sweep=True,
                                                                                         uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                                                                         elev_range=opt.camera.elev, fov_ang=opt.camera.fov,
                                                                                         dist_radius=opt.camera.dist_radius)

    for idx in pbar:
        i = idx + opt.start_iter

        if i > opt.iter:
            print("Done!")

            break

        requires_grad(generator, False)
        requires_grad(discriminator, True)
        discriminator.zero_grad()
        d_regularize = i % opt.d_reg_every == 0
        real_imgs, real_thumb_imgs = next(loader)
        real_imgs = real_imgs.to(device)
        real_thumb_imgs = real_thumb_imgs.to(device)
        noise = mixing_noise(opt.batch, opt.style_dim, opt.mixing, device)
        cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(opt.renderer_output_size, device, batch=opt.batch,
                                                                            uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                                                            elev_range=opt.camera.elev, fov_ang=opt.camera.fov,
                                                                            dist_radius=opt.camera.dist_radius)

        for j in range(0, opt.batch, opt.chunk):
            curr_real_imgs = real_imgs[j:j+opt.chunk]
            curr_real_thumb_imgs = real_thumb_imgs[j:j+opt.chunk]
            curr_noise = [n[j:j+opt.chunk] for n in noise]
            gen_imgs, _ = generator(curr_noise,
                                    cam_extrinsics[j:j+opt.chunk],
                                    focal[j:j+opt.chunk],
                                    near[j:j+opt.chunk],
                                    far[j:j+opt.chunk])

            fake_pred = discriminator(gen_imgs.detach())

            if d_regularize:
                curr_real_imgs.requires_grad = True
                curr_real_thumb_imgs.requires_grad = True

            real_pred = discriminator(curr_real_imgs)
            d_gan_loss = d_logistic_loss(real_pred, fake_pred)

            if d_regularize:
                grad_penalty = d_r1_loss(real_pred, curr_real_imgs)
                r1_loss = opt.r1 * 0.5 * grad_penalty * opt.d_reg_every
            else:
                r1_loss = torch.zeros_like(r1_loss)

            d_loss = d_gan_loss + r1_loss
            d_loss.backward()

        d_optim.step()

        loss_dict["d"] = d_gan_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()
        if d_regularize or i == opt.start_iter:
            loss_dict["r1"] = r1_loss.mean()

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        for j in range(0, opt.batch, opt.chunk):
            noise = mixing_noise(opt.chunk, opt.style_dim, opt.mixing, device)
            cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(opt.renderer_output_size, device, batch=opt.chunk,
                                                                                uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                                                                elev_range=opt.camera.elev, fov_ang=opt.camera.fov,
                                                                                dist_radius=opt.camera.dist_radius)

            fake_img, _ = generator(noise, cam_extrinsics, focal, near, far)
            fake_pred = discriminator(fake_img)
            g_gan_loss = g_nonsaturating_loss(fake_pred)

            g_loss = g_gan_loss
            g_loss.backward()


        g_optim.step()
        generator.zero_grad()

        loss_dict["g"] = g_gan_loss

        # generator path regularization
        g_regularize = (opt.g_reg_every > 0) and (i % opt.g_reg_every == 0)
        if g_regularize:
            path_batch_size = max(1, opt.batch // opt.path_batch_shrink)
            path_noise = mixing_noise(path_batch_size, opt.style_dim, opt.mixing, device)
            path_cam_extrinsics, path_focal, path_near, path_far, _ = generate_camera_params(opt.renderer_output_size, device, batch=path_batch_size,
                                                                                        uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                                                                        elev_range=opt.camera.elev, fov_ang=opt.camera.fov,
                                                                                        dist_radius=opt.camera.dist_radius)

            for j in range(0, path_batch_size, opt.chunk):
                path_fake_img, path_latents = generator(path_noise, path_cam_extrinsics,
                                                        path_focal, path_near, path_far,
                                                        return_latents=True)

                path_loss, mean_path_length, path_lengths = g_path_regularize(
                    path_fake_img, path_latents, mean_path_length
                )

                weighted_path_loss = opt.path_regularize * opt.g_reg_every * path_loss# * opt.chunk / path_batch_size
                if opt.path_batch_shrink:
                    weighted_path_loss += 0 * path_fake_img[0, 0, 0, 0]

                weighted_path_loss.backward()

            g_optim.step()
            generator.zero_grad()

            mean_path_length_avg = (reduce_sum(mean_path_length).item() / get_world_size())

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)
        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; path: {path_loss_val:.4f}")
            )

            if i % 1000 == 0 or i == opt.start_iter:
                with torch.no_grad():
                    thumbs_samples = torch.Tensor(0, 3, opt.renderer_output_size, opt.renderer_output_size)
                    samples = torch.Tensor(0, 3, opt.size, opt.size)
                    step_size = 8
                    mean_latent = g_module.mean_latent(10000, device)
                    for k in range(0, opt.val_n_sample * 8, step_size):
                        curr_samples, curr_thumbs = g_ema([sample_z[0][k:k+step_size]],
                                                           sample_cam_extrinsics[k:k+step_size],
                                                           sample_focals[k:k+step_size],
                                                           sample_near[k:k+step_size],
                                                           sample_far[k:k+step_size],
                                                           truncation=0.7,
                                                           truncation_latent=mean_latent)
                        samples = torch.cat([samples, curr_samples.cpu()], 0)
                        thumbs_samples = torch.cat([thumbs_samples, curr_thumbs.cpu()], 0)

                    if i % 10000 == 0:
                        utils.save_image(samples,
                            os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'full_pipeline', f"samples/{str(i).zfill(7)}.png"),
                            nrow=int(opt.val_n_sample),
                            normalize=True,
                            value_range=(-1, 1),)

                        utils.save_image(thumbs_samples,
                            os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'full_pipeline', f"samples/{str(i).zfill(7)}_thumbs.png"),
                            nrow=int(opt.val_n_sample),
                            normalize=True,
                            value_range=(-1, 1),)

            if wandb and opt.wandb:
                wandb_log_dict = {"Generator": g_loss_val,
                                  "Discriminator": d_loss_val,
                                  "R1": r1_val,
                                  "Real Score": real_score_val,
                                  "Fake Score": fake_score_val,
                                  "Path Length Regularization": path_loss_val,
                                  "Path Length": path_length_val,
                                  "Mean Path Length": mean_path_length,
                                  }
                if i % 5000 == 0:
                    wandb_grid = utils.make_grid(samples, nrow=int(opt.val_n_sample),
                                                   normalize=True, value_range=(-1, 1))
                    wandb_ndarr = (255 * wandb_grid.permute(1, 2, 0).numpy()).astype(np.uint8)
                    wandb_images = Image.fromarray(wandb_ndarr)
                    wandb_log_dict.update({"examples": [wandb.Image(wandb_images,
                                            caption="Generated samples for azimuth angles of: -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35 Radians.")]})

                    wandb_thumbs_grid = utils.make_grid(thumbs_samples, nrow=int(opt.val_n_sample),
                                                        normalize=True, value_range=(-1, 1))
                    wandb_thumbs_ndarr = (255 * wandb_thumbs_grid.permute(1, 2, 0).numpy()).astype(np.uint8)
                    wandb_thumbs = Image.fromarray(wandb_thumbs_ndarr)
                    wandb_log_dict.update({"thumb_examples": [wandb.Image(wandb_thumbs,
                                            caption="Generated samples for azimuth angles of: -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35 Radians.")]})

                wandb.log(wandb_log_dict)

            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                    },
                    os.path.join(opt.checkpoints_dir, experiment_opt.expname, 'full_pipeline', f"models_{str(i).zfill(7)}.pt")
                )
                print('Successfully saved checkpoint for iteration {}.'.format(i))

    if get_rank() == 0:
        # create final model directory
        final_model_path = os.path.join('full_models', opt.experiment.expname)
        os.makedirs(final_model_path, exist_ok=True)
        torch.save(
            {
                "g": g_module.state_dict(),
                "d": d_module.state_dict(),
                "g_ema": g_ema.state_dict(),
            },
            os.path.join(final_model_path, experiment_opt.expname + '.pt')
        )
        print('Successfully saved final model.')


if __name__ == "__main__":
    device = "cuda"
    opt = BaseOptions().parse()
    opt.training.camera = opt.camera
    opt.training.size = opt.model.size
    opt.training.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.training.style_dim = opt.model.style_dim
    opt.model.freeze_renderer = True

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.training.distributed = n_gpu > 1

    if opt.training.distributed:
        torch.cuda.set_device(opt.training.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # create checkpoints directories
    os.makedirs(os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'full_pipeline'), exist_ok=True)
    os.makedirs(os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'full_pipeline', 'samples'), exist_ok=True)

    discriminator = Discriminator(opt.model).to(device)
    generator = Generator(opt.model, opt.rendering).to(device)
    g_ema = Generator(opt.model, opt.rendering, ema=True).to(device)
    g_ema.eval()

    g_reg_ratio = opt.training.g_reg_every / (opt.training.g_reg_every + 1) if opt.training.g_reg_every > 0 else 1
    d_reg_ratio = opt.training.d_reg_every / (opt.training.d_reg_every + 1)

    params_g = []
    params_dict_g = dict(generator.named_parameters())
    for key, value in params_dict_g.items():
        decoder_cond = ('decoder' in key)
        if decoder_cond:
            params_g += [{'params':[value], 'lr':opt.training.lr * g_reg_ratio}]

    g_optim = optim.Adam(params_g, #generator.parameters(),
                         lr=opt.training.lr * g_reg_ratio,
                         betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
    d_optim = optim.Adam(discriminator.parameters(),
                         lr=opt.training.lr * d_reg_ratio,# * g_d_ratio,
                         betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

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
    else:
        # save configuration
        opt_path = os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'full_pipeline', f"opt.yaml")
        with open(opt_path,'w') as f:
            yaml.safe_dump(opt, f)

    if not opt.experiment.continue_training:
        if get_rank() == 0:
            print("loading pretrained renderer weights...")

        pretrained_renderer_path = os.path.join('./pretrained_renderer',
                                                opt.experiment.expname + '_vol_renderer.pt')
        try:
            ckpt = torch.load(pretrained_renderer_path, map_location=lambda storage, loc: storage)
        except:
            print('Pretrained volume renderer experiment name does not match the full pipeline experiment name.')
            vol_renderer_expname = str(input('Please enter the pretrained volume renderer experiment name:'))
            pretrained_renderer_path = os.path.join('./pretrained_renderer',
                                                    vol_renderer_expname + '.pt')
            ckpt = torch.load(pretrained_renderer_path, map_location=lambda storage, loc: storage)

        pretrained_renderer_dict = ckpt["g_ema"]
        model_dict = generator.state_dict()
        for k, v in pretrained_renderer_dict.items():
            if v.size() == model_dict[k].size():
                model_dict[k] = v

        generator.load_state_dict(model_dict)

    # initialize g_ema weights to generator weights
    accumulate(g_ema, generator, 0)

    # set distributed models
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

    if get_rank() == 0 and wandb is not None and opt.training.wandb:
        wandb.init(project="StyleSDF")
        wandb.run.name = opt.experiment.expname
        wandb.config.dataset = os.path.basename(opt.dataset.dataset_path)
        wandb.config.update(opt.training)
        wandb.config.update(opt.model)
        wandb.config.update(opt.rendering)

    train(opt.training, opt.experiment, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
