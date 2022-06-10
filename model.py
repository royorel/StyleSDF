import math
import random
import trimesh
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from volume_renderer import VolumeFeatureRenderer
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from pdb import set_trace as st

from utils import (
    create_cameras,
    create_mesh_renderer,
    add_textures,
    create_depth_mesh_renderer,
)
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
from pytorch3d.transforms import matrix_to_euler_angles


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class MappingLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None, is_last=False):
        super().__init__()
        if is_last:
            weight_std = 0.25
        else:
            weight_std = 1

        self.weight = nn.Parameter(weight_std * nn.init.kaiming_normal_(torch.empty(out_dim, in_dim), a=0.2, mode='fan_in', nonlinearity='leaky_relu'))

        if bias:
            self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), a=-np.sqrt(1/in_dim), b=np.sqrt(1/in_dim)))
        else:
            self.bias = None

        self.activation = activation

    def forward(self, input):
        if self.activation != None:
            out = F.linear(input, self.weight)
            out = fused_leaky_relu(out, self.bias, scale=1)
        else:
            out = F.linear(input, self.weight, bias=self.bias)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1,
                 activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input, self.weight * self.scale,
                           bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True,
                 upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)

        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self, project=False):
        super().__init__()
        self.project = project
        self.weight = nn.Parameter(torch.zeros(1))
        self.prev_noise = None
        self.mesh_fn = None
        self.vert_noise = None

    def create_pytorch_mesh(self, trimesh):
        v=trimesh.vertices; f=trimesh.faces
        verts = torch.from_numpy(np.asarray(v)).to(torch.float32).cuda()
        mesh_pytorch = Meshes(
            verts=[verts],
            faces = [torch.from_numpy(np.asarray(f)).to(torch.float32).cuda()],
            textures=None
        )
        if self.vert_noise == None or verts.shape[0] != self.vert_noise.shape[1]:
            self.vert_noise = torch.ones_like(verts)[:,0:1].cpu().normal_().expand(-1,3).unsqueeze(0)

        mesh_pytorch = add_textures(meshes=mesh_pytorch, vertex_colors=self.vert_noise.to(verts.device))

        return mesh_pytorch

    def load_mc_mesh(self, filename, resolution=128, im_res=64):
        import trimesh

        mc_tri=trimesh.load_mesh(filename)
        v=mc_tri.vertices; f=mc_tri.faces
        mesh2=trimesh.base.Trimesh(vertices=v, faces=f)
        if im_res==64 or im_res==128:
            pytorch3d_mesh = self.create_pytorch_mesh(mesh2)
            return pytorch3d_mesh
        v,f = trimesh.remesh.subdivide(v,f)
        mesh2_subdiv = trimesh.base.Trimesh(vertices=v, faces=f)
        if im_res==256:
            pytorch3d_mesh = self.create_pytorch_mesh(mesh2_subdiv);
            return pytorch3d_mesh
        v,f = trimesh.remesh.subdivide(mesh2_subdiv.vertices,mesh2_subdiv.faces)
        mesh3_subdiv = trimesh.base.Trimesh(vertices=v, faces=f)
        if im_res==256:
            pytorch3d_mesh = self.create_pytorch_mesh(mesh3_subdiv);
            return pytorch3d_mesh
        v,f = trimesh.remesh.subdivide(mesh3_subdiv.vertices,mesh3_subdiv.faces)
        mesh4_subdiv = trimesh.base.Trimesh(vertices=v, faces=f)

        pytorch3d_mesh = self.create_pytorch_mesh(mesh4_subdiv)

        return pytorch3d_mesh

    def project_noise(self, noise, transform, mesh_path=None):
        batch, _, height, width = noise.shape
        assert(batch == 1)  # assuming during inference batch size is 1

        angles = matrix_to_euler_angles(transform[0:1,:,:3], "ZYX")
        azim = float(angles[0][1])
        elev = float(-angles[0][2])

        cameras = create_cameras(azim=azim*180/np.pi,elev=elev*180/np.pi,fov=12.,dist=1)

        renderer = create_depth_mesh_renderer(cameras, image_size=height,
                specular_color=((0,0,0),), ambient_color=((1.,1.,1.),),diffuse_color=((0,0,0),))


        if self.mesh_fn is None or self.mesh_fn != mesh_path:
            self.mesh_fn = mesh_path

        pytorch3d_mesh = self.load_mc_mesh(mesh_path, im_res=height)
        rgb, depth = renderer(pytorch3d_mesh)

        depth_max = depth.max(-1)[0].view(-1) # (NxN)
        depth_valid = depth_max > 0.
        if self.prev_noise is None:
            self.prev_noise = noise
        noise_copy = self.prev_noise.clone()
        noise_copy.view(-1)[depth_valid] = rgb[0,:,:,0].view(-1)[depth_valid]
        noise_copy = noise_copy.reshape(1,1,height,height)  # 1x1xNxN

        return noise_copy


    def forward(self, image, noise=None, transform=None, mesh_path=None):
        batch, _, height, width = image.shape
        if noise is None:
            noise = image.new_empty(batch, 1, height, width).normal_()
        elif self.project:
            noise = self.project_noise(noise, transform, mesh_path=mesh_path)

        return image + self.weight * noise


class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 upsample=False, blur_kernel=[1, 3, 3, 1], project_noise=False):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
        )

        self.noise = NoiseInjection(project=project_noise)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None, transform=None, mesh_path=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise, transform=transform, mesh_path=mesh_path)
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.upsample = upsample
        out_channels = 3
        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, out_channels, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            if self.upsample:
                skip = self.upsample(skip)

            out = out + skip

        return out


class ConvLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False,
                 blur_kernel=[1, 3, 3, 1], bias=True, activate=True):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class Decoder(nn.Module):
    def __init__(self, model_opt, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        # decoder mapping network
        self.size = model_opt.size
        self.style_dim = model_opt.style_dim * 2
        thumb_im_size = model_opt.renderer_spatial_output_dim

        layers = [PixelNorm(),
                   EqualLinear(
                       self.style_dim // 2, self.style_dim, lr_mul=model_opt.lr_mapping, activation="fused_lrelu"
                   )]

        for i in range(4):
            layers.append(
                EqualLinear(
                    self.style_dim, self.style_dim, lr_mul=model_opt.lr_mapping, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        # decoder network
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * model_opt.channel_multiplier,
            128: 128 * model_opt.channel_multiplier,
            256: 64 * model_opt.channel_multiplier,
            512: 32 * model_opt.channel_multiplier,
            1024: 16 * model_opt.channel_multiplier,
        }

        decoder_in_size = model_opt.renderer_spatial_output_dim

        # image decoder
        self.log_size = int(math.log(self.size, 2))
        self.log_in_size = int(math.log(decoder_in_size, 2))

        self.conv1 = StyledConv(
            model_opt.feature_encoder_in_channels,
            self.channels[decoder_in_size], 3, self.style_dim, blur_kernel=blur_kernel,
            project_noise=model_opt.project_noise)

        self.to_rgb1 = ToRGB(self.channels[decoder_in_size], self.style_dim, upsample=False)

        self.num_layers = (self.log_size - self.log_in_size) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[decoder_in_size]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 2 * self.log_in_size + 1) // 2
            shape = [1, 1, 2 ** (res), 2 ** (res)]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(self.log_in_size+1, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(in_channel, out_channel, 3, self.style_dim, upsample=True,
                           blur_kernel=blur_kernel, project_noise=model_opt.project_noise)
            )

            self.convs.append(
                StyledConv(out_channel, out_channel, 3, self.style_dim,
                           blur_kernel=blur_kernel, project_noise=model_opt.project_noise)
            )

            self.to_rgbs.append(ToRGB(out_channel, self.style_dim))

            in_channel = out_channel

        self.n_latent = (self.log_size - self.log_in_size) * 2 + 2

    def mean_latent(self, renderer_latent):
        latent = self.style(renderer_latent).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def styles_and_noise_forward(self, styles, noise, inject_index=None, truncation=1,
                                 truncation_latent=None, input_is_latent=False,
                                 randomize_noise=True):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if (truncation < 1):
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent[1] + truncation * (style - truncation_latent[1])
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        return latent, noise

    def forward(self, features, styles, rgbd_in=None, transform=None,
                return_latents=False, inject_index=None, truncation=1,
                truncation_latent=None, input_is_latent=False, noise=None,
                randomize_noise=True, mesh_path=None):
        latent, noise = self.styles_and_noise_forward(styles, noise, inject_index, truncation,
                                                      truncation_latent, input_is_latent,
                                                      randomize_noise)

        out = self.conv1(features, latent[:, 0], noise=noise[0],
                         transform=transform, mesh_path=mesh_path)

        skip = self.to_rgb1(out, latent[:, 1], skip=rgbd_in)

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1,
                           transform=transform, mesh_path=mesh_path)
            out = conv2(out, latent[:, i + 1], noise=noise2,
                                   transform=transform, mesh_path=mesh_path)
            skip = to_rgb(out, latent[:, i + 2], skip=skip)

            i += 2

        out_latent = latent if return_latents else None
        image = skip

        return image, out_latent


class Generator(nn.Module):
    def __init__(self, model_opt, renderer_opt, blur_kernel=[1, 3, 3, 1], ema=False, full_pipeline=True):
        super().__init__()
        self.size = model_opt.size
        self.style_dim = model_opt.style_dim
        self.num_layers = 1
        self.train_renderer = not model_opt.freeze_renderer
        self.full_pipeline = full_pipeline
        model_opt.feature_encoder_in_channels = renderer_opt.width

        if ema or 'is_test' in model_opt.keys():
            self.is_train = False
        else:
            self.is_train = True

        # volume renderer mapping_network
        layers = []
        for i in range(3):
            layers.append(
                MappingLinear(self.style_dim, self.style_dim, activation="fused_lrelu")
            )

        self.style = nn.Sequential(*layers)

        # volume renderer
        thumb_im_size = model_opt.renderer_spatial_output_dim
        self.renderer = VolumeFeatureRenderer(renderer_opt, style_dim=self.style_dim,
                                              out_im_res=thumb_im_size)

        if self.full_pipeline:
            self.decoder = Decoder(model_opt)

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent, device):
        latent_in = torch.randn(n_latent, self.style_dim, device=device)
        renderer_latent = self.style(latent_in)
        renderer_latent_mean = renderer_latent.mean(0, keepdim=True)
        if self.full_pipeline:
            decoder_latent_mean = self.decoder.mean_latent(renderer_latent)
        else:
            decoder_latent_mean = None

        return [renderer_latent_mean, decoder_latent_mean]

    def get_latent(self, input):
        return self.style(input)

    def styles_and_noise_forward(self, styles, inject_index=None, truncation=1,
                                 truncation_latent=None, input_is_latent=False):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent[0] + truncation * (style - truncation_latent[0])
                )

            styles = style_t

        return styles

    def init_forward(self, styles, cam_poses, focals, near=0.88, far=1.12):
        latent = self.styles_and_noise_forward(styles)

        sdf, target_values = self.renderer.mlp_init_pass(cam_poses, focals, near, far, styles=latent[0])

        return sdf, target_values

    def forward(self, styles, cam_poses, focals, near=0.88, far=1.12, return_latents=False,
                inject_index=None, truncation=1, truncation_latent=None,
                input_is_latent=False, noise=None, randomize_noise=True,
                return_sdf=False, return_xyz=False, return_eikonal=False,
                project_noise=False, mesh_path=None):

        # do not calculate renderer gradients if renderer weights are frozen
        with torch.set_grad_enabled(self.is_train and self.train_renderer):
            latent = self.styles_and_noise_forward(styles, inject_index, truncation,
                                                   truncation_latent, input_is_latent)

            thumb_rgb, features, sdf, mask, xyz, eikonal_term = self.renderer(cam_poses, focals, near, far, styles=latent[0], return_eikonal=return_eikonal)

        if self.full_pipeline:
            rgb, decoder_latent = self.decoder(features, latent,
                                               transform=cam_poses if project_noise else None,
                                               return_latents=return_latents,
                                               inject_index=inject_index, truncation=truncation,
                                               truncation_latent=truncation_latent, noise=noise,
                                               input_is_latent=input_is_latent, randomize_noise=randomize_noise,
                                               mesh_path=mesh_path)

        else:
            rgb = None

        if return_latents:
            return rgb, decoder_latent
        else:
            out = (rgb, thumb_rgb)
            if return_xyz:
                out += (xyz,)
            if return_sdf:
                out += (sdf,)
            if return_eikonal:
                out += (eikonal_term,)
            if return_xyz:
                out += (mask,)

            return out

############# Volume Renderer Building Blocks & Discriminator ##################
class VolumeRenderDiscConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, activate=False):
        super(VolumeRenderDiscConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=bias and not activate)

        self.activate = activate
        if self.activate:
            self.activation = FusedLeakyReLU(out_channels, bias=bias, scale=1)
            bias_init_coef = np.sqrt(1 / (in_channels * kernel_size * kernel_size))
            nn.init.uniform_(self.activation.bias, a=-bias_init_coef, b=bias_init_coef)


    def forward(self, input):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: (N,C_out,H_out,W_out）
        :return: Conv2d + activation Result
        """
        out = self.conv(input)
        if self.activate:
            out = self.activation(out)

        return out


class AddCoords(nn.Module):
    def __init__(self):
        super(AddCoords, self).__init__()

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        xx_channel = torch.arange(dim_x, dtype=torch.float32, device=input_tensor.device).repeat(1,1,dim_y,1)
        yy_channel = torch.arange(dim_y, dtype=torch.float32, device=input_tensor.device).repeat(1,1,dim_x,1).transpose(2,3)

        xx_channel = xx_channel / (dim_x - 1)
        yy_channel = yy_channel / (dim_y - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)
        out = torch.cat([input_tensor, yy_channel, xx_channel], dim=1)

        return out


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):
        super(CoordConv2d, self).__init__()

        self.addcoords = AddCoords()
        self.conv = nn.Conv2d(in_channels + 2, out_channels,
                              kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, bias=True, activate=True):
        super(CoordConvLayer, self).__init__()
        layers = []
        stride = 1
        self.activate = activate
        self.padding = kernel_size // 2 if kernel_size > 2 else 0

        self.conv = CoordConv2d(in_channel, out_channel, kernel_size,
                                padding=self.padding, stride=stride,
                                bias=bias and not activate)

        if activate:
            self.activation = FusedLeakyReLU(out_channel, bias=bias, scale=1)

        bias_init_coef = np.sqrt(1 / (in_channel * kernel_size * kernel_size))
        nn.init.uniform_(self.activation.bias, a=-bias_init_coef, b=bias_init_coef)

    def forward(self, input):
        out = self.conv(input)
        if self.activate:
            out = self.activation(out)

        return out


class VolumeRenderResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = CoordConvLayer(in_channel, out_channel, 3)
        self.conv2 = CoordConvLayer(out_channel, out_channel, 3)
        self.pooling = nn.AvgPool2d(2)
        self.downsample = nn.AvgPool2d(2)
        if out_channel != in_channel:
            self.skip = VolumeRenderDiscConv2d(in_channel, out_channel, 1)
        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.pooling(out)

        downsample_in = self.downsample(input)
        if self.skip != None:
            skip_in = self.skip(downsample_in)
        else:
            skip_in = downsample_in

        out = (out + skip_in) / math.sqrt(2)

        return out


class VolumeRenderDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        init_size = opt.renderer_spatial_output_dim
        self.viewpoint_loss = not opt.no_viewpoint_loss
        final_out_channel = 3 if self.viewpoint_loss else 1
        channels = {
            2: 400,
            4: 400,
            8: 400,
            16: 400,
            32: 256,
            64: 128,
            128: 64,
        }

        convs = [VolumeRenderDiscConv2d(3, channels[init_size], 1, activate=True)]

        log_size = int(math.log(init_size, 2))

        in_channel = channels[init_size]

        for i in range(log_size-1, 0, -1):
            out_channel = channels[2 ** i]

            convs.append(VolumeRenderResBlock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = VolumeRenderDiscConv2d(in_channel, final_out_channel, 2)

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)
        gan_preds = out[:,0:1]
        gan_preds = gan_preds.view(-1, 1)
        if self.viewpoint_loss:
            viewpoints_preds = out[:,1:]
            viewpoints_preds = viewpoints_preds.view(-1,2)
        else:
            viewpoints_preds = None

        return gan_preds, viewpoints_preds

######################### StyleGAN Discriminator ########################
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], merge=False):
        super().__init__()

        self.conv1 = ConvLayer(2 * in_channel if merge else in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(2 * in_channel if merge else in_channel, out_channel,
                              1, downsample=True, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = (out + self.skip(input)) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, opt, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        init_size = opt.size

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * opt.channel_multiplier,
            128: 128 * opt.channel_multiplier,
            256: 64 * opt.channel_multiplier,
            512: 32 * opt.channel_multiplier,
            1024: 16 * opt.channel_multiplier,
        }

        convs = [ConvLayer(3, channels[init_size], 1)]

        log_size = int(math.log(init_size, 2))

        in_channel = channels[init_size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        # minibatch discrimination
        in_channel += 1

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        # minibatch discrimination
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        if batch % group != 0:
            group = 3 if batch % 3 == 0 else 2

        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        final_out = torch.cat([out, stddev], 1)

        # final layers
        final_out = self.final_conv(final_out)
        final_out = final_out.view(batch, -1)
        final_out = self.final_linear(final_out)
        gan_preds = final_out[:,:1]

        return gan_preds
