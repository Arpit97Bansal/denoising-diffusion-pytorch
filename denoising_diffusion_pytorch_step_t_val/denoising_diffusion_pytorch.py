import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from torch.autograd import Variable
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
import torchvision
#import torchattacks
from PIL import Image

import numpy as np
from tqdm import tqdm
from einops import rearrange

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)

        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        timesteps,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 3,
        with_time_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        ####
        self.flatten = nn.Flatten()
        self.step_mlp1 = nn.Linear(512 * int(image_size / 8) * int(image_size / 8), timesteps)
        self.softmax = nn.Softmax(dim=1)
        self.step_mlp2 = nn.Linear(timesteps, 1)
        ####

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        step = self.flatten(x)
        step = self.step_mlp1(step)
        step_val = self.softmax(step)
        step_val = self.step_mlp2(step_val)
        step_val = torch.clamp(step_val, min=-1.0, max=1.0)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x), step_val

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

# dataset classes

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        noise, t_val = self.denoise_fn(x, t)
        # _, t_cat = t_cat.max(1)
        t_val = t_val.squeeze()
        t_val = t_val * (self.num_timesteps/2) + (self.num_timesteps / 2)
        t_val = t_val.type(torch.int64)
        t_val = torch.clamp(t_val, min=0, max=999)
        print(t)
        print(t_val)
        t = t_val


        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon, step_val = self.denoise_fn(x_noisy, t)


        step_val = step_val.squeeze()
        target_t = (t - self.num_timesteps / 2) / (self.num_timesteps/2)
        #loss_step_val = F.huber_loss(step_val, target_t, reduction='mean', delta=0.08) #(step_val - target_t).abs().mean() + 0.5 * F.mse_loss(step_val, target_t) # criterion(step_val, target_t).half()

        ## huber loss attempt
        delta = 0.1
        l1 = (step_val - target_t).abs().mean()
        if l1 < delta:
            loss_step_val = 0.5 * F.mse_loss(step_val, target_t)
        else:
            loss_step_val = delta * (l1 - 0.5*delta)


        predicted = (step_val * self.num_timesteps/2 + self.num_timesteps / 2).int()
        correct_val = predicted.eq(t).sum().item()


        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss, loss_step_val, correct_val, l1

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)

class GaussianDiffusionIter(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        noise, t_val = self.denoise_fn(x, t)
        t_val = t_val.squeeze()
        t_val = t_val * (self.num_timesteps/2) + (self.num_timesteps / 2)
        t_val = t_val.type(torch.int64)
        t_val = torch.clamp(t_val, min=0, max=999)

        # print(t)
        # print(t_val)
        t = t_val


        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, t

    @torch.no_grad()
    def p_sample(self, x, t, done_mask, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance, t = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(t.shape[0], *((1,) * (len(x.shape) - 1)))
        x_next = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        x_next = x_next * done_mask  # zero the already done ones
        x = x * (1 - done_mask)  # take the one that are already done
        x_next = x_next + x

        new_done_mask = (1 - (t < 5).float()).reshape(t.shape[0], *((1,) * (len(x.shape) - 1)))
        done_mask = done_mask * new_done_mask

        return x_next, done_mask

    @torch.no_grad()
    def p_sample_loop(self, shape, img=None):
        device = self.betas.device

        b = shape[0]
        if img == None:
            img = torch.randn(shape, device=device)

        t = torch.full((b,), self.num_timesteps, device=device, dtype=torch.long)
        done_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(img.shape) - 1)))

        for i in tqdm(reversed(range(0, 2 * self.num_timesteps)), desc='sampling loop time step',
                      total=2 * self.num_timesteps):
            img, done_mask = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), done_mask)
            if torch.sum(done_mask) == 0:
                break
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def sample_from_img(self, batch_size=16, img=None):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), img)

    @torch.no_grad()
    def sample_x0(self, batch_size=16):

        image_size = self.image_size
        channels = self.channels
        shape = (batch_size, channels, image_size, image_size)
        device = self.betas.device

        img = torch.randn(shape).cuda()
        t = torch.full((shape[0],), self.num_timesteps, device=device, dtype=torch.long)
        done_mask = (1 - (t < 5).float()).reshape(shape[0], *((1,) * (len(img.shape) - 1)))

        for i in tqdm(reversed(range(0, 2 * self.num_timesteps)), desc='sampling loop time step',
                      total=2 * self.num_timesteps):

            noise, t = self.denoise_fn(img, None)

            t_val = t.squeeze()
            t_val = t_val * (self.num_timesteps / 2) + (self.num_timesteps / 2)
            t_val = t_val.type(torch.int64)
            t = torch.clamp(t_val, min=0, max=999)
            x_next = self.predict_start_from_noise(img, t=t, noise=noise)

            x_next = x_next * done_mask  # zero the already done ones
            img = img * (1 - done_mask)  # take the one that are already done
            img = x_next + img

            new_done_mask = (1 - (t < 5).float()).reshape(t.shape[0], *((1,) * (len(img.shape) - 1)))
            done_mask = done_mask * new_done_mask
            if torch.sum(done_mask) == 0:
                break

        return img

    def sample_opt(self, batch_size=16):

        image_size = self.image_size
        channels = self.channels
        shape = (batch_size, channels, image_size, image_size)
        device = self.betas.device

        img = torch.randn(shape).cuda()
        img_v = Variable(img, requires_grad=True)
        optim = torch.optim.Adam([img_v], lr=0.01)
        #optim = torch.optim.SGD([img_v], lr=1.0)


        t_target = torch.full((batch_size,), 0, device=device, dtype=torch.float)
        t = torch.full((batch_size,), 1000, device=device, dtype=torch.float)
        done_mask = (1 - (t < 5).float()).reshape(shape[0], *((1,) * (len(img.shape) - 1)))

        t_target = (t_target - self.num_timesteps / 2) / (self.num_timesteps / 2)
        #criterion = torch.nn.MSELoss()
        criterion = torch.nn.L1Loss()

        for i in tqdm(reversed(range(0, 2 * self.num_timesteps)), desc='sampling loop time step',
                      total=2 * self.num_timesteps):
            x_old = img_v.data
            _, t = self.denoise_fn(img_v, None)

            t = t.squeeze()

            loss = criterion(t, t_target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            t = t * (self.num_timesteps / 2) + (self.num_timesteps / 2)
            t = t.type(torch.int64)
            t = torch.clamp(t, min=0, max=999)

            x_next = img_v.data

            x_next = x_next * done_mask  # zero the already done ones
            x_old = x_old * (1 - done_mask)  # take the one that are already done
            img_v.data = x_next + x_old

            new_done_mask = (1 - (t < 5).float()).reshape(t.shape[0], *((1,) * (len(img.shape) - 1)))
            done_mask = done_mask * new_done_mask
            if torch.sum(done_mask) == 0:
                break


        # i=900
        # while(i!=0):
        #     print("#################")
        #     t_target = torch.full((batch_size,), i, device=device, dtype=torch.long)
        #
        #     for j in range(100):
        #         _, t = self.denoise_fn(img_v, None)
        #         loss = criterion(t, t_target)
        #
        #         loss.backward()
        #         optim.step()
        #         optim.zero_grad()
        #
        #         _, t = t.max(1)
        #         print(t)
        #         print(loss)
        #
        #     i=i-100

        return img_v.data

    def sample_opt_mix(self, batch_size=16):

        image_size = self.image_size
        channels = self.channels
        shape = (batch_size, channels, image_size, image_size)
        device = self.betas.device

        img = torch.randn(shape).cuda()
        img_v = Variable(img, requires_grad=True)
        optim = torch.optim.Adam([img_v], lr=0.001)
        # optim = torch.optim.SGD([img_v], lr=1.0)

        t_target = torch.full((batch_size,), 0, device=device, dtype=torch.float)
        t = torch.full((batch_size,), 1000, device=device, dtype=torch.float)
        done_mask = (1 - (t < 5).float()).reshape(shape[0], *((1,) * (len(img.shape) - 1)))

        t_target = (t_target - self.num_timesteps / 2) / (self.num_timesteps / 2)
        # criterion = torch.nn.MSELoss()
        criterion = torch.nn.L1Loss()

        for i in tqdm(reversed(range(0, 2 * self.num_timesteps)), desc='sampling loop time step',
                      total=2 * self.num_timesteps):
            x_old = img_v.data
            _, t = self.denoise_fn(img_v, None)

            t = t.squeeze()
            loss = criterion(t, t_target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            t = t * (self.num_timesteps / 2) + (self.num_timesteps / 2)
            t = t.type(torch.int64)
            t = torch.clamp(t, min=0, max=999)
            x_next = img_v.data
            x_next = x_next * done_mask  # zero the already done ones
            x_old = x_old * (1 - done_mask)  # take the one that are already done
            img_v.data = x_next + x_old
            new_done_mask = (1 - (t < 5).float()).reshape(t.shape[0], *((1,) * (len(img.shape) - 1)))
            done_mask = done_mask * new_done_mask

            # mix it with regular
            img, done_mask = self.p_sample(img_v.data, t, done_mask)
            img_v.data = img

            if torch.sum(done_mask) == 0:
                break


        return img_v.data

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon, step_val = self.denoise_fn(x_noisy, t)


        step_val = step_val.squeeze()
        target_t = (t - self.num_timesteps / 2) / (self.num_timesteps/2)
        #loss_step_val = F.huber_loss(step_val, target_t, reduction='mean', delta=0.08) #(step_val - target_t).abs().mean() + 0.5 * F.mse_loss(step_val, target_t) # criterion(step_val, target_t).half()

        ## huber loss attempt
        delta = 0.1
        l1 = (step_val - target_t).abs().mean()
        if l1 < delta:
            loss_step_val = 0.5 * F.mse_loss(step_val, target_t)
        else:
            loss_step_val = delta * (l1 - 0.5*delta)


        predicted = (step_val * self.num_timesteps/2 + self.num_timesteps / 2).int()
        correct_val = predicted.eq(t).sum().item()


        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss, loss_step_val, correct_val, l1

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png', 'JPEG']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            #transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        return self.transform(img)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        do_load = False,
        test_folder = None
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = Dataset(folder, image_size)
        if test_folder != None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                # transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()
            ])
            self.train_ds = torchvision.datasets.ImageFolder(root=folder, transform=self.transform)
            self.test_ds = torchvision.datasets.ImageFolder(root=test_folder, transform=self.transform)
        else:
            self.train_ds = None
            self.test_ds = None
        #print("Debug", len(self.ds))
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0
        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()

        if do_load:
            self.load(0)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                # loss_recons, loss_step_cat, correct_cat, loss_step_val, correct_val = self.model(data)
                # if self.step < self.train_num_steps//7000:
                #     print("Category")
                #     loss = loss_recons + loss_step_cat + loss_step_val
                # else:
                #     loss = loss_recons + loss_step_val
                # print(f'{self.step}: {loss_recons.item()}, {loss_step_cat.item()}, {correct_cat}, {loss_step_val}, {correct_val}')

                loss_recons, loss_step_val, correct_val, l1 = self.model(data)
                loss = loss_recons + loss_step_val
                print(f'{self.step}: {loss_recons.item()}, {loss_step_val}, {correct_val}, {l1}')

                backwards(loss / self.gradient_accumulate_every, self.opt)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = num_to_groups(36, self.batch_size)
                all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = 6)
                self.save(0)

            self.step += 1

        print('training completed')


    def train_predictor(self, model, lr):
        model = model.cuda()
        train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(5):
            running_loss=0
            train_accuracy=0
            test_accuracy=0

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                max_vals, max_indices = torch.max(outputs, 1)
                correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
                train_accuracy += 100 * correct

            running_loss /= len(train_loader)
            train_accuracy /= len(train_loader)
            model.eval()

            for i, data in enumerate(test_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                max_vals, max_indices = torch.max(outputs, 1)
                correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
                test_accuracy += 100 * correct

            test_accuracy /= len(test_loader)

            print(epoch)
            print(running_loss, train_accuracy, test_accuracy)

            data = {
                'model': model.state_dict()
            }
            torch.save(data, str(self.results_folder / f'model-prediction.pt'))



    def test(self):

        print("Normal")
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
        all_images = torch.cat(all_images_list, dim=0)
        temp = all_images
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test.png'), nrow = 6)

        print("Add noise of 0.01 to Normal")
        temp1 = temp + 0.01*torch.randn_like(temp)
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp1), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-1.png'), nrow=6)

        print("Add noise of 0.1 to Normal")
        temp2 = temp + 0.1 * torch.randn_like(temp)
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp2), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-2.png'), nrow=6)


        print("Add noise of 0.5 to Normal")
        temp3 = temp + 0.5 * torch.randn_like(temp)
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp3), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-3.png'), nrow=6)

        print("Add noise of 1.0 to Normal")
        temp4 = temp + 1.0 * torch.randn_like(temp)
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp4), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-4.png'), nrow=6)

        print("Add noise of 1.5 to Normal")
        temp5 = temp + 1.5 * torch.randn_like(temp)
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp5), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-5.png'), nrow=6)


        print("Add noise of 2.0 to Normal")
        temp6 = temp + 2.0 * torch.randn_like(temp)
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp6), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-6.png'), nrow=6)


        print("Add noise of 4.0 to Normal")
        temp7 = temp + 4.0 * torch.randn_like(temp)
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp7), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-7.png'), nrow=6)

        print("Ones")
        temp8 = torch.ones_like(temp) * 2 - 1
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp8), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-ones.png'), nrow=6)

        print("Zeros")
        temp9 = torch.zeros_like(temp) * 2 - 1
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp9), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-zeros.png'), nrow=6)


        print("Only optimization")
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_opt(batch_size=n), batches))
        all_images = torch.cat(all_images_list, dim=0)
        temp = all_images
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-8.png'), nrow=6)

        print("Add noise to results of only optimization")
        temp9 = temp + torch.randn_like(temp)
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp9), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-9.png'), nrow=6)


        print("Mix of optimization and diffusion")
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_opt_mix(batch_size=n), batches))
        all_images = torch.cat(all_images_list, dim=0)
        temp = all_images
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-10.png'), nrow=6)

        print("Add noise to the result of mix of optimization and diffusion")
        temp9 = temp + torch.randn_like(temp)
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp9), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-11.png'), nrow=6)

        print("Directly estimating X0")
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_x0(batch_size=n), batches))
        all_images = torch.cat(all_images_list, dim=0)
        temp = all_images
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-12.png'), nrow=6)

        print("Add noise to the results of Directly estimating X0")
        temp10 = temp + torch.randn_like(temp)
        batches = num_to_groups(36, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp10), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-test-13.png'), nrow=6)

        print('testing completed')

    def test_denoise(self):
        print("Test")
        test_loader = torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True,
                                                  pin_memory=True)
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            all_images = inputs
            all_images = all_images * 2 - 1
            break

        temp = all_images
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'original.png'), nrow=6)

        print("Add noise of 0.5 to Normal")
        temp3 = temp + 0.5 * torch.randn_like(temp)
        all_images = (temp3 + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'original_noise_0_5.png'), nrow=6)

        batches = [self.batch_size]
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp3), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'original_denoise_0_5.png'), nrow=6)

        print("Add noise of 1.0 to Normal")
        temp3 = temp + 1.0 * torch.randn_like(temp)
        all_images = (temp3 + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'original_noise_1_0.png'), nrow=6)

        batches = [self.batch_size]
        all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=temp3), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'original_denoise_1_0.png'), nrow=6)

    def test_adversarial(self, model):

        import torchattacks

        model = model.cuda()
        train_loader = torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True,
                                                   pin_memory=True)

        data = torch.load(str(self.results_folder / f'model-prediction.pt'))
        model.load_state_dict(data['model'])
        model.eval()
        #atk = torchattacks.PGD(model, eps=12 / 255, alpha=1 / 255, steps=40)
        atk = torchattacks.CW(model, c=2)

        orig_acc = 0
        adv_acc = 0
        dn_acc = 0
        adv_dn_acc = 0
        cnt = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            max_vals, max_indices = torch.max(outputs, 1)
            correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
            orig_acc += 100 * correct
            if i==0:
                print("Original Prediction")
                print(correct)
                utils.save_image(inputs, str(self.results_folder / f'original.png'), nrow=6)

            adv_images = atk(inputs, labels)
            outputs = model(adv_images)
            max_vals, max_indices = torch.max(outputs, 1)
            correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
            adv_acc += 100*correct
            if i==0:
                print("Adversarial Prediction")
                print(correct)
                utils.save_image(inputs, str(self.results_folder / f'adversarial.png'), nrow=6)




            batches = [self.batch_size]
            #convert this original image to same input as for diffusion
            inputs = inputs*2 - 1
            inp_noisy = inputs + 1.0 * torch.randn_like(inputs)
            all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=inp_noisy), batches))
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1) * 0.5
            if i==0:
                utils.save_image(all_images, str(self.results_folder / f'original_denoise_1_0.png'), nrow=6)

            outputs = model(all_images)
            max_vals, max_indices = torch.max(outputs, 1)
            correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
            dn_acc += 100 * correct
            if i==0:
                print("Original Denoise Prediction")
                print(correct)




            adv_images = adv_images * 2 - 1
            adv_noisy = adv_images + 1.0 * torch.randn_like(adv_images)
            all_images_list = list(map(lambda n: self.ema_model.sample_from_img(batch_size=n, img=adv_noisy), batches))
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1) * 0.5
            if i==0:
                utils.save_image(all_images, str(self.results_folder / f'adversarial_denoise_1_0.png'), nrow=6)

            outputs = model(all_images)
            max_vals, max_indices = torch.max(outputs, 1)
            correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
            adv_dn_acc += 100 * correct
            if i==0:
                print("Adversarial Denoise Prediction")
                print(correct)

            cnt+= 1
            print("Count")
            print(cnt)

            print("Original Accuracy")
            print(orig_acc/cnt)

            print("Adversarial Accuracy")
            print(adv_acc/cnt)

            print("Denoised Accuracy")
            print(dn_acc/cnt)

            print("Denoised Adversarial Accuracy")
            print(adv_dn_acc/cnt)
