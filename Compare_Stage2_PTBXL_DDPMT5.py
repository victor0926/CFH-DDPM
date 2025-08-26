import math
import torch
import torch.nn.functional as F
from torch import nn
from fastai.vision.all import *
from fastai.vision.models.xresnet import xresnet18
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

def plot_ecg_signals(x_in, y_in, noise, x_noisy, x_recon, x_restore):
    # 提取第一条数据并使用 .detach() 分离张量
    x_in_first = x_in[17].cpu().detach().numpy()
    y_in_first = y_in[17].cpu().detach().numpy()
    noise_first = noise[17].cpu().detach().numpy()
    x_noisy_first = x_noisy[17].cpu().detach().numpy()
    x_recon_first = x_recon[17].cpu().detach().numpy()
    x_restore_first = x_restore[17].cpu().detach().numpy()

    def plot_12_leads(ecg_data, title, ax):
        for i in range(6):
            ax[i, 0].plot(ecg_data[2*i], color='black')  # 黑色线条
            ax[i, 0].set_title(f'Lead {2*i+1}')
            ax[i, 0].set_facecolor('white')  # 粉红色背景
            ax[i, 0].grid(True, color='#FFC0CB')  # 白色网格

            ax[i, 1].plot(ecg_data[2*i+1], color='black')  # 黑色线条
            ax[i, 1].set_title(f'Lead {2*i+2}')
            ax[i, 1].set_facecolor('white')  # 粉红色背景
            ax[i, 1].grid(True, color='#FFC0CB')  # 白色网格

        fig.suptitle(title)

    # 创建子图布局
    fig, axes = plt.subplots(6, 2, figsize=(36, 18))

    # 绘制 x_in 的12导联ECG
    plot_12_leads(x_in_first, 'x_in (First Data)', axes)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(6, 2, figsize=(36, 18))
    # 绘制 noise 的12导联ECG
    plot_12_leads(noise_first, 'noise (First Data)', axes)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(6, 2, figsize=(36, 18))
    # 绘制 x_noisy 的12导联ECG
    plot_12_leads(x_noisy_first, 'x_noisy (First Data)', axes)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(6, 2, figsize=(36, 18))
    # 绘制 x_recon 的12导联ECG
    plot_12_leads(x_recon_first, 'x_recon (First Data)', axes)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(6, 2, figsize=(36, 18))
    # 绘制 x_restore 的12导联ECG
    plot_12_leads(x_restore_first, 'x_restore (First Data)', axes)
    plt.tight_layout()
    plt.show()

    # 对 y_in 进行单独绘制，因为它只有一个导联
    plt.figure(figsize=(12, 4))
    plt.plot(y_in_first[0], color='black')  # 黑色线条
    plt.gca().set_facecolor('white')  # 粉红色背景
    plt.grid(True, color='#FFC0CB')  # 白色网格
    plt.title('y_in (First Data)')
    plt.show()

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Conv1dXResNet(nn.Sequential):
    def __init__(self):
        layers = [ConvLayer(12, 64, ks=7, stride=2, ndim=1),
                  ConvLayer(64, 32, ks=3, stride=1, ndim=1)]
        super().__init__(*layers)

def create_xresnet1d(arch):
    model = arch(n_out=23, pretrained=False, act_cls=Mish, ndim=1)
    model[0] = Conv1dXResNet()
    return model

class CustomXResNet1d(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.model = create_xresnet1d(arch)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 512)
        self.fc2 = nn.Linear(64, 512)
        self.fc3 = nn.Linear(64, 512)
        self.fc4 = nn.Linear(64, 512)

    def forward(self, x, return_features=False):
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in [1, 2, 3, 4]:
                features.append(x)
        if return_features:
            pooled_features = [self.pool(f).view(f.size(0), -1) for f in features]
            v1 = self.fc1(pooled_features[0])
            v2 = self.fc2(pooled_features[1])
            v3 = self.fc3(pooled_features[2])
            v4 = self.fc4(pooled_features[3])
            return x, [v1, v2, v3, v4]
        else:
            return x

class DDPM(nn.Module):
    def __init__(self, base_model, config, device, conditional=True):
        super().__init__()
        self.device = device
        self.model = base_model
        self.config = config
        self.device = device
        self.conditional = conditional

        self.loss_func = nn.L1Loss(reduction='mean').to(device)
        self.classification_model = CustomXResNet1d(xresnet18).to(device)
        self.classification_loss_func = nn.KLDivLoss(reduction='batchmean')
        self.classification_loss_weight = 0.01
        self.contrastive_loss_weight = 0.01

        # 加载预训练权重
        checkpoint_path = 'F:/Paper03_Project/P240801_DDPM_Divide/PreTrainModel/XResNet_PTBXL_Temp5/best_model_epoch_96.pth'
        state_dict = torch.load(checkpoint_path)
        state_dict = {k: v for k, v in state_dict.items() if k in self.classification_model.state_dict()}
        self.classification_model.load_state_dict(state_dict, strict=False)

        config_diff = config["diffusion"]

        self.num_steps = config_diff["num_steps"]

        self.set_new_noise_schedule(config_diff, device)

    def make_beta_schedule(self, schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas

    def set_new_noise_schedule(self, config_diff, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = self.make_beta_schedule(schedule=config_diff["schedule"], n_timesteps=config_diff["num_steps"],
                                        start=config_diff["beta_start"], end=config_diff["beta_end"])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, describe=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.model(x, noise_level, condition_x, describe))  # Unet
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.model(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, condition_x=None, describe=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, describe=describe)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        result = model_mean + noise * (0.5 * model_log_variance).exp()
        return result

    @torch.no_grad()
    def p_sample_loop(self, x_in, describe, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_steps // 10))
        x = x_in
        shape = (x.shape[0], 12, 1000)
        cur_x = torch.randn(shape, device=device)
        ret_x = [cur_x]
        for i in reversed(range(0, self.num_steps)):
            cur_x = self.p_sample(cur_x, i, condition_x=x, describe=describe)
            if i % sample_inter == 0:
                ret_x.append(cur_x)

        if continous:
            return ret_x
        else:
            return ret_x[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, shape=[1, 512], describe=None, continous=False):
        return self.p_sample_loop((batch_size, shape[0], shape[1]), describe, continous)

    @torch.no_grad()
    def denoising(self, x_in, describe, continous=False):
        return self.p_sample_loop(x_in, describe, continous)

    def q_sample_loop(self, x_start, continous=False):
        sample_inter = (1 | (self.num_steps // 10))
        ret_x = [x_start]
        cur_x = x_start
        for t in range(1, self.num_steps + 1):
            B, C, L = cur_x.shape
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[t - 1],
                    self.sqrt_alphas_cumprod_prev[t],
                    size=B
                )
            ).to(cur_x.device)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
                B, -1)

            noise = torch.randn_like(cur_x)
            cur_x = self.q_sample(
                x_start=cur_x, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1), noise=noise)
            if t % sample_inter == 0:
                ret_x.append(cur_x)
        if continous:
            return ret_x
        else:
            return ret_x[-1]

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def restore_x0(self, xt, epsilon, continuous_sqrt_alpha_cumprod):
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - continuous_sqrt_alpha_cumprod ** 2)

        x0 = (xt - sqrt_one_minus_alpha_cumprod * epsilon) / continuous_sqrt_alpha_cumprod
        return x0

    def info_nce_loss(self, anchor, positive, negatives, temperature=5):
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        positive_logit = torch.sum(anchor * positive, dim=-1, keepdim=True) / temperature
        negative_logits = torch.einsum('nc,nkc->nk', [anchor, negatives]) / temperature

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def p_losses(self, x_in, y_in, describe, noise=None):
        x_start = x_in
        B, C, L = x_start.shape
        t = np.random.randint(1, self.num_steps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=B
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            B, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1), noise=noise)

        x_recon = self.model(x_noisy, continuous_sqrt_alpha_cumprod, y_in, describe)  # Unet

        # 计算L1损失
        l1_loss = self.loss_func(noise, x_recon)

        # 使用分类模型计算分类损失
        self.classification_model.eval()  # 确保分类模型在评估模式
        with torch.no_grad():
            x_restore = self.restore_x0(xt=x_noisy, epsilon=x_recon, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1))
            logits_real, features_real = self.classification_model(x_in, return_features=True)
            logits_recon, features_recon = self.classification_model(x_restore, return_features=True)

        probs_real = F.softmax(logits_real, dim=-1)
        log_probs_recon = F.log_softmax(logits_recon, dim=-1)
        classification_loss = self.classification_loss_func(log_probs_recon, probs_real)

        # 计算对比损失
        batch_size = x_in.size(0)
        negative_indices = torch.randperm(batch_size).repeat_interleave(16).view(batch_size, -1)
        negatives = x_in[negative_indices].view(batch_size, 16, *x_in.shape[1:])
        _, features_negatives = self.classification_model(negatives.view(-1, *x_in.shape[1:]), return_features=True)
        features_negatives = [f.view(batch_size, 16, -1) for f in features_negatives]

        contrastive_loss = 0
        for f_recon, f_real, f_neg in zip(features_recon, features_real, features_negatives):
            # f_recon = f_recon.unsqueeze(1)  # (batch_size, 1, feature_dim)
            # f_real = f_real.unsqueeze(1)    # (batch_size, 1, feature_dim)
            contrastive_loss += self.info_nce_loss(f_recon, f_real, f_neg)

        # plot_ecg_signals(x_in, y_in, noise, x_noisy, x_recon, x_restore)
        # 总损失
        total_loss = l1_loss + self.classification_loss_weight * classification_loss + self.contrastive_loss_weight * contrastive_loss
        # total_loss = l1_loss + self.classification_loss_weight * classification_loss
        # total_loss = l1_loss + self.classification_loss_weight * classification_loss

        return total_loss

    def forward(self, x, y, describe, *args, **kwargs):
        return self.p_losses(x, y, describe, *args, **kwargs)
