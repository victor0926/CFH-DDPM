import math
import torch
from torch import nn
from typing import Optional, Tuple, Union, List

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.lin1(emb)
        emb = self.act(emb)
        emb = self.lin2(emb)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(self.time_act(t))[:, :, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)

class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_channels, max(1, in_channels // 16), kernel_size=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(max(1, in_channels // 16), in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        device = x.device  # 获取输入的设备
        self.conv1 = self.conv1.to(device)  # 确保卷积层在同一设备上
        self.conv2 = self.conv2.to(device)  # 确保卷积层在同一设备上
        x = self.avg_pool(x)  # 输出形状为 (batch_size, in_channels, 1)
        x = self.act(self.conv1(x))  # 输出形状为 (batch_size, max(1, in_channels // 16), 1)
        x = self.sigmoid(self.conv2(x))  # 输出形状为 (batch_size, in_channels, 1)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_channels: int, key_value_channels: int, num_heads: int = 1, d_k: int = None):
        super().__init__()
        if d_k is None:
            d_k = query_channels
        self.scale = d_k ** -0.5
        self.query_proj = nn.Linear(query_channels, num_heads * d_k)
        self.key_proj = nn.Linear(key_value_channels, num_heads * d_k)
        self.value_proj = nn.Linear(key_value_channels, num_heads * d_k)
        self.output_proj = nn.Linear(num_heads * d_k, query_channels)
        self.num_heads = num_heads
        self.d_k = d_k

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        batch_size, _, _ = query.shape
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_k)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_k)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_k)
        attn_weights = torch.einsum('bqhd,bkhd->bhqk', query, key) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        query = torch.einsum('bhqk,bkhd->bqhd', attn_weights, value)
        query = query.contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.output_proj(query)

class DescribeResize(nn.Module):
    def __init__(self, input_dim: int = 768, output_channels: int = 64, output_length: int = 1000):
        super(DescribeResize, self).__init__()
        self.output_channels = output_channels
        self.output_length = output_length
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_channels * output_length)
        )

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        x = x.view(-1, self.output_channels, self.output_length)
        return x


# class ConditionalFusionModule(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.channel_attention = None
#         # self.downsample = None
#         # self.cross_attention = None
#         self.channel_attention = ChannelAttentionBlock(256 + 256)
#         self.downsample = nn.Conv1d(256 + 256, 256, kernel_size=1)
#         self.cross_attention = CrossAttentionBlock(query_channels=256,
#                                                    key_value_channels=64)
#
#     def forward(self, feature_noisy: torch.Tensor, feature_condition: torch.Tensor, describe_encode: torch.Tensor):
#         device = feature_noisy.device  # 获取输入的设备
#         channel_num_n = feature_noisy.size(1)
#         channel_num_c = feature_condition.size(1)
#
#         x = torch.cat([feature_noisy, feature_condition], dim=1)
#
#         # # 初始化并移动到相同设备
#         # if self.channel_attention is None:
#         #     self.channel_attention = ChannelAttentionBlock(channel_num_n + channel_num_c).to(device)
#         # if self.downsample is None:
#         #     self.downsample = nn.Conv1d(channel_num_n + channel_num_c, channel_num_n, kernel_size=1).to(device)
#         # if self.cross_attention is None:
#         #     self.cross_attention = CrossAttentionBlock(query_channels=channel_num_n,
#         #                                                key_value_channels=describe_encode.size(1)).to(device)
#
#         weight_concat = self.channel_attention(x)
#         x = x * weight_concat
#         x = x + weight_concat
#         x = self.downsample(x)
#         x = self.cross_attention(x.permute(0, 2, 1), describe_encode.permute(0, 2, 1), describe_encode.permute(0, 2, 1))
#         return x.permute(0, 2, 1)

class ConditionalFusionModule(nn.Module):
    def __init__(self, noisy_channels: int, condition_channels: int, encode_channels: int):
        super().__init__()
        self.channel_attention = ChannelAttentionBlock(noisy_channels + condition_channels)
        self.downsample = nn.Conv1d(noisy_channels + condition_channels, noisy_channels, kernel_size=1)
        self.cross_attention = CrossAttentionBlock(query_channels=noisy_channels,
                                                   key_value_channels=encode_channels)

    def forward(self, feature_noisy: torch.Tensor, feature_condition: torch.Tensor, describe_encode: torch.Tensor):
        device = feature_noisy.device  # 获取输入的设备
        channel_num_n = feature_noisy.size(1)
        channel_num_c = feature_condition.size(1)
        encode_channels = describe_encode.size(1)

        x0 = torch.cat([feature_noisy, feature_condition], dim=1)

        weight_concat = self.channel_attention(x0)
        x = x0 * weight_concat
        # x = x + x0
        x = self.downsample(x)
        x = self.cross_attention(x.permute(0, 2, 1), describe_encode.permute(0, 2, 1), describe_encode.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = nn.Identity()
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = nn.Identity()
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x

class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(n_channels, n_channels, (4, ), (2, ), (1, ))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, n_channels, (3, ), (2, ), (1, ))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)

class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.cond_fusion = ConditionalFusionModule(256, 256, 64)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, x_cond_encode: torch.Tensor, describe_encode: torch.Tensor):
        x = self.res1(x, t)
        x = self.cond_fusion(x, x_cond_encode, describe_encode)
        x = self.res2(x, t)
        return x

class ConditionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.res_block1 = ResidualBlock(64, 64, 256)

        self.downsample1 = Downsample(64)
        self.res_block2 = ResidualBlock(64, 64, 256)

        self.downsample2 = Downsample(64)
        self.res_block3 = ResidualBlock(64, 128, 256)

        self.downsample3 = Downsample(128)
        self.res_block4 = ResidualBlock(128, 256, 256)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.initial(x)
        x_1 = self.res_block1(x, t)
        x = self.downsample1(x_1, t)
        x_2 = self.res_block2(x, t)
        x = self.downsample2(x_2, t)
        x_3 = self.res_block3(x, t)
        x = self.downsample3(x_3, t)
        x_4 = self.res_block4(x, t)
        return x_4, x_3, x_2, x_1

class MultiUNet(nn.Module):
    def __init__(self, image_channels: int = 13, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 1, 2, 2),
                 is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
                 n_blocks: int = 1):
        super().__init__()
        n_resolutions = len(ch_mults)
        self.image_proj = nn.Conv1d(image_channels, n_channels, kernel_size=3, padding=1)
        self.time_emb = TimeEmbedding(n_channels * 4)
        self.describe_resize = DescribeResize(input_dim=768, output_channels=64, output_length=1000)
        self.cond_encoder = ConditionalEncoder()

        down = []
        out_channels = n_channels
        in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)
        self.middle = MiddleBlock(out_channels, n_channels * 4)
        self.cond_fusion_4 = ConditionalFusionModule(256, 256, 64)
        self.cond_fusion_3 = ConditionalFusionModule(128, 128, 64)
        self.cond_fusion_2 = ConditionalFusionModule(64, 64, 64)
        self.cond_fusion_1 = ConditionalFusionModule(64, 64, 64)

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv1d(in_channels, 12, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, x_self_cond: torch.Tensor, describe: torch.Tensor):
        t = t.squeeze()
        t = self.time_emb(t)

        describe_encode = self.describe_resize(describe)
        x_self_cond_encoded_4, x_self_cond_encoded_3, x_self_cond_encoded_2, x_self_cond_encoded_1 = self.cond_encoder(x_self_cond, t)
        # fusion_module = ConditionalFusionModule()
        # x = fusion_module(x, x_self_cond, describe_encode)

        x = torch.cat((x, x_self_cond), dim=1)
        x = self.image_proj(x)
        h = [x]

        for m in self.down:
            x = m(x, t)
            h.append(x)

        x = self.middle(x, t, x_self_cond_encoded_4, describe_encode)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                if len(h) == 8:
                    s = h.pop()
                    s = self.cond_fusion_4(s, x_self_cond_encoded_4, describe_encode)
                    x = torch.cat((x, s), dim=1)
                elif len(h) == 6:
                    s = h.pop()
                    s = self.cond_fusion_3(s, x_self_cond_encoded_3, describe_encode)
                    x = torch.cat((x, s), dim=1)
                elif len(h) == 4:
                    s = h.pop()
                    s = self.cond_fusion_2(s, x_self_cond_encoded_2, describe_encode)
                    x = torch.cat((x, s), dim=1)
                elif len(h) == 2:
                    s = h.pop()
                    s = self.cond_fusion_1(s, x_self_cond_encoded_1, describe_encode)
                    x = torch.cat((x, s), dim=1)
                else:
                    s = h.pop()
                    x = torch.cat((x, s), dim=1)
                x = m(x, t)

        return self.final(self.act(self.norm(x)))