import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightfake.layers.conformer import ConformerBlock
from lightfake.utils.common import length_to_mask, compute_statistics


class ConvolutionSubsampling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        factor: int,
        num_filters: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.stride = 2
        self.factor = factor

        in_channels = 1
        padding = (kernel_size - 1) // 2

        self.layers = nn.ModuleList()
        for _ in range(int(math.log(factor, 2))):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=num_filters,
                        kernel_size=kernel_size,
                        stride=self.stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(num_filters),
                    nn.SiLU(),
                )
            )
            in_channels = num_filters

        num_filters = 1 if len(self.layers) == 0 else num_filters
        self.proj = nn.Linear(
            num_filters * math.ceil(input_dim / self.factor),
            output_dim,
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = xs[:, None, :, :]
        masks = length_to_mask(x_lens, xs.size(2))
        masks = masks[:, None, :, None]

        for layer in self.layers:
            masks = masks[:, :, :: self.stride, :]
            xs = layer(xs) * masks

        b, c, t, f = xs.size()
        xs = xs.transpose(1, 2).contiguous().view(b, t, c * f)

        xs = self.proj(xs)
        xs = self.drop(xs)

        x_lens = torch.div(x_lens - 1, self.factor, rounding_mode="trunc")
        x_lens = (x_lens + 1).type(torch.long)

        return xs, x_lens


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, d_model: int, att_dim: int, emb_dim: int):
        super().__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(d_model * 3, att_dim, 1),
            nn.ReLU(),
            nn.BatchNorm1d(att_dim),
        )
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(att_dim, d_model, 1)
        self.norm = nn.BatchNorm1d(d_model * 2)
        self.proj = nn.Conv1d(d_model * 2, emb_dim, 1)

    def forward(self, xs, x_lens):
        xs = xs.transpose(1, 2)
        L = xs.shape[-1]

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(x_lens, L)
        mask = mask.unsqueeze(1).float()

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        total = mask.sum(dim=2, keepdim=True).float()
        mean, std = compute_statistics(xs, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, L)
        std = std.unsqueeze(2).repeat(1, 1, L)
        attn = torch.cat([xs, mean, std], dim=1)

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0.0, -1e3)
        attn = F.softmax(attn, dim=2)
        mean, std = compute_statistics(xs, attn)

        stats = torch.cat((mean, std), dim=1)
        stats = stats.unsqueeze(2)

        outs = self.proj(self.norm(stats))
        outs = outs.transpose(1, 2).squeeze(1)

        return outs


class Conformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        subsampling_factor: int,
        subsampling_filters: int,
        subsampling_kernel: int,
        encoder_num_layers_per_stage: int,
        encoder_num_heads: int,
        encoder_ffn_dim: int,
        encoder_kernel_size: int,
        pooling_att_dim: int,
        pooling_emb_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.subsampling = ConvolutionSubsampling(
            input_dim=input_dim,
            output_dim=d_model,
            factor=subsampling_factor,
            num_filters=subsampling_filters,
            kernel_size=subsampling_kernel,
            dropout=dropout,
        )

        self.encoder_1 = ConformerBlock(
            input_dim=d_model,
            num_heads=encoder_num_heads,
            ffn_dim=encoder_ffn_dim,
            num_layers=encoder_num_layers_per_stage,
            depthwise_conv_kernel_size=encoder_kernel_size,
            dropout=dropout,
        )
        self.encoder_2 = ConformerBlock(
            input_dim=d_model,
            num_heads=encoder_num_heads,
            ffn_dim=encoder_ffn_dim,
            num_layers=encoder_num_layers_per_stage,
            depthwise_conv_kernel_size=encoder_kernel_size,
            dropout=dropout,
        )
        self.encoder_3 = ConformerBlock(
            input_dim=d_model,
            num_heads=encoder_num_heads,
            ffn_dim=encoder_ffn_dim,
            num_layers=encoder_num_layers_per_stage,
            depthwise_conv_kernel_size=encoder_kernel_size,
            dropout=dropout,
        )

        self.local_pooling = nn.MaxPool1d(kernel_size=2, ceil_mode=True)
        self.global_pooling = AttentiveStatisticsPooling(
            d_model=d_model, att_dim=pooling_att_dim, emb_dim=pooling_emb_dim
        )

        self.projection = nn.Linear(d_model, pooling_emb_dim)
        self.cls_tokens = nn.Parameter(torch.empty(1, 3, d_model))
        nn.init.normal_(self.cls_tokens)

        self.embedding = nn.Linear(4 * pooling_emb_dim, pooling_emb_dim)
        self.center = nn.Parameter(torch.empty(1, pooling_emb_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        # Subsampling
        xs, x_lens = self.subsampling(xs, x_lens)
        cls_tokens = self.cls_tokens.repeat_interleave(xs.size(0), dim=0)

        h0 = torch.cat((cls_tokens[:, 0:], xs), dim=1)
        l0 = x_lens + 3

        # Stage 1
        h1, l1 = self.encoder_1(h0, l0)
        cls = h1[:, :3].clone()
        e1 = self.projection(cls[:, 0])

        h1 = self.local_pooling(h1[:, 3:].transpose(1, 2)).transpose(1, 2)
        h1 = torch.cat((cls[:, 1:], h1), dim=1)

        l1 = torch.div((l1 - 3) - 1, 2, rounding_mode="trunc")
        l1 = (l1 + 1).type(torch.long) + 2

        # Stage 2
        h2, l2 = self.encoder_2(h1, l1)
        cls = h2[:, :2].clone()
        e2 = self.projection(cls[:, 0])

        h2 = self.local_pooling(h2[:, 2:].transpose(1, 2)).transpose(1, 2)
        h2 = torch.cat((cls[:, 1:], h2), dim=1)

        l2 = torch.div((l2 - 2) - 1, 2, rounding_mode="trunc")
        l2 = (l2 + 1).type(torch.long) + 1

        # Stage 3
        h3, l3 = self.encoder_2(h2, l2)
        e3 = self.projection(h3[:, 0])
        e4 = self.global_pooling(h3, l3)

        # Final stage
        e5 = torch.cat((e1, e2, e3, e4), dim=1)
        e5 = self.embedding(e5)

        # Score
        c1 = self._classifier(e1)
        c2 = self._classifier(e2)
        c3 = self._classifier(e3)
        c4 = self._classifier(e4)
        c5 = self._classifier(e5)

        return c5, c4, c3, c2, c1

    def _classifier(self, embedding: torch.Tensor):
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(embedding, p=2, dim=1)
        y = torch.matmul(x, w.T).squeeze(1)
        return y
