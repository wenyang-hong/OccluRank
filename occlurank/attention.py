# Adapted from Diffusers attention processors:
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEmbedding:
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

    def __call__(self, seq_len, device):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        t = torch.arange(seq_len, device=device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class OrderAwareInstanceInteraction(nn.Module):
    def __init__(self, dim, num_heads, depth=1, dropout=0.1, num_objects=None, pos_encoding_type='learned'):
        """Instance-dimension interaction used by OccluRank.

        ``pos_encoding_type`` can be ``learned``, ``rope``, or ``none``.
        The owning attention processor retains the ``self.lrt`` attribute for
        compatibility with existing checkpoint state-dict keys.
        """
        super().__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.dim = dim
        self.pos_encoding_type = pos_encoding_type

        if pos_encoding_type == 'rope':
            assert dim % 2 == 0, "RoPE requires even feature dimension"
            self.rope = RotaryPositionEmbedding(dim)
        elif pos_encoding_type == 'learned':
            assert num_objects is not None, "num_objects must be provided for 'learned' position encoding"
            self.pos_embed = nn.Embedding(num_objects, dim)
        elif pos_encoding_type is None or pos_encoding_type == 'none':
            pass
        else:
            raise ValueError(f"Unsupported pos_encoding_type: {pos_encoding_type}")

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                ),
            ]) for _ in range(depth)
        ])

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor with shape ``(B, O, T, C)``.
            mask: Optional validity mask with shape ``(B, O, T)``. A value of
                one denotes a valid in-box token and zero denotes background
                or object padding.

        Returns:
            Tensor with shape ``(B, O, T, C)``.
        """
        B, O, T, C = x.shape
        device = x.device

        if mask is not None:
            if mask.dim() != 3 or mask.shape != (B, O, T):
                raise ValueError(f"mask must be (B, O, T) = ({B}, {O}, {T}), got {mask.shape}")

            fine_mask = mask.permute(0, 2, 1).reshape(B * T, O).bool()
            invalid_mask = ~fine_mask
        else:
            fine_mask = None
            invalid_mask = None
        x_init = x.permute(0, 2, 1, 3).reshape(B * T, O, C)

        if self.pos_encoding_type == 'learned':
            assert O <= self.pos_embed.num_embeddings, \
                f"Input sequence length ({O}) exceeds max allowed ({self.pos_embed.num_embeddings})"
            positions = torch.arange(O, device=device).expand(B * T, -1)
            x_reason = x_init + self.pos_embed(positions)
        elif self.pos_encoding_type == 'rope':
            x_reason = x_init.clone()
        elif self.pos_encoding_type is None or self.pos_encoding_type == 'none':
            x_reason = x_init.clone()
        else:
            x_reason = x_init.clone()

        if self.pos_encoding_type == 'rope':
            cos, sin = self.rope(O, device)
            cos_h = cos.view(1, O, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            sin_h = sin.view(1, O, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            cos_h = sin_h = None

        for i in range(self.depth):
            ln1, attn, ln2, ff = self.blocks[i]
            x_norm = ln1(x_reason)

            W_q, W_k, W_v = attn.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, b_v = attn.in_proj_bias.chunk(3, dim=0)

            q = F.linear(x_norm, W_q, b_q)
            k = F.linear(x_norm, W_k, b_k)
            v = F.linear(x_norm, W_v, b_v)

            H = self.num_heads
            D = C // H
            q = q.view(B * T, O, H, D).transpose(1, 2)
            k = k.view(B * T, O, H, D).transpose(1, 2)
            v = v.view(B * T, O, H, D).transpose(1, 2)

            if self.pos_encoding_type == 'rope':
                q = (q * cos_h) + (rotate_half(q) * sin_h)
                k = (k * cos_h) + (rotate_half(k) * sin_h)

            scale = D ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

            if invalid_mask is not None:
                invalid_key_mask = invalid_mask.unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(invalid_key_mask, float('-inf'))
                attn_weights = F.softmax(attn_weights, dim=-1)
                # Fully masked rows produce NaNs after softmax; they represent
                # background tokens and therefore contribute zero attention.
                attn_weights = attn_weights.nan_to_num(0.0)
            else:
                attn_weights = F.softmax(attn_weights, dim=-1)
            if attn.dropout > 0:
                attn_weights = F.dropout(attn_weights, p=attn.dropout, training=self.training)

            attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.transpose(1, 2).reshape(B * T, O, C)
            attn_out = attn.out_proj(attn_out)

            x_reason = x_reason + attn_out
            x_reason = x_reason + ff(ln2(x_reason))

            if invalid_mask is not None:
                x_reason = x_reason.masked_fill(invalid_mask.unsqueeze(-1), 0.0)

        x_out = x_reason

        x_out = x_out.reshape(B, T, O, C).permute(0, 2, 1, 3)

        return x_out


class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1,
                              bias=self.bias)

    def forward(self, x):
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        concat = torch.cat((max_pool, avg), dim=1)
        output = self.conv(concat)
        output = torch.sigmoid(output) * x
        return output


class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels // self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels // self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max_pool = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max_pool.view(b, c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b, c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = torch.sigmoid(output) * x
        return output


class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
            self,
            hidden_size=None,
            cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            phrase_num_arr=None,
            ap_tokens=None,
            boxes=None,
            use_cond=True,
            *args,
            **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class OccluRankAttnProcessor2_0(torch.nn.Module):
    r"""
    OccluRank cross-attention processor for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, num_heads=None, scale=1.0, num_tokens=4,
                 enable_alpha=False, use_gate=False, max_obj=8, adapter_variant="lrt",
                 lrt_pos_encoding="learned", lrt_depth=1, lrt_num_objects=10):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        if adapter_variant not in {"base", "lrt", "lrt_reweight"}:
            raise ValueError(f"Unsupported adapter_variant: {adapter_variant}")
        if lrt_pos_encoding not in {"learned", "none", "rope"}:
            raise ValueError(f"Unsupported lrt_pos_encoding: {lrt_pos_encoding}")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.use_gate = use_gate
        self.max_obj = max_obj
        self.adapter_variant = adapter_variant
        self.lrt_pos_encoding = lrt_pos_encoding
        self.lrt_depth = lrt_depth
        self.lrt_num_objects = lrt_num_objects
        self.to_k_ap = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ap = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.q_norm = GemmaRMSNorm(hidden_size // num_heads)
        self.k_norm = GemmaRMSNorm(hidden_size // num_heads)

        # The main ``lrt`` variant applies OII after CBAM. The other variants
        # implement the scalar-reweighting ablations used in the paper.
        self.cbam = CBAM(hidden_size, 16)

        if self.adapter_variant == "base":
            self.conv = nn.Conv2d(hidden_size, 1, 1, 1)

        if self.adapter_variant in {"lrt", "lrt_reweight"}:
            self.lrt = OrderAwareInstanceInteraction(
                dim=hidden_size,
                num_heads=num_heads,
                depth=lrt_depth,
                dropout=0.1,
                # This capacity must match the learned embedding in the checkpoint.
                num_objects=lrt_num_objects,
                pos_encoding_type=lrt_pos_encoding,
            )

        if self.adapter_variant == "lrt_reweight":
            self.score_proj = nn.Linear(hidden_size, 1)

        self.enable_alpha = enable_alpha
        if enable_alpha:
            self.alpha = nn.Parameter(torch.tensor(0.))

        self._validate_variant_modules()

    def _validate_variant_modules(self):
        if self.adapter_variant == "base":
            assert hasattr(self, "conv"), "base variant must have self.conv"
            assert not hasattr(self, "lrt"), "base variant should not have self.lrt"
            assert not hasattr(self, "score_proj"), "base variant should not have self.score_proj"
        elif self.adapter_variant == "lrt":
            assert hasattr(self, "lrt"), "lrt variant must have self.lrt"
            assert not hasattr(self, "conv"), "lrt variant should not have self.conv"
            assert not hasattr(self, "score_proj"), "lrt variant should not have self.score_proj"
            assert self.lrt.pos_encoding_type == self.lrt_pos_encoding
        elif self.adapter_variant == "lrt_reweight":
            assert hasattr(self, "lrt"), "lrt_reweight variant must have self.lrt"
            assert hasattr(self, "score_proj"), "lrt_reweight variant must have self.score_proj"
            assert not hasattr(self, "conv"), "lrt_reweight variant should not have self.conv"
            assert self.lrt.pos_encoding_type == self.lrt_pos_encoding

    def extra_repr(self):
        return (
            f"variant={self.adapter_variant}, "
            f"lrt_pos_encoding={self.lrt_pos_encoding}, "
            f"lrt_depth={self.lrt_depth}, "
            f"lrt_num_objects={self.lrt_num_objects}, "
            f"max_obj={self.max_obj}, "
            f"has_conv={hasattr(self, 'conv')}, "
            f"has_lrt={hasattr(self, 'lrt')}, "
            f"has_score_proj={hasattr(self, 'score_proj')}"
        )

    def map_construction(self, bboxes_list, area, batch_size, num_heads, head_dim, dtype, device):
        """
        input:
        output: (B, heads, L, n_tokens, dim)
        """
        num_block_row_col = int(area ** 0.5)
        all_sample_background_mask_list = []
        all_sample_cond_mask_list = []
        mask_scale = []

        for bboxes in bboxes_list:
            single_sample_cond_mask = []

            for i in range(bboxes.shape[0]):
                single_box_cond_mask = torch.zeros((1, num_block_row_col, num_block_row_col), dtype=dtype,
                                                   device=device)

                current_box_start = torch.floor(bboxes[i, 0:2] * num_block_row_col)
                current_box_end = torch.ceil(bboxes[i, 2:4] * num_block_row_col)

                top_left_x = int(current_box_start[0])
                top_left_y = int(current_box_start[1])
                bottom_right_x = int(current_box_end[0])
                bottom_right_y = int(current_box_end[1])

                single_box_cond_mask[:, top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1
                single_sample_cond_mask.append(single_box_cond_mask)

            single_sample_cond_mask_torch = torch.concat(single_sample_cond_mask)
            background_map = (single_sample_cond_mask_torch.sum(axis=0) < 1).int()

            box_areas = single_sample_cond_mask_torch.sum(axis=[1, 2])
            box_area_scale = (area - background_map.sum()) / box_areas

            mask_scale.append(box_area_scale)
            all_sample_background_mask_list.append(background_map[None, :, :])
            all_sample_cond_mask_list.append(single_sample_cond_mask_torch)

        return all_sample_cond_mask_list, all_sample_background_mask_list, mask_scale

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            phrase_num_arr=None,
            ap_tokens=None,
            boxes=None,
            use_cond=True,
            *args,
            **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        ap_key = self.to_k_ap(ap_tokens)
        ap_value = self.to_v_ap(ap_tokens)

        bboxes_list = list(torch.split(boxes, phrase_num_arr, dim=0))
        ap_key = ap_key.view(ap_key.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        ap_key = self.k_norm(ap_key)

        ap_value = ap_value.view(ap_value.shape[0], -1, attn.heads, head_dim).transpose(1, 2)

        all_sample_cond_mask_list, all_sample_background_mask_list, mask_scale = self.map_construction(bboxes_list,
                                                                                                       hidden_states.shape[
                                                                                                           1],
                                                                                                       hidden_states.shape[
                                                                                                           0],
                                                                                                       attn.heads,
                                                                                                       head_dim,
                                                                                                       ap_key.dtype,
                                                                                                       hidden_states.device)

        ap_key_list = list(torch.split(ap_key, phrase_num_arr, dim=0))
        ap_value_list = list(torch.split(ap_value, phrase_num_arr, dim=0))

        num_tokens = ap_key.shape[2]
        padded_key_list = []
        padded_value_list = []
        cond_mask_list = []
        mask_scale_list = []

        for ap_key_one_sample, ap_value_one_sample, cond_mask_one_sample, m_scale in zip(ap_key_list, ap_value_list,
                                                                                         all_sample_cond_mask_list,
                                                                                         mask_scale):
            padded_key = torch.zeros(1, self.max_obj, attn.heads, num_tokens, head_dim).to(ap_key_one_sample)
            padded_value = torch.zeros(1, self.max_obj, attn.heads, num_tokens, head_dim).to(ap_value_one_sample)
            padded_mask = torch.zeros(1, self.max_obj, cond_mask_one_sample.shape[1], cond_mask_one_sample.shape[2]).to(
                cond_mask_one_sample)
            padded_mask_scale = torch.zeros(1, self.max_obj).to(m_scale)

            padded_key[:, :ap_key_one_sample.shape[0], :, :, :] = padded_key[:, :ap_key_one_sample.shape[0], :, :,
                                                                  :] + ap_key_one_sample
            padded_value[:, :ap_value_one_sample.shape[0], :, :, :] = padded_value[:, :ap_value_one_sample.shape[0], :,
                                                                      :, :] + ap_value_one_sample
            padded_mask[:, :cond_mask_one_sample.shape[0], :, :] = padded_mask[:, :cond_mask_one_sample.shape[0], :,
                                                                   :] + cond_mask_one_sample
            padded_mask_scale[:, :m_scale.shape[0]] = padded_mask_scale[:, :m_scale.shape[0]] + m_scale

            padded_key_list.append(padded_key)
            padded_value_list.append(padded_value)
            cond_mask_list.append(padded_mask)
            mask_scale_list.append(padded_mask_scale)

        padded_key = torch.concat(padded_key_list)
        padded_value = torch.concat(padded_value_list)
        cond_mask = torch.concat(cond_mask_list)
        mask_scale = torch.concat(mask_scale_list)

        query = self.q_norm(query)
        query = query.unsqueeze(1).repeat(1, self.max_obj, 1, 1, 1)

        cond_mask = cond_mask.view(batch_size, self.max_obj, -1, 1)
        attn_mask = cond_mask.view(batch_size, self.max_obj, 1, -1, 1).repeat(1, 1, attn.heads, 1, num_tokens).to(
            torch.bool)

        attn_mask = attn_mask.to(dtype=ap_key.dtype)
        attn_mask = (1.0 - attn_mask) * torch.finfo(ap_key.dtype).min

        ap_hidden_states = F.scaled_dot_product_attention(
            query, padded_key, padded_value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        ap_hidden_states = ap_hidden_states.transpose(2, 3).reshape(batch_size, self.max_obj, -1, attn.heads * head_dim)
        ap_hidden_states = ap_hidden_states * cond_mask

        # Per-object layout features after spatial masking.
        x1_ap_hidden_states = ap_hidden_states

        cbam_input = x1_ap_hidden_states.view(batch_size * self.max_obj, -1, attn.heads * head_dim).transpose(1, 2)
        num_blocks = int(hidden_states.shape[1] ** 0.5)
        cbam_input = cbam_input.reshape(-1, attn.heads * head_dim, num_blocks, num_blocks)

        x2_cbam = self.cbam(cbam_input)
        object_mask = cond_mask.squeeze(-1)

        if self.adapter_variant == "base":
            cond_score = self.conv(x2_cbam)
            mask_scale = mask_scale.view(-1, 1, 1, 1)
            cond_score = (cond_score * torch.sigmoid(mask_scale)).view(
                batch_size, self.max_obj, hidden_states.shape[1]
            ).softmax(dim=1)
            ap_hidden_states = (x1_ap_hidden_states * cond_score.unsqueeze(-1)).sum(dim=1)

        elif self.adapter_variant == "lrt":
            x2_features = x2_cbam.permute(0, 2, 3, 1).reshape(
                batch_size, self.max_obj, -1, attn.heads * head_dim
            )
            x3_lrt = self.lrt(x2_features, mask=object_mask)
            ap_hidden_states = x3_lrt.sum(dim=1)

        elif self.adapter_variant == "lrt_reweight":
            x2_features = x2_cbam.permute(0, 2, 3, 1).reshape(
                batch_size, self.max_obj, -1, attn.heads * head_dim
            )
            x3_lrt = self.lrt(x2_features, mask=object_mask)
            score_logits = self.score_proj(x3_lrt).squeeze(-1)
            score_logits = score_logits.masked_fill(object_mask <= 0, torch.finfo(score_logits.dtype).min)
            cond_score = F.softmax(score_logits, dim=1).nan_to_num(0.0)
            ap_hidden_states = (x1_ap_hidden_states * cond_score.unsqueeze(-1)).sum(dim=1)

        else:
            raise ValueError(f"Unsupported adapter_variant at runtime: {self.adapter_variant}")

        all_sample_background_mask = (1 - torch.concat(all_sample_background_mask_list)).view(batch_size, -1, 1)

        ap_hidden_states = ap_hidden_states * all_sample_background_mask
        ap_hidden_states = ap_hidden_states.to(query.dtype)

        if use_cond:
            if self.enable_alpha:
                hidden_states = hidden_states + torch.tanh(self.alpha) * self.scale * ap_hidden_states
            else:
                hidden_states = hidden_states + self.scale * ap_hidden_states

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
