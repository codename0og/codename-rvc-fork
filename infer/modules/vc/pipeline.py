from typing import Union

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
# from torch.nn.utils.parametrizations import weight_norm
from torchaudio.transforms import Resample
import os
import librosa
import soundfile as sf
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
import math
from functools import partial

from einops import rearrange, repeat
from local_attention import LocalAttention
from torch import nn

os.environ["LRU_CACHE_CAPACITY"] = "3"

import sys
import traceback

from functools import lru_cache
from time import time as ttime
from typing import Union


import faiss
import parselmouth
import pyworld
import torchcrepe

from scipy import signal
import gc

import logging
logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(now_dir)


fcpe_root = os.environ.get("FCPE_ROOT")
if fcpe_root:
    fcpe_model_path = os.path.join(fcpe_root, "fcpe.pt")
else:
    # Fallback for fcpe.pt at the root of rvc ~
    fcpe_model_path = "fcpe.pt"



# FCPE handling borrowed and ported from applio; Thank you for your work and all kind of contribution! - codename.


bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}


@lru_cache
def cache_harvest_f0(input_audio_path0, input_audio_path1, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path0, input_audio_path1]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0



    # FCPE (PORT) REGION START -----------------------
def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)  # than soundfile.
    except Exception as error:
        print(f"'{full_path}' failed to load with {error}")
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 48000
        else:
            raise Exception(error)

    if len(data.shape) > 1:
        data = data[:, 0]
        assert (
            len(data) > 2
        )  # check duration of audio file is > 2 samples (because otherwise the slice operation was on the wrong dimension)

    if np.issubdtype(data.dtype, np.integer):  # if audio data is type int
        max_mag = -np.iinfo(
            data.dtype
        ).min  # maximum magnitude = min possible value of intXX
    else:  # if audio data is type fp32
        max_mag = max(np.amax(data), -np.amin(data))
        max_mag = (
            (2**31) + 1
            if max_mag > (2**15)
            else ((2**15) + 1 if max_mag > 1.01 else 1.0)
        )  # data should be either 16-bit INT, 32-bit INT or [-1 to 1] float32

    data = torch.FloatTensor(data.astype(np.float32)) / max_mag

    if (
        torch.isinf(data) | torch.isnan(data)
    ).any() and return_empty_on_exception:  # resample will crash with inf/NaN inputs. return_empty_on_exception will return empty arr instead of except
        return [], sampling_rate or target_sr or 48000
    if target_sr is not None and sampling_rate != target_sr:
        data = torch.from_numpy(
            librosa.core.resample(
                data.numpy(), orig_sr=sampling_rate, target_sr=target_sr
            )
        )
        sampling_rate = target_sr

    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


class STFT:
    def __init__(
        self,
        sr=22050,
        n_mels=80,
        n_fft=1024,
        win_size=1024,
        hop_length=256,
        fmin=20,
        fmax=11025,
        clip_val=1e-5,
    ):
        self.target_sr = sr

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}

    def get_mel(self, y, keyshift=0, speed=1, center=False, train=False):
        sampling_rate = self.target_sr
        n_mels = self.n_mels
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmin = self.fmin
        fmax = self.fmax
        clip_val = self.clip_val

        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))
        if not train:
            mel_basis = self.mel_basis
            hann_window = self.hann_window
        else:
            mel_basis = {}
            hann_window = {}

        mel_basis_key = str(fmax) + "_" + str(y.device)
        if mel_basis_key not in mel_basis:
            mel = librosa_mel_fn(
                sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
            )
            mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)

        keyshift_key = str(keyshift) + "_" + str(y.device)
        if keyshift_key not in hann_window:
            hann_window[keyshift_key] = torch.hann_window(win_size_new).to(y.device)

        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max(
            (win_size_new - hop_length_new + 1) // 2,
            win_size_new - y.size(-1) - pad_left,
        )
        if pad_right < y.size(-1):
            mode = "reflect"
        else:
            mode = "constant"
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode)
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft_new,
            hop_length=hop_length_new,
            win_length=win_size_new,
            window=hann_window[keyshift_key],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + (1e-9))
        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * win_size / win_size_new
        spec = torch.matmul(mel_basis[mel_basis_key], spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        return spec

    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect

stft = STFT()
# import fast_transformers.causal_product.causal_product_cuda

def softmax_kernel(
    data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None
):
    b, h, *_ = data.shape
    # (batch size, head, length, model_dim)

    # normalize model dim
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0

    # what is ration?, projection_matrix.shape[0] --> 266

    ratio = projection_matrix.shape[0] ** -0.5

    projection = repeat(projection_matrix, "j d -> b h j d", b=b, h=h)
    projection = projection.type_as(data)

    # data_dash = w^T x
    data_dash = torch.einsum("...id,...jd->...ij", (data_normalizer * data), projection)

    # diag_data = D**2
    diag_data = data**2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer**2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(data_dash, dim=-1, keepdim=True).values
            )
            + eps
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data + eps)
        )  # - torch.max(data_dash)) + eps)

    return data_dash.type_as(data)


def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode="reduced")
    q, r = map(lambda t: t.to(device), (q, r))

    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


class PCmer(nn.Module):
    """The encoder that is used in the Transformer model."""

    def __init__(
        self,
        num_layers,
        num_heads,
        dim_model,
        dim_keys,
        dim_values,
        residual_dropout,
        attention_dropout,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout

        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(num_layers)])

    #  METHODS  ########################################################################################################

    def forward(self, phone, mask=None):

        # apply all layers to the input
        for i, layer in enumerate(self._layers):
            phone = layer(phone, mask)
        # provide the final sequence
        return phone


# ==================================================================================================================== #
#  CLASS  _ E N C O D E R  L A Y E R                                                                                   #
# ==================================================================================================================== #

class _EncoderLayer(nn.Module):
    """One layer of the encoder.

    Attributes:
        attn: (:class:`mha.MultiHeadAttention`): The attention mechanism that is used to read the input sequence.
        feed_forward (:class:`ffl.FeedForwardLayer`): The feed-forward layer on top of the attention mechanism.
    """

    def __init__(self, parent: PCmer):
        """Creates a new instance of ``_EncoderLayer``.

        Args:
            parent (Encoder): The encoder that the layers is created for.
        """
        super().__init__()

        self.conformer = ConformerConvModule(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)

        # selfatt -> fastatt: performer!
        self.attn = SelfAttention(
            dim=parent.dim_model, heads=parent.num_heads, causal=False
        )

    #  METHODS  ########################################################################################################

    def forward(self, phone, mask=None):

        # compute attention sub-layer
        phone = phone + (self.attn(self.norm(phone), mask=mask))

        phone = phone + (self.conformer(phone))

        return phone


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, "dims must be a tuple of two dimensions"
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            # nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


def linear_attention(q, k, v):
    if v is None:
        out = torch.einsum("...ed,...nd->...ne", k, q)
        return out

    else:
        k_cumsum = k.sum(dim=-2)
        # k_cumsum = k.sum(dim = -2)
        D_inv = 1.0 / (torch.einsum("...nd,...d->...n", q, k_cumsum.type_as(q)) + 1e-8)

        context = torch.einsum("...nd,...ne->...de", k, v)
        out = torch.einsum("...de,...nd,...n->...ne", context, q, D_inv)
        return out


def gaussian_orthogonal_random_matrix(
    nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None
):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(
            nb_columns, qr_uniform_q=qr_uniform_q, device=device
        )
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(
            nb_columns, qr_uniform_q=qr_uniform_q, device=device
        )

        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones(
            (nb_rows,), device=device
        )
    else:
        raise ValueError(f"Invalid scaling {scaling}")

    return torch.diag(multiplier) @ final_matrix


class FastAttention(nn.Module):
    def __init__(
        self,
        dim_heads,
        nb_features=None,
        ortho_scaling=0,
        causal=False,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        qr_uniform_q=False,
        no_projection=False,
    ):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=dim_heads,
            scaling=ortho_scaling,
            qr_uniform_q=qr_uniform_q,
        )
        projection_matrix = self.create_projection()
        self.register_buffer("projection_matrix", projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal

    @torch.no_grad()
    def redraw_projection_matrix(self):
        projections = self.create_projection()
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)
        else:
            create_kernel = partial(
                softmax_kernel, projection_matrix=self.projection_matrix, device=device
            )

            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        if v is None:
            out = attn_fn(q, k, None)
            return out
        else:
            out = attn_fn(q, k, v)
            return out


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        heads=8,
        dim_head=64,
        local_heads=0,
        local_window_size=256,
        nb_features=None,
        feature_redraw_interval=1000,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        qr_uniform_q=False,
        dropout=0.0,
        no_projection=False,
    ):
        super().__init__()
        assert dim % heads == 0, "dimension must be divisible by number of heads"
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(
            dim_head,
            nb_features,
            causal=causal,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            qr_uniform_q=qr_uniform_q,
            no_projection=no_projection,
        )

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = (
            LocalAttention(
                window_size=local_window_size,
                causal=causal,
                autopad=True,
                dropout=dropout,
                look_forward=int(not causal),
                rel_pos_emb_config=(dim_head, local_heads),
            )
            if local_heads > 0
            else None
        )

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        name=None,
        inference=False,
        **kwargs,
    ):
        _, _, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []
        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.0)
            if cross_attend:
                pass
            else:
                out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert (
                not cross_attend
            ), "local attention is not compatible with cross attention"
            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight**2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


class FCPE(nn.Module):
    def __init__(
        self,
        input_channel=128,
        out_dims=360,
        n_layers=12,
        n_chans=512,
        use_siren=False,
        use_full=False,
        loss_mse_scale=10,
        loss_l2_regularization=False,
        loss_l2_regularization_scale=1,
        loss_grad1_mse=False,
        loss_grad1_mse_scale=1,
        f0_max=1975.5,
        f0_min=32.70,
        confidence=False,
        threshold=0.05,
        use_input_conv=True,
    ):
        super().__init__()
        if use_siren is True:
            raise ValueError("Siren is not supported yet.")
        if use_full is True:
            raise ValueError("Full model is not supported yet.")

        self.loss_mse_scale = loss_mse_scale if (loss_mse_scale is not None) else 10
        self.loss_l2_regularization = (
            loss_l2_regularization if (loss_l2_regularization is not None) else False
        )
        self.loss_l2_regularization_scale = (
            loss_l2_regularization_scale
            if (loss_l2_regularization_scale is not None)
            else 1
        )
        self.loss_grad1_mse = loss_grad1_mse if (loss_grad1_mse is not None) else False
        self.loss_grad1_mse_scale = (
            loss_grad1_mse_scale if (loss_grad1_mse_scale is not None) else 1
        )
        self.f0_max = f0_max if (f0_max is not None) else 1975.5
        self.f0_min = f0_min if (f0_min is not None) else 32.70
        self.confidence = confidence if (confidence is not None) else False
        self.threshold = threshold if (threshold is not None) else 0.05
        self.use_input_conv = use_input_conv if (use_input_conv is not None) else True

        self.cent_table_b = torch.Tensor(
            np.linspace(
                self.f0_to_cent(torch.Tensor([f0_min]))[0],
                self.f0_to_cent(torch.Tensor([f0_max]))[0],
                out_dims,
            )
        )
        self.register_buffer("cent_table", self.cent_table_b)

        # conv in stack
        _leaky = nn.LeakyReLU()
        self.stack = nn.Sequential(
            nn.Conv1d(input_channel, n_chans, 3, 1, 1),
            nn.GroupNorm(4, n_chans),
            _leaky,
            nn.Conv1d(n_chans, n_chans, 3, 1, 1),
        )

        # transformer
        self.decoder = PCmer(
            num_layers=n_layers,
            num_heads=8,
            dim_model=n_chans,
            dim_keys=n_chans,
            dim_values=n_chans,
            residual_dropout=0.1,
            attention_dropout=0.1,
        )
        self.norm = nn.LayerNorm(n_chans)

        # out
        self.n_out = out_dims
        self.dense_out = weight_norm(nn.Linear(n_chans, self.n_out))

    def forward(
        self, mel, infer=True, gt_f0=None, return_hz_f0=False, cdecoder="local_argmax"
    ):
        """
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        """
        if cdecoder == "argmax":
            self.cdecoder = self.cents_decoder
        elif cdecoder == "local_argmax":
            self.cdecoder = self.cents_local_decoder
        if self.use_input_conv:
            x = self.stack(mel.transpose(1, 2)).transpose(1, 2)
        else:
            x = mel
        x = self.decoder(x)
        x = self.norm(x)
        x = self.dense_out(x)  # [B,N,D]
        x = torch.sigmoid(x)
        if not infer:
            gt_cent_f0 = self.f0_to_cent(gt_f0)  # mel f0  #[B,N,1]
            gt_cent_f0 = self.gaussian_blurred_cent(gt_cent_f0)  # #[B,N,out_dim]
            loss_all = self.loss_mse_scale * F.binary_cross_entropy(
                x, gt_cent_f0
            )  # bce loss
            # l2 regularization
            if self.loss_l2_regularization:
                loss_all = loss_all + l2_regularization(
                    model=self, l2_alpha=self.loss_l2_regularization_scale
                )
            x = loss_all
        if infer:
            x = self.cdecoder(x)
            x = self.cent_to_f0(x)
            if not return_hz_f0:
                x = (1 + x / 700).log()
        return x

    def cents_decoder(self, y, mask=True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        rtn = torch.sum(ci * y, dim=-1, keepdim=True) / torch.sum(
            y, dim=-1, keepdim=True
        )  # cents: [B,N,1]
        if mask:
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float("-INF")
            rtn = rtn * confident_mask
        if self.confidence:
            return rtn, confident
        else:
            return rtn

    def cents_local_decoder(self, y, mask=True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        confident, max_index = torch.max(y, dim=-1, keepdim=True)
        local_argmax_index = torch.arange(0, 9).to(max_index.device) + (max_index - 4)
        local_argmax_index[local_argmax_index < 0] = 0
        local_argmax_index[local_argmax_index >= self.n_out] = self.n_out - 1
        ci_l = torch.gather(ci, -1, local_argmax_index)
        y_l = torch.gather(y, -1, local_argmax_index)
        rtn = torch.sum(ci_l * y_l, dim=-1, keepdim=True) / torch.sum(
            y_l, dim=-1, keepdim=True
        )  # cents: [B,N,1]
        if mask:
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float("-INF")
            rtn = rtn * confident_mask
        if self.confidence:
            return rtn, confident
        else:
            return rtn

    def cent_to_f0(self, cent):
        return 10.0 * 2 ** (cent / 1200.0)

    def f0_to_cent(self, f0):
        return 1200.0 * torch.log2(f0 / 10.0)

    def gaussian_blurred_cent(self, cents):  # cents: [B,N,1]
        mask = (cents > 0.1) & (cents < (1200.0 * np.log2(self.f0_max / 10.0)))
        B, N, _ = cents.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        return torch.exp(-torch.square(ci - cents) / 1250) * mask.float()


class FCPEInfer:
    def __init__(self, fcpe_model_path=None, device=None, dtype=torch.float32):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        ckpt = torch.load(fcpe_model_path, map_location=torch.device(self.device))
        self.args = DotDict(ckpt["config"])
        self.dtype = dtype
        model = FCPE(
            input_channel=self.args.model.input_channel,
            out_dims=self.args.model.out_dims,
            n_layers=self.args.model.n_layers,
            n_chans=self.args.model.n_chans,
            use_siren=self.args.model.use_siren,
            use_full=self.args.model.use_full,
            loss_mse_scale=self.args.loss.loss_mse_scale,
            loss_l2_regularization=self.args.loss.loss_l2_regularization,
            loss_l2_regularization_scale=self.args.loss.loss_l2_regularization_scale,
            loss_grad1_mse=self.args.loss.loss_grad1_mse,
            loss_grad1_mse_scale=self.args.loss.loss_grad1_mse_scale,
            f0_max=self.args.model.f0_max,
            f0_min=self.args.model.f0_min,
            confidence=self.args.model.confidence,
        )
        model.to(self.device).to(self.dtype)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self.model = model
        self.wav2mel = Wav2Mel(self.args, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def __call__(self, audio, sr, threshold=0.05):
        self.model.threshold = threshold
        audio = audio[None, :]
        mel = self.wav2mel(audio=audio, sample_rate=sr).to(self.dtype)
        f0 = self.model(mel=mel, infer=True, return_hz_f0=True)
        return f0


class Wav2Mel:

    def __init__(self, args, device=None, dtype=torch.float32):
        # self.args = args
        self.sampling_rate = args.mel.sampling_rate
        self.hop_size = args.mel.hop_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.stft = STFT(
            args.mel.sampling_rate,
            args.mel.num_mels,
            args.mel.n_fft,
            args.mel.win_size,
            args.mel.hop_size,
            args.mel.fmin,
            args.mel.fmax,
        )
        self.resample_kernel = {}

    def extract_nvstft(self, audio, keyshift=0, train=False):
        mel = self.stft.get_mel(audio, keyshift=keyshift, train=train).transpose(
            1, 2
        )  # B, n_frames, bins
        return mel

    def extract_mel(self, audio, sample_rate, keyshift=0, train=False):
        audio = audio.to(self.dtype).to(self.device)
        # resample
        if sample_rate == self.sampling_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(
                    sample_rate, self.sampling_rate, lowpass_filter_width=128
                )
            self.resample_kernel[key_str] = (
                self.resample_kernel[key_str].to(self.dtype).to(self.device)
            )
            audio_res = self.resample_kernel[key_str](audio)

        # extract
        mel = self.extract_nvstft(
            audio_res, keyshift=keyshift, train=train
        )  # B, n_frames, bins
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        if n_frames > int(mel.shape[1]):
            mel = torch.cat((mel, mel[:, -1:, :]), 1)
        if n_frames < int(mel.shape[1]):
            mel = mel[:, :n_frames, :]
        return mel

    def __call__(self, audio, sample_rate, keyshift=0, train=False):
        return self.extract_mel(audio, sample_rate, keyshift=keyshift, train=train)


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class F0Predictor(object):
    def compute_f0(self, wav, p_len):
        """
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length]
        """
        pass

    def compute_f0_uv(self, wav, p_len):
        """
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length],uv:[signal_length//hop_length]
        """
        pass


class FCPEF0Predictor(F0Predictor):
    def __init__(
        self,
        fcpe_model_path=None,
        hop_length=512,
        f0_min=50,
        f0_max=1100,
        dtype=torch.float32,
        device=None,
        sampling_rate=44100,
        threshold=0.05,
    ):
        self.fcpe = FCPEInfer(fcpe_model_path=fcpe_model_path, device=device, dtype=dtype)
        self.fcpe_model_path = fcpe_model_path
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.name = "fcpe"

    def repeat_expand(
        self,
        content: Union[torch.Tensor, np.ndarray],
        target_len: int,
        mode: str = "nearest",
    ):
        ndim = content.ndim

        if content.ndim == 1:
            content = content[None, None]
        elif content.ndim == 2:
            content = content[None]

        assert content.ndim == 3

        is_np = isinstance(content, np.ndarray)
        if is_np:
            content = torch.from_numpy(content)

        results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

        if is_np:
            results = results.numpy()

        if ndim == 1:
            return results[0, 0]
        elif ndim == 2:
            return results[0]

    def post_process(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        if pad_to is None:
            return f0

        f0 = self.repeat_expand(f0, pad_to)

        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0

        # 去掉0频率, 并线性插值
        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate

        vuv_vector = F.interpolate(vuv_vector[None, None, :], size=pad_to)[0][0]

        if f0.shape[0] <= 0:
            return (
                torch.zeros(pad_to, dtype=torch.float, device=x.device).cpu().numpy(),
                vuv_vector.cpu().numpy(),
            )
        if f0.shape[0] == 1:
            return (
                torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0]
            ).cpu().numpy(), vuv_vector.cpu().numpy()

        # 大概可以用 torch 重写?
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        # vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))

        return f0, vuv_vector.cpu().numpy()

    def compute_f0(self, wav, p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            print("fcpe p_len is None")
            p_len = x.shape[0] // self.hop_length
        f0 = self.fcpe(x, sr=self.sampling_rate, threshold=self.threshold)[0, :, 0]
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn, rtn
        return self.post_process(x, self.sampling_rate, f0, p_len)[0]

    def compute_f0_uv(self, wav, p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        f0 = self.fcpe(x, sr=self.sampling_rate, threshold=self.threshold)[0, :, 0]
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn, rtn
        return self.post_process(x, self.sampling_rate, f0, p_len)


    # FCPE (PORT) REGION END -----------------------



def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    # print(data1.max(),data2.max())
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class Pipeline(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = config.device


        # Ported from Mangio's RVC fork
    # Fork Feature: Get the best torch device to use for f0 algorithms that require a torch device. Will return the type (torch.device)
    def get_optimal_torch_device(self, index: int = 0) -> torch.device:
        # Get cuda device
        if torch.cuda.is_available():
            return torch.device(
                f"cuda:{index % torch.cuda.device_count()}"
            )  # Very fast
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        # Insert an else here to grab "xla" devices if available. TO DO later. Requires the torch_xla.core.xla_model library
        # Else wise return the "cpu" as a torch device,
        return torch.device("cpu")
        
        
        
        # Ported from Mangio's RVC fork
    # Fork Feature: Compute f0 with the crepe method
    def get_f0_crepe_computation(
        self,
        x,
        f0_min,
        f0_max,
        p_len,
        hop_length=160,  # 512 before. Hop length changes the speed that the voice jumps to a different dramatic pitch. Lower hop lengths means more pitch accuracy but longer inference time.
        model="full",  # Either use crepe-tiny "tiny" or crepe "full". Default is full
    ):
        x = x.astype(
            np.float32
        )  # fixes the F.conv2D exception. We needed to convert double to float.
        x /= np.quantile(np.abs(x), 0.999)
        torch_device = self.get_optimal_torch_device()
        audio = torch.from_numpy(x).to(torch_device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
        print("Initiating prediction with a crepe_hop_length of: " + str(hop_length))
        pitch: Tensor = torchcrepe.predict(
            audio,
            self.sr,
            hop_length,
            f0_min,
            f0_max,
            model,
            batch_size=hop_length * 2,
            device=torch_device,
            pad=True,
        )
        p_len = p_len or x.shape[0] // hop_length
        # Resize the pitch for final f0
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        f0 = np.nan_to_num(target)
        return f0  # Resized f0
        
        # Ported from Mangio's RVC fork
    def get_f0_official_crepe_computation(
        self,
        x,
        f0_min,
        f0_max,
        model="full",
    ):
        # Pick a batch size that doesn't cause memory errors on your gpu
        batch_size = 512
        # Compute pitch using first gpu
        audio = torch.tensor(np.copy(x))[None].float()
        f0, pd = torchcrepe.predict(
            audio,
            self.sr,
            self.window,
            f0_min,
            f0_max,
            model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()
        return f0


    def get_f0(
        self,
        input_audio_path0,
        input_audio_path1,
        x,
        p_len,
        f0_up_key,
        f0_method,
        filter_radius,
        crepe_hop_length,
        inp_f0=None,
    ):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            input_audio_path2wav[input_audio_path0, input_audio_path1] = x.astype(np.double)
            f0 = cache_harvest_f0(input_audio_path0, input_audio_path1, self.sr, f0_max, f0_min, 10)
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
        elif f0_method == "crepe":
            model = "full"
            batch_size = 512
            audio = torch.tensor(np.copy(x))[None].float()
            f0, pd = torchcrepe.predict(
                audio,
                self.sr,
                self.window,
                f0_min,
                f0_max,
                model,
                batch_size=batch_size,
                device=self.device,
                return_periodicity=True,
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0
            f0 = f0[0].cpu().numpy()
        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from infer.lib.rmvpe import RMVPE

                logger.info(
                    "Loading rmvpe model,%s" % "%s/rmvpe.pt" % os.environ["rmvpe_root"]
                )
                self.model_rmvpe = RMVPE(
                    "%s/rmvpe.pt" % os.environ["rmvpe_root"],
                    is_half=self.is_half,
                    device=self.device,
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
            # elif fcpe
        elif f0_method == "fcpe":
            self.model_fcpe = FCPEF0Predictor(
                fcpe_model_path=fcpe_model_path,
                f0_min=int(f0_min),
                f0_max=int(f0_max),
                dtype=torch.float32,
                device=self.device,
                sampling_rate=self.sr,
                threshold=0.03,
            )
            f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
            del self.model_fcpe
            gc.collect()

        elif f0_method == "mangio-crepe":
            f0 = self.get_f0_crepe_computation(
                x, f0_min, f0_max, p_len, crepe_hop_length
            )
        elif f0_method == "mangio-crepe-tiny":
            f0 = self.get_f0_crepe_computation(
                x, f0_min, f0_max, p_len, crepe_hop_length, "tiny"
            )

            if "privateuseone" in str(self.device):  # clean ortruntime memory
                del self.model_rmvpe.model
                del self.model_rmvpe
                logger.info("Cleaning ortruntime memory")

        f0 *= pow(2, f0_up_key / 12)
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        tf0 = self.sr // self.window  # 每秒f0点数
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]
        # with open("test_opt.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak  # 1-0

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if (
            not isinstance(index, type(None))
            and not isinstance(big_npy, type(None))
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # _, I = index.search(npy, 1)
            # npy = big_npy[I.squeeze()]

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            arg = (feats, p_len, pitch, pitchf, sid) if hasp else (feats, p_len, sid)
            audio1 = (net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()
            del hasp, arg
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        input_audio_path0,
        input_audio_path1,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        crepe_hop_length, # port
        f0_file=None,
    ):
        if (
            file_index != ""
            # and file_big_npy != ""
            # and os.path.exists(file_big_npy) == True
            and os.path.exists(file_index)
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                # big_npy = np.load(file_big_npy)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i : i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        audio_sum[t - self.t_query : t + self.t_query]
                        == audio_sum[t - self.t_query : t + self.t_query].min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
                input_audio_path0,
                input_audio_path1,
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius,
                crepe_hop_length,
                inp_f0,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if "mps" not in str(self.device) or "xpu" not in str(self.device):
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        t2 = ttime()
        times[1] += t2 - t1
        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0 == 1:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if if_f0 == 1:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if tgt_sr != resample_sr >= 16000:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt
