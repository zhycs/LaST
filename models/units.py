import math

from torch import nn
import torch
from torch.fft import rfft, irfft


def autocorrelation(input, dim=0):
    """
    Computes the autocorrelation of samples at dimension ``dim``.
    Reference: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation
    """
    if (not input.is_cuda) and (not torch.backends.mkl.is_available()):
        raise NotImplementedError(
            "For CPU tensor, this method is only supported " "with MKL installed."
        )

    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = input.size(dim)
    M = next_fast_len(N)
    M2 = 2 * M

    # transpose dim with -1 for Fourier transform
    input = input.transpose(dim, -1)

    centered_signal = input - input.mean(dim=-1, keepdim=True)

    # Fourier transform
    freqvec = torch.view_as_real(rfft(centered_signal, n=M2))
    # take square of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec.pow(2).sum(-1)
    # inverse Fourier transform
    autocorr = irfft(freqvec_gram, n=M2)

    # truncate and normalize the result, then transpose back to original shape
    autocorr = autocorr[..., :N]
    autocorr = autocorr / torch.tensor(range(N, 0, -1), dtype=input.dtype, device=input.device)
    autocorr = autocorr / autocorr[..., :1]
    return autocorr.transpose(dim, -1)


class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)

        return h


def log_Normal_diag(x, mean, var, average=True, dim=None):
    log_normal = -0.5 * (torch.log(var) + torch.pow(x - mean, 2) / var ** 2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_Normal_standard(x, average=True, dim=None):
    log_normal = -0.5 * torch.pow(x, 2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def topk_mask(input, k, dim, return_mask=False):
    topk, indices = torch.topk(input, k, dim=dim)
    fill = 1 if return_mask else topk
    masked = torch.zeros_like(input, device="cuda:0").scatter_(dim, indices, fill)
    # vals, idx = input.topk(k, dim=dim)
    # topk = torch.zeros_like(input)
    # topk[idx] = 1 if return_mask else vals
    return masked


class CriticFunc(nn.Module):
    def __init__(self, x_dim, y_dim, dropout=0.1):
        super(CriticFunc, self).__init__()
        cat_dim = x_dim + y_dim
        self.critic = nn.Sequential(
            nn.Linear(cat_dim, cat_dim // 4),
            nn.ReLU(),
            nn.Linear(cat_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        cat = torch.cat((x, y), dim=-1)
        return self.critic(cat)


class NeuralFourierLayer(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len=168, pred_len=24):
        super().__init__()

        self.out_len = seq_len + pred_len
        self.freq_num = (seq_len // 2) + 1

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(torch.empty((self.freq_num, in_dim, out_dim), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.freq_num, out_dim), dtype=torch.cfloat))
        self.init_parameters()

    def forward(self, x_emb, mask=False):
        # input - b t d
        x_fft = rfft(x_emb, dim=1)[:, :self.freq_num]
        # output_fft = torch.einsum('bti,tio->bto', x_fft.type_as(self.weight), self.weight) + self.bias
        output_fft = x_fft
        if mask:
            amp = output_fft.abs().permute(0, 2, 1).reshape((-1, self.freq_num))
            output_fft_mask = topk_mask(amp, k=8, dim=1, return_mask=True)
            output_fft_mask = output_fft_mask.reshape(x_emb.shape[0], self.out_dim, self.freq_num).permute(0, 2, 1)
            output_fft = output_fft * output_fft_mask
        return irfft(output_fft, n=self.out_len, dim=1)

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)


def shape_extract(x, mode="value"):
    x_diff = x[:, 1:] - x[:, :-1]
    if mode == "binary":
        x_diff[x_diff > 0] = 1.0
        x_diff[x_diff <= 0] = 0.0
        x_diff = x_diff.type(torch.LongTensor).to(x.device)
    return x_diff


def period_sim(x, y):
    x = x.reshape(-1, x.shape[1])
    y = y.reshape(-1, y.shape[1])
    # input size: batch x length
    """ Autocorrelation """
    x_ac = autocorrelation(x, dim=1)[:, 1:]
    y_ac = autocorrelation(y, dim=1)[:, 1:]

    distance = ((x_ac - y_ac) ** 2).mean(dim=1).mean()

    return -distance


def trend_sim(x, y):
    # input size: batch x length
    x = x.reshape(-1, x.shape[1])
    y = y.reshape(-1, y.shape[1])
    x_t = shape_extract(x)
    y_t = shape_extract(y)

    """ The First Order Temporal Correlation Coefficient (CORT) """
    denominator = torch.sqrt(torch.pow(x_t, 2).sum(dim=1)) * torch.sqrt(torch.pow(y_t, 2).sum(dim=1))
    numerator = (x_t * y_t).sum(dim=1)
    cort = (numerator / denominator)

    return cort.mean()


_NEXT_FAST_LEN = {}


def next_fast_len(size):
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.
    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass

    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1
