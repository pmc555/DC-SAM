import torch
import torch.nn as nn
import torch.nn.functional as F

# fp16 min normal ≈ 6.1e-5; keep EPS representable in fp16 so
# (d_norm_sq + EPS) never collapses to (0 + 0) = 0 and causes NaN.
EPS = 1e-4

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]

        AMP-safety note: map_to_dir runs in fp16 under autocast (fast), but
        the subsequent dot-product and squared-norm can overflow fp16
        (65504² >> fp16 max). Cast x and d to fp32 for the math only.
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        # upcast to fp32 so the element-wise math never overflows
        x = x.float()
        d = d.float()
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]

        AMP-safety note: both linear layers run in fp16 under autocast (fast).
        After that, upcast to fp32 for the LeakyReLU math (dot-product and
        squared-norm can overflow fp16 and produce inf/inf = NaN).
        '''
        # Linear (fp16 under autocast — kept for speed)
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)

        # Upcast to fp32 for numerically sensitive math
        p = p.float()
        d = d.float()

        # Normalise p (zero-norm guard via eps=EPS)
        p = F.normalize(p, p=2, dim=2, eps=EPS)

        # LeakyReLU in direction d
        dotprod = (p * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm', negative_slope=0.2):
        super(VNLinearAndLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        # LeakyReLU (fp32-safe, see VNLeakyReLU.forward)
        x_out = self.leaky_relu(x)
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.num_features = num_features
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]

        AMP-safety: torch.norm in fp16 can overflow; cast to fp32 first.
        '''
        if self.num_features != 1:
            norm = torch.norm(x.float(), dim=2) + EPS   # fp32 norm, never overflows
            norm_bn = self.bn(norm)
            norm = norm.unsqueeze(2)
            norm_bn = norm_bn.unsqueeze(2)
            x = x.float() / norm * norm_bn
        return x

class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]

        AMP-safety: dot-product can overflow in fp16; cast to fp32 after linear.
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        x = x.float()
        d = d.float()
        dotprod = (x * d).sum(2, keepdim=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]], indexing='ij') + (idx,)
        x_max = x[index_tuple]
        return x_max

class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]

        AMP-safety: Gram-Schmidt orthonormalization involves squared norms and
        cross-products that overflow fp16. vn1/vn2 already return fp32 (via
        VNLinearLeakyReLU fp32 upcast); vn_lin runs in fp16 under autocast, so
        we upcast its output before the orthonormalization math.
        '''
        z0 = x
        z0 = self.vn1(z0)    # returns fp32
        z0 = self.vn2(z0)    # returns fp32
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1).float()  # upcast after linear

        if self.normalize_frame:
            # Gram-Schmidt: all math in fp32
            v1 = z0[:, 0, :]
            v1_norm = torch.sqrt((v1 * v1 + EPS).sum(1, keepdim=True))
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdim=True) * u1
            v2_norm = torch.sqrt((v2 * v2 + EPS).sum(1, keepdim=True))
            u2 = v2 / (v2_norm + EPS)

            # cross product — specify dim=1 to silence the deprecation warning
            u3 = torch.linalg.cross(u1, u2, dim=1)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        x = x.float()  # ensure einsum inputs are fp32
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        else:
            raise NotImplementedError
        return x_std, z0
