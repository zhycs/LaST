from models.units import *
import math
import torch
from torch import nn
from torch.autograd import Variable


class FeedNet(nn.Module):
    def __init__(self, in_dim, out_dim, type="mlp", n_layers=1, inner_dim=None, activaion=None, dropout=0.1):
        super(FeedNet, self).__init__()
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.type = type
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer_in = in_dim if i == 0 else inner_dim[i - 1]
            layer_out = out_dim if i == n_layers - 1 else inner_dim[i]
            if type == "mlp":
                self.layers.append(nn.Linear(layer_in, layer_out))
            else:
                raise Exception("KeyError: Feedward Net keyword error. Please use word in ['mlp']")
            if i != n_layers - 1 and activaion is not None:
                self.layers.append(activaion)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class VarUnit(nn.Module):
    def __init__(self, in_dim, z_dim, VampPrior=False, pseudo_dim=201, device="cuda:0"):
        super(VarUnit, self).__init__()

        self.in_dim = in_dim
        self.z_dim = z_dim
        self.prior = VampPrior
        self.device = device

        self.loc_net = FeedNet(in_dim, z_dim, type="mlp", n_layers=1)
        self.var_net = nn.Sequential(
            FeedNet(in_dim, z_dim, type="mlp", n_layers=1),
            nn.Softplus()
        )

        if self.prior:
            self.pseudo_dim = pseudo_dim
            self.pseudo_mean = 0
            self.pseudo_std = 0.01
            self.add_pseudoinputs()

        self.critic_xz = CriticFunc(z_dim, in_dim)

    def add_pseudoinputs(self):
        self.idle_input = Variable(torch.eye(self.pseudo_dim, self.pseudo_dim, dtype=torch.float64, device=self.device),
                                   requires_grad=False).cuda()

        nonlinearity = nn.ReLU()
        self.means = NonLinear(self.pseudo_dim, self.in_dim, bias=False, activation=nonlinearity)
        self.normal_init(self.means.linear, self.pseudo_mean, self.pseudo_std)

    def normal_init(self, m, mean=0., std=0.01):
        m.weight.data.normal_(mean, std)

    def log_p_z(self, z):
        if self.prior:
            C = self.pseudo_dim
            X = self.means(self.idle_input).unsqueeze(dim=0)

            z_p_mean = self.loc_net(X)
            z_p_var = self.var_net(X)

            # expand z
            z_expand = z.unsqueeze(1)
            means = z_p_mean.unsqueeze(0)
            vars = z_p_var.unsqueeze(0)

            if len(z.shape) > 3:
                means = means.unsqueeze(-2).repeat(1, 1, 1, z.shape[-2], 1)
                vars = vars.unsqueeze(-2).repeat(1, 1, 1, z.shape[-2], 1)

            a = log_Normal_diag(z_expand, means, vars, dim=2) - math.log(C)
            a_max, _ = torch.max(a, 1)
            log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1
        else:
            log_prior = log_Normal_standard(z, dim=1)
        return log_prior

    def compute_KL(self, z_q, z_q_mean, z_q_var):
        log_p_z = self.log_p_z(z_q)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_var, dim=1)
        KL = -(log_p_z - log_q_z)

        return KL.mean()

    def compute_MLBO(self, x, z_q, method="our"):
        idx = torch.randperm(z_q.shape[0])
        z_q_shuffle = z_q[idx].view(z_q.size())
        if method == "MINE":
            mlbo = self.critic_xz(x, z_q).mean() - torch.log(
                torch.exp(self.critic_xz(x, z_q_shuffle)).squeeze(dim=-1).mean(dim=-1)).mean()
        else:
            point = 1 / torch.exp(self.critic_xz(x, z_q_shuffle)).squeeze(dim=-1).mean()
            point = point.detach()

            if len(x.shape) == 3:
                mlbo = self.critic_xz(x, z_q) - point * torch.exp(
                    self.critic_xz(x, z_q_shuffle))  # + 1 + torch.log(point)
            else:
                mlbo = self.critic_xz(x, z_q) - point * torch.exp(self.critic_xz(x, z_q_shuffle))

        return mlbo.mean()

    def forward(self, x, return_para=True):
        mean, var = self.loc_net(x), self.var_net(x)
        qz_gaussian = torch.distributions.Normal(loc=mean, scale=var)
        qz = qz_gaussian.rsample()  # mu+sigma*epsilon
        return (qz, mean, var) if return_para else qz


class CalculateMubo(nn.Module):
    def __init__(self, x_dim, y_dim, dropout=0.1):
        super().__init__()
        self.critic_st = CriticFunc(x_dim, y_dim, dropout)

    def forward(self, x_his, var_net_t, var_net_s):
        zs, zt = var_net_s(x_his, return_para=False), var_net_t(x_his, return_para=False)
        idx = torch.randperm(zt.shape[0])
        zt_shuffle = zt[idx].view(zt.size())
        f_st = self.critic_st(zs, zt)
        f_s_t = self.critic_st(zs, zt_shuffle)

        # if len(x_his.shape) == 3:
        #     b, t, d = zs.shape
        #     q_zs = zs.repeat(len(zs), 1, 1)
        #     q_zt = zs.unsqueeze(dim=1).repeat(1, len(zs), 1, 1).reshape(-1, t, d)
        # else:
        #     b, t, v, d = zs.shape
        #     q_zs = zs.repeat(len(zs), 1, 1, 1)
        #     q_zt = zs.unsqueeze(dim=1).repeat(1, len(zs), 1, 1, 1).reshape(-1, t, v, d)
        # f_s_t = self.critic_st(q_zs, q_zt)

        mubo = f_st - f_s_t
        pos_mask = torch.zeros_like(f_st)
        pos_mask[mubo < 0] = 1
        mubo_musk = mubo * pos_mask
        reg = (mubo_musk ** 2).mean()

        return mubo.mean() + reg


class LaSTBlock(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len, pred_len, s_func, inner_s, t_func, inner_t, dropout=0.1):
        super().__init__()
        self.input_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.SNet = s_func(in_dim, out_dim, seq_len, pred_len, inner_s, dropout=dropout)
        self.TNet = t_func(in_dim, out_dim, seq_len, pred_len, inner_t, dropout=dropout)
        self.MuboNet = CalculateMubo(inner_s, inner_t, dropout=dropout)

    def forward(self, x_his):
        x_s, xs_rec, elbo_s, mlbo_s = self.SNet(x_his)
        x_t, xt_rec, elbo_t, mlbo_t = self.TNet(x_his)

        rec_err = ((xs_rec + xt_rec - x_his) ** 2).mean()
        elbo = elbo_t + elbo_s - rec_err
        mlbo = mlbo_t + mlbo_s
        mubo = self.MuboNet(x_his, self.SNet.VarUnit_s, self.TNet.VarUnit_t)

        return x_s, x_t, elbo, mlbo, mubo


class SNet(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len, pred_len, inner_s, dropout=0.1, device="cuda:0"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.inner_s = inner_s
        """ VAE Net """
        self.VarUnit_s = VarUnit(in_dim, inner_s)
        self.rec_s = nn.Linear(inner_s, in_dim)
        self.RecUnit_s = FeedNet(inner_s, in_dim, type="mlp", n_layers=1, dropout=dropout)
        """ Fourier """
        self.FourierNet = NeuralFourierLayer(inner_s, out_dim, seq_len, pred_len)
        self.pred = FeedNet(self.inner_s, self.out_dim, type="mlp", n_layers=1)

    def forward(self, x_his):
        qz_s, mean_qz_s, var_qz_s = self.VarUnit_s(x_his)
        xs_rec = self.RecUnit_s(qz_s)
        elbo_s = period_sim(xs_rec, x_his) - self.VarUnit_s.compute_KL(
            qz_s, mean_qz_s, var_qz_s)
        mlbo_s = self.VarUnit_s.compute_MLBO(x_his, qz_s)

        """ Fourier """
        xs_pred = self.pred(self.FourierNet(qz_s)[:, -self.pred_len:])

        return xs_pred, xs_rec, elbo_s, mlbo_s


class TNet(nn.Module):
    def __init__(self, in_dim, out_dim, seq_len, pred_len, inner_t, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.inner_t = inner_t

        self.VarUnit_t = VarUnit(in_dim, inner_t)
        self.RecUnit_t = FeedNet(inner_t, in_dim, type="mlp", n_layers=1)

        self.t_pred_1 = FeedNet(self.seq_len, self.pred_len, type="mlp", n_layers=1)
        self.t_pred_2 = FeedNet(self.inner_t, self.out_dim, type="mlp", n_layers=1)

    def forward(self, x_his):
        qz_t, mean_qz_t, var_qz_t = self.VarUnit_t(x_his)
        xt_rec = self.RecUnit_t(qz_t)
        elbo_t = trend_sim(xt_rec, x_his) - self.VarUnit_t.compute_KL(qz_t, mean_qz_t, var_qz_t)
        mlbo_t = self.VarUnit_t.compute_MLBO(x_his, qz_t)

        # mlp
        if len(x_his.shape) == 3:
            xt_pred = self.t_pred_2(self.t_pred_1(qz_t.permute(0, 2, 1)).permute(0, 2, 1))
        else:
            xt_pred = self.t_pred_2(self.t_pred_1(qz_t.permute(0, 3, 2, 1)).permute(0, 3, 2, 1))

        return xt_pred, xt_rec, elbo_t, mlbo_t


class LaST(nn.Module):
    def __init__(self, input_len, output_len, input_dim, out_dim, var_num=1, latent_dim=64, dropout=0.1,
                 device="cuda:0"):
        super(LaST, self).__init__()
        print("------------LaST Net---------------")
        self.in_dim = input_dim
        self.out_dim = out_dim
        self.seq_len = input_len
        self.pred_len = output_len

        self.v_num = var_num
        self.inner_s = latent_dim
        self.inner_t = latent_dim
        self.dropout = dropout
        self.device = device

        self.LaSTLayer = LaSTBlock(self.in_dim, self.out_dim, input_len, output_len, SNet, self.inner_s, TNet,
                                   self.inner_t, dropout=dropout)

    def forward(self, x, x_mark=None):
        b, t, _ = x.shape
        x_his = x
        if self.v_num > 1:
            x_his = x_his.reshape(b, t, self.v_num, -1)
            if x_mark is not None:
                x_his = torch.cat([x_his, x_mark.unsqueeze(dim=2).repeat(1, 1, self.v_num, 1)], dim=-1)
        else:
            if x_mark is not None:
                x_his = torch.cat([x_his, x_mark], dim=-1)

        x_s, x_t, elbo, mlbo, mubo = self.LaSTLayer(x_his)
        x_pred = x_s + x_t
        x_pred = x_pred.squeeze(-1) if self.v_num > 1 else x_pred

        return x_pred, elbo, mlbo, mubo
