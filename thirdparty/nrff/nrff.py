from ..tensorf.models.tensorBase import *
from .quaternion_utils import *
import numpy as np


class RenderingEquationEncoding(torch.nn.Module):
    def __init__(self, num_theta, num_phi, device):
        super(RenderingEquationEncoding, self).__init__()

        self.num_theta = num_theta
        self.num_phi = num_phi

        omega, omega_la, omega_mu = init_predefined_omega(num_theta, num_phi)
        self.omega = omega.view(1, num_theta, num_phi, 3).to(device)
        self.omega_la = omega_la.view(1, num_theta, num_phi, 3).to(device)
        self.omega_mu = omega_mu.view(1, num_theta, num_phi, 3).to(device)

    def forward(self, omega_o, a, la, mu):
        Smooth = F.relu((omega_o[:, None, None] * self.omega).sum(dim=-1, keepdim=True)) # N, num_theta, num_phi, 1

        la = F.softplus(la - 1)
        mu = F.softplus(mu - 1)
        exp_input = -la * (self.omega_la * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2) -mu * (self.omega_mu * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2)
        out = a * Smooth * torch.exp(exp_input)

        return out

class RenderingNet(torch.nn.Module):
    def __init__(self, num_theta = 8, num_phi=16, data_dim_color=192, featureC=256, device='cpu', 
                 view_pe=-1, fea_pe=-1, btn_freq=[]):
        super(RenderingNet, self).__init__()

        self.ch_cd = 3
        self.ch_s = 3
        self.ch_normal = 3
        self.ch_bottleneck = 128

        self.num_theta = 8
        self.num_phi = 16
        self.num_asg = self.num_theta * self.num_phi

        self.ch_asg_feature = 128
        self.ch_per_theta = self.ch_asg_feature // self.num_theta

        self.ch_a = 2
        self.ch_la = 1
        self.ch_mu = 1
        self.ch_per_asg = self.ch_a + self.ch_la + self.ch_mu

        self.ch_normal_dot_viewdir = 1

        self.view_pe = view_pe
        self.fea_pe = fea_pe

        if len(btn_freq) >= 2:
            self.btn_freq = torch.linspace(np.log2(btn_freq[0]), np.log2(btn_freq[1]), 
                                            self.ch_bottleneck, device=0)
            self.btn_freq = torch.exp2(self.btn_freq)
        else:
            self.btn_freq = None

        self.ree_function = RenderingEquationEncoding(num_theta, num_phi, device)

        self.spatial_mlp = torch.nn.Sequential(
                torch.nn.Linear(data_dim_color, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, self.ch_cd + self.ch_s + self.ch_bottleneck + self.ch_normal + self.ch_asg_feature)).to(device)

        self.asg_mlp = torch.nn.Sequential(torch.nn.Linear(self.ch_per_theta, self.num_phi * self.ch_per_asg)).to(device)

        in_dim = self.ch_bottleneck + self.num_asg * self.ch_a + self.ch_normal_dot_viewdir
        if self.view_pe > -1:
           in_dim += 3
        if self.view_pe > 0:
           in_dim += 3 * self.view_pe * 2
        self.directional_mlp = torch.nn.Sequential(
                torch.nn.Linear(in_dim, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, featureC),
                torch.nn.GELU(),
                torch.nn.Linear(featureC, 3)).to(device)


    def spatial_mlp_forward(self, x):
        out = self.spatial_mlp(x)
        sections = [self.ch_cd, self.ch_s, self.ch_normal, self.ch_bottleneck, self.ch_asg_feature]
        diffuse_color, tint, normals, bottleneck, asg_features = torch.split(out, sections, dim=-1)
        normals = -F.normalize(normals, dim=1)
        return diffuse_color, tint, normals, bottleneck, asg_features

    def asg_mlp_forward(self, asg_feature):
        N = asg_feature.size(0)
        asg_feature = asg_feature.view(N, self.num_theta, -1)
        asg_params = self.asg_mlp(asg_feature)
        asg_params = asg_params.view(N, self.num_theta, self.num_phi, -1)

        a, la, mu = torch.split(asg_params, [self.ch_a, self.ch_la, self.ch_mu], dim=-1)
        return a, la, mu

    def directional_mlp_forward(self, x):
        out = self.directional_mlp(x)
        return out

    def reflect(self, viewdir, normal):
        out = 2 * (viewdir * normal).sum(dim=-1, keepdim=True) * normal - viewdir
        return out

    def forward(self, pts, viewdir, feature):
        diffuse_color, tint, normal, bottleneck, asg_feature = self.spatial_mlp_forward(feature)
        refdir = self.reflect(-viewdir, normal)

        a, la, mu = self.asg_mlp_forward(asg_feature)
        ree = self.ree_function(refdir, a, la, mu) # N, num_theta, num_phi, ch_per_asg
        ree = ree.view(ree.size(0), -1)

        if self.btn_freq is not None:
            bottleneck = bottleneck + torch.sin(bottleneck * self.btn_freq[None])

        normal_dot_viewdir = ((-viewdir) * normal).sum(dim=-1, keepdim=True)
        dir_mlp_input = [bottleneck, ree, normal_dot_viewdir]
        if self.view_pe > -1:
            dir_mlp_input += [viewdir]
        if self.view_pe > 0:
            dir_mlp_input += [positional_encoding(viewdir, self.view_pe)]
        dir_mlp_input = torch.cat(dir_mlp_input, dim=-1)
        specular_color = self.directional_mlp_forward(dir_mlp_input)

        raw_rgb = diffuse_color + tint * specular_color
        rgb = torch.sigmoid(raw_rgb)

        return rgb, normal
