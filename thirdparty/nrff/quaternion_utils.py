import torch
import numpy as np



def quaternion_product(p, q):
    p_r = p[..., [0]]
    p_i = p[..., 1:]
    q_r = q[..., [0]]
    q_i = q[..., 1:]

    out_r = p_r * q_r - (p_i * q_i).sum(dim=-1)
    out_i = p_r * q_i + q_r * p_i + torch.linalg.cross(p_i, q_i, dim=-1)

    return torch.cat([out_r, out_i], dim=-1)

def quaternion_inverse(p):
    p_r = p[..., [0]]
    p_i = -p[..., 1:]

    return torch.cat([p_r, p_i], dim=-1)

def quaternion_rotate(p, q):
    q_inv = quaternion_inverse(q)

    qp = quaternion_product(q, p)
    out = quaternion_product(qp, q_inv)
    return out

def build_q(vec, angle):
    out_r = torch.cos(angle / 2)
    out_i = torch.sin(angle / 2) * vec

    return torch.cat([out_r, out_i], dim=-1)


def cartesian2quaternion(x):
    zeros_ = x.new_zeros([*x.shape[:-1], 1])
    return torch.cat([zeros_, x], dim=-1)


def spherical2cartesian(theta, phi):
    x = torch.cos(phi) * torch.sin(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(theta)

    return [x, y, z]

def init_predefined_omega(n_theta, n_phi):
    theta_list = torch.linspace(0, np.pi, n_theta)
    phi_list = torch.linspace(0, np.pi*2, n_phi)

    out_omega = []
    out_omega_lambda = []
    out_omega_mu = []

    for i in range(n_theta):
        theta = theta_list[i].view(1, 1)

        for j in range(n_phi):
            phi = phi_list[j].view(1, 1)

            omega = spherical2cartesian(theta, phi)
            omega = torch.stack(omega, dim=-1).view(1, 3)

            omega_lambda = spherical2cartesian(theta+np.pi/2, phi)
            omega_lambda = torch.stack(omega_lambda, dim=-1).view(1, 3)

            p = cartesian2quaternion(omega_lambda)
            q = build_q(omega, torch.tensor(np.pi/2).view(1, 1))
            omega_mu = quaternion_rotate(p, q)[..., 1:]

            out_omega.append(omega)
            out_omega_lambda.append(omega_lambda)
            out_omega_mu.append(omega_mu)


    out_omega = torch.stack(out_omega, dim=0)
    out_omega_lambda = torch.stack(out_omega_lambda, dim=0)
    out_omega_mu = torch.stack(out_omega_mu, dim=0)

    return out_omega, out_omega_lambda, out_omega_mu
