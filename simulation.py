import torch
from torch import Tensor
from torch.distributions.normal import Normal

from policy import Policy
from device import device

# ??
j = 10.5

# air control torque coefficients
t = torch.tensor([-400.0, -130.0, 95.0], dtype=torch.float, device=device)
m = torch.diag(torch.ones(3)).byte().to(device)
identity = torch.diag(torch.ones(3))[None, :, :].to(device)


w_max = 5.5
batch_size = 60000
meps = 1 - 1e-5


class Simulation:
    o: Tensor = None
    w: Tensor = None

    def __init__(self, policy: Policy):
        self.policy = policy

    def random_state(self):
        x_axis = Normal(0, 1).sample((batch_size, 3)).to(device)
        y_axis = Normal(0, 1).sample((batch_size, 3)).to(device)
        z_axis = torch.cross(x_axis, y_axis, dim=1)
        y_axis = torch.cross(z_axis, x_axis, dim=1)
        self.o = torch.stack((x_axis, y_axis, z_axis), dim=1)
        self.o = self.o / torch.norm(self.o, dim=2, keepdim=True)

        self.w = Normal(0, 1).sample((batch_size, 3)).to(device)
        self.w = self.w / torch.norm(self.w, dim=1, keepdim=True)
        self.w = self.w * torch.rand((batch_size, 1), device=device) * w_max

        willDodge = torch.randint(2, (batch_size,), device=device).float() # 50% chance to not contain a dodge at all
        self.noPitchTime = willDodge * torch.randint(1, int(round(.95*120))+1, (batch_size,)).to(device).float() / 120
        self.dodgeTime = willDodge * (self.noPitchTime - .3).clamp_max(0)
        self.dodgeDirection = Normal(0, 1).sample((batch_size, 2)).to(device)   # roll pitch
        self.dodgeDirection /= torch.norm(self.dodgeDirection, dim=1, keepdim=True)
        self.dodgeDirection *= self.dodgeTime.clamp_min(1)[:, None]

    def simulate(self, steps: int, dt: float):
        for _ in range(steps):
            self.step(dt)

    def w_local(self):
        return torch.sum(self.o * self.w[:, :, None], 1)

    def step(self, dt):
        w_local = self.w_local()

        rpy = self.policy(self.o.permute(0, 2, 1), w_local, self.noPitchTime, self.dodgeTime, self.dodgeDirection)

        # air damping torque coefficients
        h = torch.stack((
            torch.full_like(rpy[:, 0], -50.0),
            -30.0 * (1.0 - rpy[:, 1].abs()),
            -20.0 * (1.0 - rpy[:, 2].abs())
        ), dim=1)

        angularAcc = t[None, :] * rpy
        angularAcc[:, 1] *= self.noPitchTime < 0.01/120

        dodge = self.dodgeTime > 0.01/120
        cancel = 1 - (rpy[:, 1] * -self.dodgeDirection[:, 1].sign()).clamp_min(0)
        angularAcc[:, 0] += self.dodgeDirection[:, 0] * dodge
        angularAcc[:, 1] += self.dodgeDirection[:, 1] * dodge * cancel

        self.w = self.w + torch.sum(self.o * (angularAcc + h * w_local)[:, None, :], 2) * (dt / j)
        self.o = torch.sum(self.o[:, None, :, :] * axis_to_rotation(self.w * dt)[:, :, :, None], 2)

        self.w = self.w / torch.clamp_min(torch.norm(self.w, dim=1) / w_max, 1)[:, None]

        self.noPitchTime -= dt
        self.noPitchTime.clamp_max_(0)
        self.dodgeTime -= dt
        self.dodgeTime.clamp_max_(0)
        self.dodgeDirection *= self.dodgeTime.clamp_min(1)[:, None]

    def error(self):
        return torch.acos(meps * 0.5 * (torch.sum(torch.sum(self.o[:, :, None, :] *
                                                            identity[:, None, :, :], 3)[:, m], 1) - 1.0))


def axis_to_rotation(omega: Tensor):
    norm_omega = torch.norm(omega, dim=1)

    u = omega / norm_omega[:, None]

    c = torch.cos(norm_omega)
    s = torch.sin(norm_omega)

    result = u[:, :, None] * u[:, None, :] * (-c[:, None, None] + 1.0)
    result += c[:, None, None] * torch.diag(torch.ones(3, device=device))[None, :, :]

    result += torch.cross(s[:, None, None] * torch.diag(torch.ones(3, device=device))[None, :, :],
                          u[:, None, :].repeat(1, 3, 1), dim=2)

    return result
