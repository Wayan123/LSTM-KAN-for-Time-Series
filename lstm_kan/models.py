from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
            * h
            + grid_range[0]
        ).expand(in_features, -1)
        self.register_buffer("grid", grid.contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        else:
            self.register_parameter("spline_scaler", None)

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                torch.rand(
                    self.grid_size + 1, self.in_features, self.out_features, device=self.grid.device
                )
                - 0.5
            ) * self.scale_noise / self.grid_size
            coeff = self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
            self.spline_weight.data.copy_(
                (1.0 if self.enable_standalone_scale_spline else self.scale_spline) * coeff
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError("KANLinear expects input shape [batch, in_features].")

        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1 : (-k)])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]

        expected = (x.size(0), self.in_features, self.grid_size + self.spline_order)
        if bases.size() != expected:
            raise RuntimeError(f"Unexpected spline basis shape {bases.size()}, expected {expected}")
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError("curve2coeff expects input shape [batch, in_features].")
        expected = (x.size(0), self.in_features, self.out_features)
        if y.size() != expected:
            raise ValueError(f"curve2coeff expects target shape {expected}, got {tuple(y.size())}")
        a = self.b_splines(x).transpose(0, 1)
        b = y.transpose(0, 1)
        solution = torch.linalg.lstsq(a, b).solution
        result = solution.permute(2, 0, 1)
        expected_result = (self.out_features, self.in_features, self.grid_size + self.spline_order)
        if result.size() != expected_result:
            raise RuntimeError(
                f"Unexpected coefficient shape {result.size()}, expected {expected_result}"
            )
        return result.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError("KANLinear expects input shape [batch, in_features].")
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError("KANLinear expects input shape [batch, in_features].")
        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation.clamp_min(1e-12)
        regularization_loss_entropy = -torch.sum(p * p.clamp_min(1e-12).log())
        return regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy


class GRUNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, drop_prob: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def init_hidden(self, batch_size: int, device: torch.device | None = None):
        device = device or next(self.parameters()).device
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        if h is None:
            h = self.init_hidden(x.size(0), x.device)
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1, :]))
        return out, h


class GRUNetKAN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, drop_prob: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = KANLinear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def init_hidden(self, batch_size: int, device: torch.device | None = None):
        device = device or next(self.parameters()).device
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        if h is None:
            h = self.init_hidden(x.size(0), x.device)
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1, :]))
        return out, h


class LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, drop_prob: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def init_hidden(self, batch_size: int, device: torch.device | None = None):
        device = device or next(self.parameters()).device
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return hidden, cell

    def forward(self, x: torch.Tensor, h=None):
        if h is None:
            h = self.init_hidden(x.size(0), x.device)
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1, :]))
        return out, h


class LSTMNetKAN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, drop_prob: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = KANLinear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def init_hidden(self, batch_size: int, device: torch.device | None = None):
        device = device or next(self.parameters()).device
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return hidden, cell

    def forward(self, x: torch.Tensor, h=None):
        if h is None:
            h = self.init_hidden(x.size(0), x.device)
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1, :]))
        return out, h


def build_model(
    model_name: Literal["GRU", "GRUKAN", "LSTM", "LSTMKAN"],
    input_dim: int,
    hidden_dim: int,
    output_dim: int = 1,
    n_layers: int = 2,
    dropout: float = 0.2,
) -> nn.Module:
    if model_name == "GRU":
        return GRUNet(input_dim, hidden_dim, output_dim, n_layers, dropout)
    if model_name == "GRUKAN":
        return GRUNetKAN(input_dim, hidden_dim, output_dim, n_layers, dropout)
    if model_name == "LSTM":
        return LSTMNet(input_dim, hidden_dim, output_dim, n_layers, dropout)
    if model_name == "LSTMKAN":
        return LSTMNetKAN(input_dim, hidden_dim, output_dim, n_layers, dropout)
    raise ValueError(f"Unknown model name: {model_name}")
