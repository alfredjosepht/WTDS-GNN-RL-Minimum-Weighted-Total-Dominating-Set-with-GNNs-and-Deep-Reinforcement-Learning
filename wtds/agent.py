"""Double DQN agent (Section 7).

Target: y = R + γⁿ · Q_target(s', argmax_{a' valid} Q_online(s', a')),  y = R if done.
Invalid actions are -inf in both the behaviour argmax and the target argmax.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as Fn

from .model import QNet, collate, graph_argmax


class DDQNAgent:
    def __init__(self, model_cfg: dict, lr: float = 1e-4, gamma: float = 1.0, tau: float = 0.005,
                 target_update: str = "soft", hard_every: int = 1000, loss: str = "huber",
                 grad_clip: float = 10.0, device: str = "cpu"):
        self.device = torch.device(device)
        self.online = QNet(**model_cfg).to(self.device)
        self.target = copy.deepcopy(self.online).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        # fused Adam on CUDA: same update rule, far fewer kernel launches
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr, fused=self.device.type == "cuda")
        self._params = list(self.online.parameters())
        self._tparams = list(self.target.parameters())
        self.gamma, self.tau = gamma, tau
        self.target_update, self.hard_every = target_update, hard_every
        self.loss_name, self.grad_clip = loss, grad_clip
        self.updates = 0

    def update(self, cur, nxt, actions, rewards, dones, gammas) -> dict:
        """cur/nxt: lists of (graph, X, glob) as returned by ReplayBuffer.sample."""
        dev = self.device
        return self.update_batches(collate(cur, dev), collate(nxt, dev), actions, rewards, dones, gammas)

    def update_batches(self, b, b2, actions, rewards, dones, gammas, fused: bool = True) -> dict:
        """One DDQN step on prebuilt Batches (CPU collate or wtds.gpu_batch).

        fused=True runs the online network once on the union of s_t and s_{t+n}
        (identical outputs: every layer is per-vertex or per-graph), saving a pass."""
        from .gpu_batch import cat_batches
        dev = self.device
        self.online.train()
        if fused:
            q_both = self.online(cat_batches(b, b2))
            n1 = b.x.shape[0]
            q_all, q_next_online = q_both[:n1], q_both[n1:].detach()
        else:
            q_all = self.online(b)
            with torch.no_grad():
                q_next_online = self.online(b2)
        a_idx = b.ptr[:-1] + torch.as_tensor(actions, device=dev)
        q_sa = q_all[a_idx]
        with torch.no_grad():
            _, a_star = graph_argmax(q_next_online, b2.batch, b2.num_graphs)
            q_next_target = self.target(b2, mask=False)[a_star.clamp_max(q_next_online.numel() - 1)]
            done = torch.as_tensor(dones, device=dev)
            boot = torch.where(done, torch.zeros_like(q_next_target), q_next_target)
            y = torch.as_tensor(rewards, device=dev) + torch.as_tensor(gammas, device=dev) * boot
        if self.loss_name == "huber":
            loss = Fn.smooth_l1_loss(q_sa, y)
        else:
            loss = 0.5 * Fn.mse_loss(q_sa, y)                     # paper Eq. 8
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.opt.step()
        self.updates += 1
        self._sync_target()
        return {"loss": float(loss.detach()), "q_mean": float(q_sa.detach().mean()), "y_mean": float(y.mean()),
                "grad_norm": float(gn)}

    @torch.no_grad()
    def _sync_target(self):
        if self.target_update == "soft":       # Θ̄ <- τΘ + (1-τ)Θ̄, as two multi-tensor ops
            torch._foreach_mul_(self._tparams, 1 - self.tau)
            torch._foreach_add_(self._tparams, self._params, alpha=self.tau)
        elif self.updates % self.hard_every == 0:
            self.target.load_state_dict(self.online.state_dict())

    def state_dict(self):
        return {"online": self.online.state_dict(), "target": self.target.state_dict(),
                "opt": self.opt.state_dict(), "updates": self.updates,
                "model_cfg": self.online.config}


def load_model(path, device="cpu") -> QNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = QNet(**ck["model_cfg"]).to(device)
    m.load_state_dict(ck["online"])
    m.eval()
    m.env_kwargs = ck.get("env_kwargs", {})
    return m
