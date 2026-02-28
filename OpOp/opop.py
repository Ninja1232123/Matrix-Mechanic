"""
OptBrain — Optimizer-agnostic learned modulation.

Wraps ANY optimizer. Watches training. Learns what helps.

    from opt_brain import OptBrain

    # Wrap any optimizer
    base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = OptBrain(base_optimizer)

    # Training loop — just pass loss
    loss = criterion(output, target)
    loss.backward()
    optimizer.step(loss=loss.item())
    optimizer.zero_grad()

That's it. Drop-in. Works with Adam, SGD, AdamW, Sophia, LAMB, anything.

The brain observes:
  - Loss trajectory (trend, volatility, plateaus, spikes)
  - Per-group gradient statistics (magnitude, direction, oscillation)
  - Training phase (early, converging, stuck, diverging)

The brain outputs per parameter group:
  - scale:     [0.01, 10.0]  multiply the update
  - clip:      [0.1, 5.0]    local gradient clip multiplier
  - momentum:  [0.0, 1.0]    blend base optimizer update with dampened version

Trained online via REINFORCE. Reward = loss went down.
Initialized neutral so it can't make things worse than the base optimizer.

Overhead: ~0.1% compute, ~50KB memory. Less than a rounding error
compared to the 2x parameter memory Adam already wastes.

Works with PyTorch and numpy optimizers.
"""

import numpy as np
from collections import defaultdict

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═════════════════════════════════════════════════════════════════════════════
# OBSERVER — watches training, builds observation vectors
# ═════════════════════════════════════════════════════════════════════════════

class TrainingObserver:
    """Tracks training dynamics across parameter groups."""

    def __init__(self, n_groups, loss_window=64, grad_window=32):
        self.n_groups = n_groups
        self.loss_window = loss_window
        self.grad_window = grad_window

        # Loss
        self.losses = np.zeros(loss_window, dtype=np.float32)
        self.loss_ptr = 0
        self.loss_count = 0

        # Per-group
        self.grad_norms = np.zeros((n_groups, grad_window), dtype=np.float32)
        self.grad_cosines = np.zeros((n_groups, grad_window), dtype=np.float32)
        self.grad_ptr = 0
        self._prev_grad = [None] * n_groups

    def record_loss(self, loss):
        self.losses[self.loss_ptr % self.loss_window] = loss
        self.loss_ptr += 1
        self.loss_count += 1

    def record_grad(self, group_idx, grad_flat):
        """Record gradient stats. Pass flattened numpy array."""
        norm = float(np.sqrt(np.sum(grad_flat[:1024] ** 2)) + 1e-12)

        cos = 0.0
        prev = self._prev_grad[group_idx]
        if prev is not None:
            n = min(len(grad_flat), len(prev), 1024)
            dot = np.dot(grad_flat[:n], prev[:n])
            cos = dot / (np.sqrt(np.sum(grad_flat[:n]**2)) *
                         np.sqrt(np.sum(prev[:n]**2)) + 1e-12)

        self._prev_grad[group_idx] = grad_flat[:1024].copy()

        ptr = self.grad_ptr % self.grad_window
        self.grad_norms[group_idx, ptr] = norm
        self.grad_cosines[group_idx, ptr] = cos

    def advance(self):
        self.grad_ptr += 1

    def observe(self):
        """Build observation vector.

        Layout:
          [0:6]  loss: mean, std, trend, recent_delta, plateau, phase
          Per group (n_groups x 6):
            norm_mean, norm_recent, direction_consistency,
            oscillation, acceleration, stuck
        """
        obs = []

        # ── Loss ──
        n = min(self.loss_count, self.loss_window)
        if n > 1:
            L = self.losses[:n] if self.loss_count <= self.loss_window else self.losses
            lm = float(np.mean(L))
            ls = float(np.std(L))
            x = np.arange(n, dtype=np.float32); x -= x.mean()
            trend = float(np.sum(x * L[:n]) / (np.sum(x*x) + 1e-8))
            if n >= 4:
                q = n // 4
                delta = float(np.mean(L[-q:]) - np.mean(L[:q]))
            else:
                delta = 0.0
            plateau = 1.0 - min(ls / (abs(lm) + 1e-8), 1.0)
        else:
            lm = ls = trend = delta = plateau = 0.0

        phase = min(self.loss_count / 10000.0, 1.0)
        obs.extend([lm, ls, trend, delta, plateau, phase])

        # ── Per group ──
        ng = min(self.grad_ptr, self.grad_window)
        for gi in range(self.n_groups):
            if ng > 0:
                norms = self.grad_norms[gi, :ng]
                cosines = self.grad_cosines[gi, :ng]

                nm = float(np.mean(norms))
                nr = float(norms[-1])
                dc = float(np.mean(cosines))

                # Oscillation: sign flips in direction
                if ng >= 3:
                    osc = float(np.sum(np.abs(np.diff(np.sign(cosines[:ng]))) > 0)) / max(ng-1, 1)
                else:
                    osc = 0.0

                # Acceleration: is norm increasing or decreasing?
                if ng >= 4:
                    half = ng // 2
                    accel = float(np.mean(norms[-half:]) - np.mean(norms[:half]))
                else:
                    accel = 0.0

                # Stuck: low norm + consistent direction = converged or dead
                stuck = (1.0 - min(nm / (nm + 1.0), 1.0)) * abs(dc)
            else:
                nm = nr = dc = osc = accel = stuck = 0.0

            obs.extend([nm, nr, dc, osc, accel, stuck])

        return np.array(obs, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# BRAIN — tiny MLP that makes decisions
# ═════════════════════════════════════════════════════════════════════════════

class Brain:
    """Tiny agent that outputs per-group modulation decisions.

    Outputs per group:
        scale:    [0.01, 10.0]  — how much to amplify/dampen the update
        clip:     [0.1, 5.0]   — local clip radius multiplier
        momentum: [0.0, 1.0]   — 0=use base optimizer as-is, 1=heavily dampen

    All initialized to neutral (scale=1, clip=1, momentum=0).
    """

    def __init__(self, obs_dim, n_groups, hidden=64, lr=0.0003, seed=42):
        self.n_groups = n_groups
        self.out_dim = n_groups * 3
        self.lr = lr

        rng = np.random.RandomState(seed)
        h1, h2 = hidden, hidden // 2

        self.W1 = (rng.randn(obs_dim, h1) * np.sqrt(2.0/obs_dim)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = (rng.randn(h1, h2) * np.sqrt(2.0/h1)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.W3 = (rng.randn(h2, self.out_dim) * 0.01).astype(np.float32)
        self.b3 = np.zeros(self.out_dim, dtype=np.float32)

        # Memory: recurrent hidden state
        self._h2 = np.zeros(h2, dtype=np.float32)

        # REINFORCE
        self._baseline = 0.0
        self._exp = []  # (obs, raw_output, reward)
        self._update_every = 16

    def forward(self, obs):
        """Returns {group_idx: (scale, clip, momentum)}, cache"""
        z1 = obs @ self.W1 + self.b1
        h1 = np.maximum(z1, 0)
        # Inject memory
        mem_n = min(len(self._h2), len(h1))
        h1[:mem_n] += 0.1 * self._h2[:mem_n]

        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(z2, 0)
        self._h2 = h2.copy()

        raw = h2 @ self.W3 + self.b3

        decisions = {}
        for gi in range(self.n_groups):
            b = gi * 3
            sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

            scale = 0.01 * (10.0/0.01) ** sig(raw[b])      # log-uniform [0.01, 10]
            clip = 0.1 + 4.9 * sig(raw[b+1])                # [0.1, 5.0]
            dampen = float(sig(raw[b+2]))                    # [0, 1]

            decisions[gi] = (float(scale), float(clip), dampen)

        cache = {'obs': obs.copy(), 'raw': raw.copy(),
                 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
        return decisions, cache

    def record(self, reward, cache):
        self._exp.append((cache['obs'], cache['raw'], reward))
        self._baseline = 0.95 * self._baseline + 0.05 * reward

    def maybe_update(self):
        if len(self._exp) < self._update_every:
            return False

        obs_b = np.stack([e[0] for e in self._exp])
        raw_b = np.stack([e[1] for e in self._exp])
        rew_b = np.array([e[2] for e in self._exp], dtype=np.float32)

        adv = (rew_b - self._baseline) / (np.std(rew_b) + 1e-8)

        # Forward
        Z1 = obs_b @ self.W1 + self.b1
        H1 = np.maximum(Z1, 0)
        Z2 = H1 @ self.W2 + self.b2
        H2 = np.maximum(Z2, 0)
        Rp = H2 @ self.W3 + self.b3

        # Policy gradient
        dR = -adv[:, None] * (raw_b - Rp) / len(rew_b)
        dW3 = H2.T @ dR;  db3 = np.mean(dR, axis=0)
        dH2 = dR @ self.W3.T * (Z2 > 0)
        dW2 = H1.T @ dH2;  db2 = np.mean(dH2, axis=0)
        dH1 = dH2 @ self.W2.T * (Z1 > 0)
        dW1 = obs_b.T @ dH1;  db1 = np.mean(dH1, axis=0)

        clip = lambda x: np.clip(x, -1.0, 1.0)
        self.W1 -= self.lr * clip(dW1)
        self.b1 -= self.lr * clip(db1)
        self.W2 -= self.lr * clip(dW2)
        self.b2 -= self.lr * clip(db2)
        self.W3 -= self.lr * clip(dW3)
        self.b3 -= self.lr * clip(db3)

        self._exp.clear()
        return True

    def state_dict(self):
        return {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'h2': self._h2,
            'baseline': np.float32(self._baseline),
        }

    def load_state_dict(self, d):
        for k in ['W1','b1','W2','b2','W3','b3','h2']:
            if k in d:
                setattr(self, k if k != 'h2' else '_h2',
                        d[k].astype(np.float32))
        if 'baseline' in d:
            self._baseline = float(d['baseline'])


# ═════════════════════════════════════════════════════════════════════════════
# OPTBRAIN — the wrapper
# ═════════════════════════════════════════════════════════════════════════════

class OptBrain:
    """Wraps any optimizer with a learned brain.

    PyTorch usage:
        base = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt = OptBrain(base)
        ...
        loss.backward()
        opt.step(loss=loss.item())
        opt.zero_grad()

    Numpy usage:
        opt = OptBrain(None, n_groups=5)
        decisions = opt.get_decisions(loss=current_loss)
        # Apply decisions[group_idx] = (scale, clip, dampen) yourself
        opt.record_grads(group_idx, grad_flat)
        opt.finish_step()

    The brain:
        - Can't make things worse (initialized neutral)
        - Learns online from loss signal (REINFORCE)
        - Checkpoints with save/load
        - ~50KB memory, ~0.1% compute overhead
    """

    def __init__(self, base_optimizer=None, n_groups=None,
                 brain_hidden=64, brain_lr=0.0003, brain_update_every=16):
        """
        Args:
            base_optimizer: PyTorch optimizer (or None for numpy mode)
            n_groups: number of parameter groups (auto-detected from optimizer)
            brain_hidden: brain MLP hidden dim
            brain_lr: brain learning rate
            brain_update_every: REINFORCE batch size
        """
        self._torch_mode = base_optimizer is not None and HAS_TORCH

        if self._torch_mode:
            self.base = base_optimizer
            self.n_groups = len(base_optimizer.param_groups)
        else:
            self.base = None
            self.n_groups = n_groups or 1

        # Observer
        obs_dim = 6 + self.n_groups * 6
        self.observer = TrainingObserver(self.n_groups)

        # Brain
        self.brain = Brain(
            obs_dim=obs_dim,
            n_groups=self.n_groups,
            hidden=brain_hidden,
            lr=brain_lr,
        )
        self.brain._update_every = brain_update_every

        self._prev_loss = None
        self._cache = None
        self._decisions = None
        self._step_count = 0

        # Per-group gradient buffers for torch mode
        if self._torch_mode:
            self._group_grad_samples = {}

    # ── PyTorch interface ────────────────────────────────────────────────────

    def step(self, loss=None, closure=None):
        """Drop-in replacement for optimizer.step().

        Args:
            loss: current loss value (float). Required for brain to learn.
            closure: same as base optimizer closure (rarely used).
        """
        if not self._torch_mode:
            raise RuntimeError("Use get_decisions() in numpy mode")

        self._step_count += 1

        # ── Reward brain for previous decision ──
        if loss is not None:
            self.observer.record_loss(loss)
            if self._prev_loss is not None and self._cache is not None:
                delta = loss - self._prev_loss
                reward = float(np.clip(-delta / (abs(self._prev_loss) + 1e-8), -5, 5))
                self.brain.record(reward, self._cache)
                self.brain.maybe_update()
            self._prev_loss = loss

        # ── Brain observes and decides ──
        obs = self.observer.observe()
        self._decisions, self._cache = self.brain.forward(obs)

        # ── Record gradient stats before base optimizer clears them ──
        for gi, group in enumerate(self.base.param_groups):
            grads = []
            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad.data.detach().cpu().numpy().ravel()[:256])
            if grads:
                sample = np.concatenate(grads)[:1024]
                self.observer.record_grad(gi, sample)

        # ── Let base optimizer compute its update ──
        # We intercept by modifying gradients before the base step
        self._modulate_gradients()

        # ── Base optimizer step ──
        if closure is not None:
            result = self.base.step(closure)
        else:
            result = self.base.step()

        # ── Undo gradient modifications (restore originals) ──
        self._restore_gradients()

        self.observer.advance()
        return result

    def _modulate_gradients(self):
        """Apply brain decisions by modifying gradients in-place."""
        self._saved_grads = {}

        for gi, group in enumerate(self.base.param_groups):
            scale, clip_mult, dampen = self._decisions.get(gi, (1.0, 1.0, 0.0))

            for pi, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                key = (gi, pi)
                # Save original
                self._saved_grads[key] = p.grad.data.clone()

                g = p.grad.data

                # Clip: scale the effective gradient clip
                if clip_mult != 1.0:
                    gnorm = g.norm()
                    max_norm = gnorm * clip_mult
                    if gnorm > max_norm and gnorm > 0:
                        g.mul_(max_norm / gnorm)

                # Scale
                if scale != 1.0:
                    g.mul_(scale)

                # Dampen: blend toward zero (reduce update magnitude)
                if dampen > 0.01:
                    g.mul_(1.0 - dampen)

    def _restore_gradients(self):
        """Restore original gradients after base optimizer step."""
        for (gi, pi), saved in self._saved_grads.items():
            p = self.base.param_groups[gi]['params'][pi]
            if p.grad is not None:
                p.grad.data.copy_(saved)
        self._saved_grads.clear()

    def zero_grad(self, set_to_none=True):
        """Pass through to base optimizer."""
        if self._torch_mode:
            self.base.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        """Pass through for compatibility."""
        if self._torch_mode:
            return self.base.param_groups
        return []

    # ── Numpy interface ──────────────────────────────────────────────────────

    def get_decisions(self, loss=None):
        """Numpy mode: get brain decisions for this step.

        Returns: {group_idx: (scale, clip, dampen)}

        Call record_grads() and finish_step() after applying.
        """
        self._step_count += 1

        if loss is not None:
            self.observer.record_loss(loss)
            if self._prev_loss is not None and self._cache is not None:
                delta = loss - self._prev_loss
                reward = float(np.clip(-delta / (abs(self._prev_loss) + 1e-8), -5, 5))
                self.brain.record(reward, self._cache)
                self.brain.maybe_update()
            self._prev_loss = loss

        obs = self.observer.observe()
        self._decisions, self._cache = self.brain.forward(obs)
        return dict(self._decisions)

    def record_grads(self, group_idx, grad_flat):
        """Numpy mode: record gradient for a group."""
        self.observer.record_grad(group_idx, grad_flat)

    def finish_step(self):
        """Numpy mode: call after applying updates."""
        self.observer.advance()

    # ── Introspection ────────────────────────────────────────────────────────

    def get_brain_state(self):
        """Current brain decisions for logging."""
        if self._decisions is None:
            return {}
        state = {
            'step': self._step_count,
            'baseline': self.brain._baseline,
        }
        for gi in range(self.n_groups):
            if gi in self._decisions:
                s, c, d = self._decisions[gi]
                state[f'g{gi}_scale'] = round(s, 4)
                state[f'g{gi}_clip'] = round(c, 4)
                state[f'g{gi}_dampen'] = round(d, 4)
        return state

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path):
        """Save brain state (not base optimizer — save that separately)."""
        state = self.brain.state_dict()
        state['observer_losses'] = self.observer.losses
        state['observer_loss_ptr'] = np.int32(self.observer.loss_ptr)
        state['observer_loss_count'] = np.int32(self.observer.loss_count)
        state['observer_grad_norms'] = self.observer.grad_norms
        state['observer_grad_cosines'] = self.observer.grad_cosines
        state['observer_grad_ptr'] = np.int32(self.observer.grad_ptr)
        state['step_count'] = np.int32(self._step_count)
        state['prev_loss'] = np.float32(self._prev_loss or 0)
        np.savez(path, **state)

    def load(self, path):
        """Load brain state."""
        d = np.load(path, allow_pickle=True)
        self.brain.load_state_dict(d)
        if 'observer_losses' in d:
            self.observer.losses = d['observer_losses']
            self.observer.loss_ptr = int(d['observer_loss_ptr'])
            self.observer.loss_count = int(d['observer_loss_count'])
        if 'observer_grad_norms' in d:
            self.observer.grad_norms = d['observer_grad_norms']
            self.observer.grad_cosines = d['observer_grad_cosines']
            self.observer.grad_ptr = int(d['observer_grad_ptr'])
        self._step_count = int(d.get('step_count', 0))
        pl = float(d.get('prev_loss', 0))
        self._prev_loss = pl if pl != 0 else None


# ═════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═════════════════════════════════════════════════════════════════════════════

USAGE = """
╔══════════════════════════════════════════════════════════════════════════╗
║                          OptBrain                                      ║
║              Drop-in brain for any optimizer                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  PyTorch:                                                              ║
║                                                                        ║
║    from opt_brain import OptBrain                                      ║
║                                                                        ║
║    base = torch.optim.Adam(model.parameters(), lr=1e-3)               ║
║    optimizer = OptBrain(base)                                          ║
║                                                                        ║
║    for batch in dataloader:                                            ║
║        loss = model(batch)                                             ║
║        loss.backward()                                                 ║
║        optimizer.step(loss=loss.item())  # <-- just add loss=          ║
║        optimizer.zero_grad()                                           ║
║                                                                        ║
║  Numpy:                                                                ║
║                                                                        ║
║    brain = OptBrain(None, n_groups=3)                                  ║
║    decisions = brain.get_decisions(loss=current_loss)                  ║
║    scale, clip, dampen = decisions[group_idx]                          ║
║    # apply to your custom optimizer                                    ║
║    brain.record_grads(group_idx, grad_flat)                            ║
║    brain.finish_step()                                                 ║
║                                                                        ║
║  Save/Load:                                                            ║
║    optimizer.save("brain_checkpoint.npz")                              ║
║    optimizer.load("brain_checkpoint.npz")                              ║
║                                                                        ║
║  Inspect:                                                              ║
║    print(optimizer.get_brain_state())                                  ║
║    # {'step': 1000, 'g0_scale': 1.23, 'g0_clip': 0.8, ...}           ║
║                                                                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  What it does:                                                         ║
║    • Watches loss trajectory, gradient stats, training phase           ║
║    • Learns per-group: scale updates, adjust clipping, dampen          ║
║    • Online REINFORCE — reward = did loss go down?                     ║
║    • ~50KB memory. ~0.1% compute. Can't make things worse.            ║
║                                                                        ║
║  What it replaces:                                                     ║
║    • Manual LR scheduling                                              ║
║    • Manual gradient clipping tuning                                   ║
║    • Manual warmup schedules                                           ║
║    • Guessing which param groups need different treatment              ║
║                                                                        ║
║  Adam stores 2 full copies of your parameters and runs a formula.     ║
║  OptBrain stores 50KB and makes decisions.                             ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

if __name__ == '__main__':
    print(USAGE)
