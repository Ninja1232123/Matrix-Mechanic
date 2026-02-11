#!/usr/bin/env python3
"""
Heartbeat Native - Cardiac Rhythm Computation

Weights sit on a heart curve. Tokens arrive in a beat.
No matrix multiplication. The shape is the computation.

    flow = dot(token_pair, tangent_at_heart_position) * magnitude
    flip = dot(token_pair, normal_at_heart_position) * curvature * magnitude
    reaction = (flow + flip) * beat_intensity
"""

import numpy as np
import time


# ============================================
# THE HEART - 256 positions on a cardiac curve
# ============================================

def build_heart_curve(n=256):
    """
    x(t) = 16·sin³(t)
    y(t) = 13·cos(t) - 5·cos(2t) - 2·cos(3t) - cos(4t)
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float32)

    x = np.float32(16.0) * np.sin(t) ** 3
    y = (np.float32(13.0) * np.cos(t) - np.float32(5.0) * np.cos(2*t)
         - np.float32(2.0) * np.cos(3*t) - np.cos(4*t))

    scale = np.float32(max(np.abs(x).max(), np.abs(y).max()))
    x = x / scale
    y = y / scale
    positions = np.stack([x, y], axis=1).astype(np.float32)

    dx = np.gradient(x)
    dy = np.gradient(y)
    tmag = np.sqrt(dx**2 + dy**2).astype(np.float32)
    tmag = np.maximum(tmag, np.float32(1e-8))
    tangents = np.stack([dx / tmag, dy / tmag], axis=1).astype(np.float32)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1).astype(np.float32)

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (tmag ** 3)
    curvature = (curvature / np.maximum(curvature.max(), np.float32(1e-8))).astype(np.float32)

    return positions, tangents, normals, curvature


HEART_POS, HEART_TAN, HEART_NRM, HEART_CURV = build_heart_curve(256)


# ============================================
# THE BEAT
# ============================================

def build_heartbeat(n_tokens, bpm=72):
    """Cardiac rhythm: QRS spike + T wave + P wave + diastole rest."""
    beat_phase = np.linspace(0, np.float32(2 * np.pi) * max(1, n_tokens / 256),
                             n_tokens, dtype=np.float32)
    cycle = beat_phase % np.float32(2 * np.pi)

    qrs = np.exp(np.float32(-50.0) * (cycle - np.float32(0.9)) ** 2)
    t_wave = np.exp(np.float32(-10.0) * (cycle - np.float32(2.0)) ** 2) * np.float32(0.4)
    p_wave = np.exp(np.float32(-15.0) * (cycle - np.float32(5.5)) ** 2) * np.float32(0.2)

    beat = (qrs + t_wave + p_wave + np.float32(0.05)).astype(np.float32)
    beat = beat / np.maximum(beat.mean(), np.float32(1e-8))
    return beat, beat_phase


# ============================================
# HEARTBEAT LAYER
# ============================================

class HeartbeatLayer:
    """
    Weights on heart curve. Tokens arrive as 2D pairs.
    Flow along tangent + flip against normal * curvature.
    """

    def __init__(self, in_features, out_features, has_bias=False):
        self.in_features = in_features
        self.out_features = out_features
        self.n_groups = (in_features + 1) // 2
        self.padded_in = self.n_groups * 2

        self.heart_idx = np.zeros((out_features, self.n_groups), dtype=np.uint8)
        self.group_mag = np.zeros((out_features, self.n_groups), dtype=np.uint8)
        self.row_mag_scale = np.ones(out_features, dtype=np.float32)
        self.row_phase = np.zeros(out_features, dtype=np.float32)

        self.bias = np.zeros(out_features, dtype=np.float32) if has_bias else None

    def align_to_blueprint(self, fp32_weight, fp32_bias=None):
        """Distill FP32 weights onto heart curve."""
        out_f, in_f = fp32_weight.shape

        if in_f % 2 != 0:
            fp32_weight = np.pad(fp32_weight, ((0, 0), (0, 1)), mode='constant')

        w_pairs = fp32_weight.reshape(out_f, -1, 2)

        mags = np.linalg.norm(w_pairs, axis=-1).astype(np.float32)
        row_max = np.maximum(mags.max(axis=1), np.float32(1e-10))
        self.row_mag_scale = row_max
        self.group_mag = np.round(mags / row_max[:, np.newaxis] * np.float32(255.0)).astype(np.uint8)

        safe_mags = np.maximum(mags, 1e-10)[..., np.newaxis]
        w_unit = (w_pairs / safe_mags).astype(np.float32)

        # Find best heart position: max |dot(w_unit, tangent)|
        flat_w = w_unit.reshape(-1, 2)
        dots = flat_w @ HEART_TAN.T  # (N, 256)
        self.heart_idx = np.argmax(np.abs(dots), axis=1).astype(np.uint8).reshape(out_f, self.n_groups)

        row_mean = w_unit.mean(axis=1)
        self.row_phase = np.arctan2(row_mean[:, 1], row_mean[:, 0]).astype(np.float32)

        if fp32_bias is not None and self.bias is not None:
            self.bias = fp32_bias.astype(np.float32)

    def forward(self, x_fp32, beat_weights=None):
        """
        flow = dot(token_pair, tangent) * mag
        flip = dot(token_pair, normal) * curvature * mag
        out = sum(flow + flip) * beat * cos(row_phase)
        """
        seq_len = x_fp32.shape[0]
        in_f = x_fp32.shape[1]

        if in_f < self.padded_in:
            x_fp32 = np.pad(x_fp32, ((0, 0), (0, self.padded_in - in_f)), mode='constant')

        x_pairs = x_fp32[:, :self.padded_in].reshape(seq_len, self.n_groups, 2)

        w_tan = HEART_TAN[self.heart_idx]    # (out, n_groups, 2)
        w_nrm = HEART_NRM[self.heart_idx]    # (out, n_groups, 2)
        w_curv = HEART_CURV[self.heart_idx]   # (out, n_groups)

        mag_fp = self.group_mag.astype(np.float32) / np.float32(255.0)
        mag_fp = mag_fp * self.row_mag_scale[:, np.newaxis]

        # Flow: along tangent
        flow = np.einsum('snd,ond->so', x_pairs, w_tan * mag_fp[..., np.newaxis])

        # Flip: against normal, weighted by curvature
        flip = np.einsum('snd,ond->so', x_pairs, w_nrm * (mag_fp * w_curv)[..., np.newaxis])

        out = flow + flip

        if beat_weights is not None:
            out = out * beat_weights[:, np.newaxis]

        out = out * np.cos(self.row_phase)[np.newaxis, :]

        if self.bias is not None:
            out = out + self.bias

        return out.astype(np.float32)

    def storage_bytes(self):
        total = self.heart_idx.nbytes + self.group_mag.nbytes
        total += self.row_mag_scale.nbytes + self.row_phase.nbytes
        if self.bias is not None:
            total += self.bias.nbytes
        return total

    def equivalent_int8_bytes(self):
        return self.out_features * self.in_features


# ============================================
# FOUR HIGHWAY MODEL
# ============================================

class HeartbeatModel:
    """Four highways. Same tokens, four phases of the cardiac cycle."""

    def __init__(self, in_features, out_features, n_highways=4, has_bias=False):
        self.in_features = in_features
        self.out_features = out_features
        self.n_highways = n_highways

        self.layers = []
        for i in range(n_highways):
            self.layers.append(HeartbeatLayer(in_features, out_features, has_bias))

    def align_to_blueprint(self, fp32_weight, fp32_bias=None):
        """Each highway gets weights rotated by π/2."""
        out_f, in_f = fp32_weight.shape
        for i, layer in enumerate(self.layers):
            angle = np.float32(i * np.pi / 2)
            c, s = np.float32(np.cos(angle)), np.float32(np.sin(angle))

            w = fp32_weight.copy()
            if in_f % 2 != 0:
                w = np.pad(w, ((0, 0), (0, 1)), mode='constant')
            pairs = w.reshape(out_f, -1, 2)
            rotated = np.stack([
                pairs[..., 0] * c - pairs[..., 1] * s,
                pairs[..., 0] * s + pairs[..., 1] * c,
            ], axis=-1)
            layer.align_to_blueprint(rotated.reshape(out_f, -1)[:, :in_f].astype(np.float32), fp32_bias)

    def forward(self, x_fp32, bpm=72):
        """Four highways, each π/2 apart in the cardiac cycle."""
        seq_len = x_fp32.shape[0]
        _, beat_phase = build_heartbeat(seq_len, bpm=bpm)

        combined = np.zeros((seq_len, self.out_features), dtype=np.float32)

        for i, layer in enumerate(self.layers):
            phase_offset = np.float32(i * np.pi / 2)
            cycle = (beat_phase + phase_offset) % np.float32(2 * np.pi)

            qrs = np.exp(np.float32(-50.0) * (cycle - np.float32(0.9)) ** 2)
            t_wave = np.exp(np.float32(-10.0) * (cycle - np.float32(2.0)) ** 2) * np.float32(0.4)
            p_wave = np.exp(np.float32(-15.0) * (cycle - np.float32(5.5)) ** 2) * np.float32(0.2)
            hw_beat = (qrs + t_wave + p_wave + np.float32(0.05)).astype(np.float32)
            hw_beat = hw_beat / np.maximum(hw_beat.mean(), np.float32(1e-8))

            combined = combined + layer.forward(x_fp32, beat_weights=hw_beat)

        return combined


# ============================================
# TESTS
# ============================================

if __name__ == "__main__":

    print("=== HEART GEOMETRY ===")
    print(f"Positions: {HEART_POS.shape}")
    print(f"Tangents unit: {np.allclose(np.linalg.norm(HEART_TAN, axis=1), 1.0, atol=1e-3)}")
    print(f"Normals unit:  {np.allclose(np.linalg.norm(HEART_NRM, axis=1), 1.0, atol=1e-3)}")
    print(f"T·N = 0:       {np.allclose(np.sum(HEART_TAN * HEART_NRM, axis=1), 0.0, atol=1e-3)}")
    top_curv = np.argsort(HEART_CURV)[-3:]
    print(f"Highest curvature at: {top_curv} = {HEART_POS[top_curv]}")
    print()

    print("=== HEARTBEAT ===")
    beat, phase = build_heartbeat(100)
    print(f"100 tokens: peak={beat.max():.2f} min={beat.min():.3f} mean={beat.mean():.3f}")
    print()

    print("=== SINGLE LAYER (256x256) ===")
    layer = HeartbeatLayer(256, 256)
    fp32_w = np.random.randn(256, 256).astype(np.float32) * 0.02
    layer.align_to_blueprint(fp32_w)

    print(f"heart_idx:  {layer.heart_idx.shape} {layer.heart_idx.dtype}")
    print(f"group_mag:  {layer.group_mag.shape} {layer.group_mag.dtype}")
    print(f"Unique pos: {len(np.unique(layer.heart_idx))}/256")
    sb = layer.storage_bytes()
    ib = layer.equivalent_int8_bytes()
    print(f"Storage: {sb:,} vs INT8 {ib:,} = {sb/ib*100:.1f}%")

    x = np.random.randn(10, 256).astype(np.float32)
    y = layer.forward(x)
    print(f"Forward: {y.shape} {y.dtype} NaN={np.any(np.isnan(y))}")
    print(f"  range: [{y.min():.4f}, {y.max():.4f}]")

    beat_w, _ = build_heartbeat(10)
    y_beat = layer.forward(x, beat_weights=beat_w)
    print(f"With beat: [{y_beat.min():.4f}, {y_beat.max():.4f}]")
    print()

    print("=== FOUR HIGHWAYS (256x256) ===")
    model = HeartbeatModel(256, 256, n_highways=4)
    model.align_to_blueprint(fp32_w)

    t0 = time.time()
    y = model.forward(x, bpm=72)
    t1 = time.time()
    print(f"Output: {y.shape} NaN={np.any(np.isnan(y))}")
    print(f"Range:  [{y.min():.4f}, {y.max():.4f}]")
    print(f"Time:   {(t1-t0)*1000:.1f} ms")

    print("\nPhase diversity:")
    for i, l in enumerate(model.layers):
        print(f"  Highway {i}: mean_phase={l.row_phase.mean():.3f} "
              f"mean_pos={l.heart_idx.astype(float).mean():.1f}")

    print("\n=== BPM SENSITIVITY ===")
    for bpm in [40, 72, 120, 180]:
        y = model.forward(x, bpm=bpm)
        print(f"  BPM {bpm:>3}: mean={y.mean():+.5f} std={y.std():.4f}")

    print("\n=== SIZE ESTIMATES ===")
    for name, h, inter, nl in [("3B", 2560, 6912, 32), ("7B", 4096, 14336, 32)]:
        params_per_layer = 4 * h * h + 3 * h * inter
        total_params = nl * params_per_layer
        # Heartbeat: 2 bytes per pair (idx + mag) + row overhead
        n_grp = (h + 1) // 2
        n_grp_inter = (inter + 1) // 2
        heart_per_layer = (4 * h * n_grp * 2 + 3 * inter * n_grp * 2  # approximate
                           + (4 * h + 3 * inter) * 8)  # row scales + phases
        heart_total = nl * heart_per_layer
        int8_total = total_params
        fp32_total = total_params * 4
        # Four highways
        heart_4hw = heart_total * 4

        print(f"\n  {name} model ({total_params/1e9:.1f}B params):")
        print(f"    FP32:         {fp32_total/1e9:.1f} GB")
        print(f"    INT8:         {int8_total/1e9:.1f} GB")
        print(f"    Heart (1hw):  {heart_total/1e9:.1f} GB  ({heart_total/int8_total*100:.0f}% of INT8)")
        print(f"    Heart (4hw):  {heart_4hw/1e9:.1f} GB  ({heart_4hw/int8_total*100:.0f}% of INT8)")

    print()
    print("No matrix multiply. Just geometry and rhythm.")
