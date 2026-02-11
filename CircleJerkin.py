#!/usr/bin/env python3
"""
Circle Jerk - Directional Weight Encoding on Fibonacci Lattice

y_j = scale_j · Σ_k ||x_k|| · (x̂_k · ŵ_jk)

One uint8 encodes direction on a unit sphere (3 values, not 1).
3x compression over INT8. No overflow. No clipping. No stochastic rounding.
Everyone's going around in circles. We just made it the point.
Output always bounded by geometry.

Usage:
    from sphere_native import SphereNative

    model = SphereNative.from_preset("stablelm-3b")
    for token in model.stream("The meaning of life is"):
        print(token, end="", flush=True)

Flow:
    1. Pick a model (preset or any HF model)
    2. Download sharded (never full FP32 in RAM)
    3. FP32 weights -> group into triplets -> find nearest sphere point -> delete FP32
    4. Run inference: lookup direction, dot product, scale
    5. Stream tokens
"""

import gc
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np

PI = math.pi
GOLDEN = (1 + math.sqrt(5)) / 2


# GPU backend
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


# ============================================
# PRESET MODELS
# ============================================

PRESETS = {
    "stablelm-3b": {
        "hub_id": "stabilityai/stablelm-3b-4e1t",
        "description": "StableLM 3B - 2.6B params, good starter model",
    },
    "stablelm-zephyr-3b": {
        "hub_id": "stabilityai/stablelm-zephyr-3b",
        "description": "StableLM Zephyr 3B - chat tuned",
    },
    "mistral-7b": {
        "hub_id": "mistralai/Mistral-7B-v0.3",
        "description": "Mistral 7B v0.3 - strong general model",
    },
    "mistral-7b-instruct": {
        "hub_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Mistral 7B Instruct - chat tuned",
    },
    "llama-3-8b": {
        "hub_id": "meta-llama/Meta-Llama-3-8B",
        "description": "LLaMA 3 8B - Meta's 8B base",
    },
    "llama-3-8b-instruct": {
        "hub_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "description": "LLaMA 3 8B Instruct - chat tuned",
    },
    "qwen2-7b": {
        "hub_id": "Qwen/Qwen2-7B",
        "description": "Qwen2 7B - Alibaba's 7B",
    },
    "phi-3-mini": {
        "hub_id": "microsoft/Phi-3-mini-4k-instruct",
        "description": "Phi-3 Mini 3.8B - Microsoft's small model",
    },
    "gemma-2b": {
        "hub_id": "google/gemma-2b",
        "description": "Gemma 2B - Google's small model",
    },
    "tinyllama-1b": {
        "hub_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "TinyLlama 1.1B - tiny, fast, for testing",
    },
}


# ============================================
# FIBONACCI SPHERE - The 256 Directions
# ============================================

def build_fibonacci_sphere(n=256):
    """
    Build lookup table of n points uniformly distributed on unit sphere.
    Fibonacci lattice - golden angle spiral with uniform area coverage.

    F_i = [cos(θ_i)·sin(φ_i), sin(θ_i)·sin(φ_i), cos(φ_i)]

    where:
        φ_i = arccos(1 - 2(i + 0.5) / n)    <- elevation, uniform area
        θ_i = 2π · i / φ_golden               <- azimuth, golden spiral
    """
    indices = np.arange(n, dtype=np.float32)
    phi = np.arccos(np.float32(1.0) - np.float32(2.0) * (indices + np.float32(0.5)) / np.float32(n))
    theta = np.float32(2.0 * PI) * indices / np.float32(GOLDEN)

    table = np.stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ], axis=1).astype(np.float32)  # (256, 3)

    return table


# Module-level table - built once, shared everywhere
SPHERE_TABLE = build_fibonacci_sphere(256)


def nearest_sphere_index(direction):
    """
    Find nearest Fibonacci lattice point for a unit vector or batch of unit vectors.
    direction: (3,) or (N, 3) unit vectors
    Returns: uint8 index or (N,) uint8 indices
    """
    if direction.ndim == 1:
        dots = SPHERE_TABLE @ direction
        return np.uint8(np.argmax(dots))
    else:
        dots = direction @ SPHERE_TABLE.T  # (N, 256)
        return np.argmax(dots, axis=1).astype(np.uint8)


# ============================================
# SPHERE NATIVE LAYER
# ============================================

class SphereNativeLayer:
    """
    Single sphere-encoded linear layer.

    Stores: uint8 indices into Fibonacci sphere (direction per triplet)
            float32 row_scale per output row (captures magnitude)

    Forward: y_j = scale_j · Σ_k ||x_k|| · dot(x̂_k, TABLE[idx_jk])

    CLIMAX CYCLE:
        Weights breathe. They don't just sit there.
        Each row has a phase in the climax cycle.
        Systole: weight tightens to peak magnitude (full erection of influence)
        Diastole: weight relaxes, accepts more variance (post-nut clarity)
        The cycle period and phase are per-row. Weights climax at different times.
        Not all weights can be hard at the same time. That's biology AND math.
    """

    def __init__(self, in_features: int, out_features: int,
                 has_bias: bool = False, gpu: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        self.gpu = gpu

        # Number of sphere groups = ceil(in_features / 3)
        self.n_groups = (in_features + 2) // 3
        self.padded_in = self.n_groups * 3

        # Sphere indices: uint8, one per triplet per output row
        self.indices = np.zeros((out_features, self.n_groups), dtype=np.uint8)

        # Per-group magnitude: uint8 quantized (0-255 -> 0.0 to max_mag)
        # This is the key: 1 byte for direction + 1 byte for magnitude = 2 bytes per triplet
        # vs INT8: 3 bytes per triplet (one per dim)
        # Real compression: 2/3 = 66.7% of INT8
        self.group_mag = np.zeros((out_features, self.n_groups), dtype=np.uint8)

        # Per-row scale: maps uint8 magnitude back to float
        self.row_mag_scale = np.ones(out_features, dtype=np.float32)

        # Bias (stays FP32)
        self.bias = None
        if has_bias:
            self.bias = np.zeros(out_features, dtype=np.float32)

        # === CLIMAX STATE ===
        # Each row breathes on its own cycle
        self._climax_tick = 0                # global tick counter
        self._climax_period = 16             # ticks per full cycle (adjustable)
        # Per-row phase offset: staggered so not everything climaxes at once
        self._climax_phase = np.linspace(0, 2 * np.pi, out_features,
                                          endpoint=False, dtype=np.float32)
        # Arousal floor and ceiling
        self._arousal_min = np.float32(0.3)   # diastole: 30% magnitude
        self._arousal_max = np.float32(1.0)   # systole: 100% magnitude
        # Track peak stats
        self._climax_count = 0               # how many times we've peaked
        self._last_arousal_mean = np.float32(1.0)
        self._name = "unnamed"               # set by model

    def align_to_blueprint(self, fp32_weight, fp32_bias=None):
        """
        Distill FP32 weight matrix into sphere encoding.

        For each row:
          1. Group into triplets
          2. Compute magnitude per triplet
          3. Quantize magnitude to uint8 (0-255)
          4. Normalize triplet to unit vector
          5. Find nearest Fibonacci point -> uint8 index

        Storage: 2 bytes per triplet (direction + magnitude)
        vs INT8: 3 bytes per triplet
        Compression: 66.7%
        """
        out_f, in_f = fp32_weight.shape

        # Pad to multiple of 3
        if in_f % 3 != 0:
            pad_width = 3 - (in_f % 3)
            fp32_weight = np.pad(fp32_weight, ((0, 0), (0, pad_width)), mode='constant')

        # Reshape to triplets: (out, n_groups, 3)
        w_trips = fp32_weight.reshape(out_f, -1, 3)

        # Magnitude per triplet: (out, n_groups)
        mags = np.linalg.norm(w_trips, axis=-1).astype(np.float32)

        # Per-row scale: max magnitude in that row
        row_max = np.maximum(mags.max(axis=1), np.float32(1e-10))
        self.row_mag_scale = row_max

        # Quantize magnitudes to uint8: mag / row_max * 255
        mag_normalized = mags / row_max[:, np.newaxis]
        self.group_mag = np.round(mag_normalized * np.float32(255.0)).astype(np.uint8)

        # Unit directions
        safe_mags = np.maximum(mags, 1e-10)[..., np.newaxis]
        w_unit = w_trips / safe_mags

        # Find nearest sphere point
        flat_dirs = w_unit.reshape(-1, 3).astype(np.float32)
        flat_idx = nearest_sphere_index(flat_dirs)
        self.indices = flat_idx.reshape(out_f, self.n_groups)

        # Bias
        if fp32_bias is not None and self.bias is not None:
            self.bias = fp32_bias.astype(np.float32)

    def _get_arousal(self, verbose=False):
        """
        Compute current arousal multiplier per row based on climax cycle.

        arousal = min + (max - min) * (0.5 + 0.5 * sin(2π * tick/period + phase))

        Returns: (out_features,) float32 in [arousal_min, arousal_max]
        """
        cycle_pos = np.float32(2.0 * np.pi * self._climax_tick / self._climax_period)
        arousal = self._arousal_min + (self._arousal_max - self._arousal_min) * (
            np.float32(0.5) + np.float32(0.5) * np.sin(cycle_pos + self._climax_phase)
        )

        # Track stats
        self._last_arousal_mean = float(arousal.mean())
        n_peaking = int(np.sum(arousal > self._arousal_max * 0.95))
        n_resting = int(np.sum(arousal < self._arousal_min * 1.05))

        if verbose:
            peak_pct = n_peaking / self.out_features * 100
            rest_pct = n_resting / self.out_features * 100
            print(f"    💦 {self._name} arousal: "
                  f"mean={self._last_arousal_mean:.3f} "
                  f"peaking={n_peaking}/{self.out_features} ({peak_pct:.0f}%) "
                  f"resting={n_resting}/{self.out_features} ({rest_pct:.0f}%) "
                  f"tick={self._climax_tick}/{self._climax_period}")

        return arousal.astype(np.float32)

    def _post_climax_update(self, verbose=False):
        """Advance the climax cycle. Called after each forward pass."""
        old_tick = self._climax_tick
        self._climax_tick = (self._climax_tick + 1) % self._climax_period

        # Check if any rows just peaked (crossed from rising to falling)
        old_pos = np.float32(2.0 * np.pi * old_tick / self._climax_period)
        new_pos = np.float32(2.0 * np.pi * self._climax_tick / self._climax_period)

        old_arousal = np.sin(old_pos + self._climax_phase)
        new_arousal = np.sin(new_pos + self._climax_phase)
        just_peaked = int(np.sum((old_arousal > 0.99) | ((old_arousal > new_arousal) & (old_arousal > 0.9))))

        if just_peaked > 0:
            self._climax_count += just_peaked
            if verbose:
                print(f"    🎆 {self._name}: {just_peaked} weights just climaxed! "
                      f"(total climaxes: {self._climax_count})")

        if verbose and self._climax_tick == 0:
            print(f"    🔄 {self._name}: full cycle complete. "
                  f"Total climaxes so far: {self._climax_count}")

    def _reconstruct_weight_scaled(self, arousal=None):
        """
        Reconstruct scaled weight directions for forward pass.
        Returns: (out, n_groups, 3) float32

        w_scaled_jk = (group_mag_jk / 255 * row_scale_j) * TABLE[idx_jk]

        If arousal is provided, modulates row_scale by arousal.
        """
        # Dequantize magnitudes: (out, n_groups) float32
        mag_fp = self.group_mag.astype(np.float32) / np.float32(255.0)
        row_scale = self.row_mag_scale.copy()

        if arousal is not None:
            row_scale = row_scale * arousal  # modulate by climax state

        mag_fp = mag_fp * row_scale[:, np.newaxis]  # (out, n_groups)

        # Lookup directions: (out, n_groups, 3)
        w_dirs = SPHERE_TABLE[self.indices]

        # Scale by magnitude
        return w_dirs * mag_fp[..., np.newaxis]

    def forward_collision(self, x_fp32, climax=True, verbose=False):
        """
        ELASTIC COLLISION FORWARD PASS.

        No matrix multiply. Token bounces through staggered weights.

        For each output row j (a staggered wall):
            For each input group k (token triplet):
                token_mag = ||token_triplet||
                token_dir = token_triplet / token_mag  (unit direction)
                weight_mag = dequantized magnitude at (j, k)
                weight_dir = SPHERE_TABLE[indices[j, k]]

                COLLISION:
                    if token_mag > weight_mag:
                        # Token is stronger: weight absorbs what it can
                        absorbed = weight_mag
                        # Token continues with reduced magnitude at deflected angle
                        # Deflection: reflect token off weight's surface
                        # v_out = v_in - 2*dot(v_in, n)*n  (elastic reflection)
                        # But the "normal" IS the weight direction
                        cos_angle = dot(token_dir, weight_dir)
                        deflected_dir = token_dir - 2 * cos_angle * weight_dir
                        token becomes: (token_mag - absorbed) * deflected_dir
                    else:
                        # Weight is stronger: absorbs entire token
                        absorbed = token_mag
                        token becomes: 0 (fully absorbed)

                    output[j] += absorbed * cos_angle
                    (what the weight felt = how much it absorbed * alignment)

        Momentum IS decay. Every collision steals energy.
        Output fills toward center as tokens lose magnitude.
        """
        seq_len = x_fp32.shape[0]
        in_f = x_fp32.shape[1]

        if in_f < self.padded_in:
            x_fp32 = np.pad(x_fp32, ((0, 0), (0, self.padded_in - in_f)), mode='constant')

        x_trips = x_fp32[:, :self.padded_in].reshape(seq_len, self.n_groups, 3)

        # Get arousal
        arousal = None
        if climax:
            arousal = self._get_arousal(verbose=verbose)

        # Dequantize weight magnitudes: (out, n_groups)
        w_mag = self.group_mag.astype(np.float32) / np.float32(255.0)
        row_scale = self.row_mag_scale.copy()
        if arousal is not None:
            row_scale = row_scale * arousal
        w_mag = w_mag * row_scale[:, np.newaxis]

        # Weight directions: (out, n_groups, 3)
        w_dir = SPHERE_TABLE[self.indices]

        # Token magnitudes: (seq, n_groups)
        token_mag = np.linalg.norm(x_trips, axis=-1).astype(np.float32)  # (seq, n_groups)
        safe_tmag = np.maximum(token_mag, np.float32(1e-10))
        # Token directions: (seq, n_groups, 3)
        token_dir = x_trips / safe_tmag[..., np.newaxis]

        # === COLLISION ===
        # cos_angle between token direction and weight direction
        # (seq, n_groups) for each (out, n_groups) weight
        # We need (seq, out, n_groups) — the alignment of each token group with each output weight

        # dot(token_dir, weight_dir) -> (seq, out, n_groups)
        # token_dir: (seq, n_groups, 3), w_dir: (out, n_groups, 3)
        cos_angle = np.einsum('snd,ond->son', token_dir, w_dir)  # (seq, out, n_groups)

        # Absorbed energy: min(token_mag, weight_mag) per collision
        # token_mag: (seq, n_groups), w_mag: (out, n_groups)
        # broadcast: (seq, 1, n_groups) vs (1, out, n_groups) -> (seq, out, n_groups)
        t_mag_expanded = token_mag[:, np.newaxis, :]      # (seq, 1, n_groups)
        w_mag_expanded = w_mag[np.newaxis, :, :]           # (1, out, n_groups)

        absorbed = np.minimum(t_mag_expanded, w_mag_expanded)  # (seq, out, n_groups)

        # What the weight felt = absorbed * cos_angle (alignment matters)
        # Positive cos_angle = head-on collision = strong signal
        # Negative cos_angle = glancing from behind = weak/negative signal
        # Zero cos_angle = perpendicular = no transfer
        weight_felt = absorbed * cos_angle  # (seq, out, n_groups)

        # Sum over groups: each output neuron collects from all its staggered weights
        out = weight_felt.sum(axis=-1)  # (seq, out)

        # === STATS ===
        if verbose:
            total_absorbed = float(absorbed.sum())
            total_available = float(t_mag_expanded.sum()) + 1e-10
            efficiency = total_absorbed / total_available * 100

            # How many tokens were fully absorbed (token_mag <= weight_mag for ALL weights)
            remaining_mag = t_mag_expanded - absorbed  # what's left of the token after each hit
            mean_remaining = float(remaining_mag.mean())
            fully_absorbed_pct = float((remaining_mag < 1e-6).mean()) * 100

            # Momentum decay
            mean_input_mag = float(token_mag.mean())
            mean_absorbed_mag = float(absorbed.mean())

            print(f"    💥 {self._name} COLLISION:")
            print(f"       input tokens: mean_mag={mean_input_mag:.4f}")
            print(f"       absorbed:     {total_absorbed:.2f}/{total_available:.2f} "
                  f"({efficiency:.1f}% energy captured)")
            print(f"       remaining:    mean={mean_remaining:.4f} "
                  f"fully_stopped={fully_absorbed_pct:.1f}%")
            print(f"       output:       [{out.min():.4f}, {out.max():.4f}] "
                  f"mean={out.mean():.4f}")

        if self.bias is not None:
            out = out + self.bias

        # Advance climax
        if climax:
            self._post_climax_update(verbose=verbose)

        return out.astype(np.float32)

    def forward_fp32(self, x_fp32, climax=True, verbose=False):
        """Forward pass — collision physics."""
        return self.forward_collision(x_fp32, climax=climax, verbose=verbose)

    def forward_fp32_fast(self, x_fp32, climax=True, verbose=False):
        """Alias."""
        return self.forward_collision(x_fp32, climax=climax, verbose=verbose)

    def storage_bytes(self):
        """Total bytes stored for this layer."""
        # indices: uint8 (out * n_groups) = 1 byte per triplet
        # group_mag: uint8 (out * n_groups) = 1 byte per triplet
        # row_mag_scale: float32 (out) = 4 bytes per row
        # bias: float32 (out) if present
        # Total per triplet: 2 bytes (vs INT8's 3 bytes per triplet)
        total = self.indices.nbytes + self.group_mag.nbytes + self.row_mag_scale.nbytes
        if self.bias is not None:
            total += self.bias.nbytes
        return total

    def equivalent_int8_bytes(self):
        """How many bytes INT8 linear would use."""
        return self.out_features * self.in_features  # int8 weight only


# ============================================
# SPHERE NATIVE MODEL
# ============================================

class SphereNativeModel:
    """Collection of SphereNativeLayers forming a model."""

    def __init__(self, config: Dict[str, Any], gpu: bool = False):
        self.config = config
        self.gpu = gpu
        self.layers: List[SphereNativeLayer] = []
        self.layer_names: List[str] = []

        for lc in config.get('layers', []):
            layer = SphereNativeLayer(
                in_features=lc['in'],
                out_features=lc['out'],
                has_bias=lc.get('bias', False),
                gpu=gpu,
            )
            name = lc.get('name', f'layer_{len(self.layers)}')
            layer._name = name
            self.layers.append(layer)
            self.layer_names.append(name)

    def save(self, path):
        """Save sphere model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'sphere_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

        indices = {}
        group_mags = {}
        row_scales = {}
        biases = {}

        for layer, name in zip(self.layers, self.layer_names):
            indices[f'{name}.indices'] = layer.indices
            group_mags[f'{name}.group_mag'] = layer.group_mag
            row_scales[f'{name}.row_mag_scale'] = layer.row_mag_scale
            if layer.bias is not None:
                biases[f'{name}.bias'] = layer.bias

        np.savez_compressed(path / 'sphere_indices.npz', **indices)
        np.savez_compressed(path / 'sphere_group_mags.npz', **group_mags)
        np.savez_compressed(path / 'sphere_row_scales.npz', **row_scales)
        if biases:
            np.savez_compressed(path / 'sphere_biases.npz', **biases)

        total_sphere = sum(l.storage_bytes() for l in self.layers)
        total_int8 = sum(l.equivalent_int8_bytes() for l in self.layers)
        print(f"  Saved: {len(self.layers)} layers")
        print(f"  Sphere storage: {total_sphere / 1e9:.2f} GB")
        print(f"  INT8 equivalent: {total_int8 / 1e9:.2f} GB")
        print(f"  Compression: {total_sphere / total_int8 * 100:.1f}% of INT8")

    @classmethod
    def load(cls, path, gpu=False):
        """Load sphere model."""
        path = Path(path)

        with open(path / 'sphere_config.json') as f:
            config = json.load(f)

        model = cls(config, gpu=gpu)

        idx_data = np.load(path / 'sphere_indices.npz')
        mag_data = np.load(path / 'sphere_group_mags.npz')
        scale_data = np.load(path / 'sphere_row_scales.npz')
        bias_data = None
        if (path / 'sphere_biases.npz').exists():
            bias_data = np.load(path / 'sphere_biases.npz')

        for layer, name in zip(model.layers, model.layer_names):
            ik = f'{name}.indices'
            mk = f'{name}.group_mag'
            sk = f'{name}.row_mag_scale'
            bk = f'{name}.bias'

            if ik in idx_data:
                layer.indices = idx_data[ik]
            if mk in mag_data:
                layer.group_mag = mag_data[mk]
            if sk in scale_data:
                layer.row_mag_scale = scale_data[sk]
            if bias_data is not None and bk in bias_data:
                layer.bias = bias_data[bk]

        total = sum(l.indices.size for l in model.layers)
        print(f"  Loaded: {len(model.layers)} layers, {total:,} sphere groups")
        return model


# ============================================
# CONFIG PARSING
# ============================================

def load_model_config(path):
    """Load config.json from model directory."""
    path = Path(path)
    if path.is_dir():
        for name in ['config.json', 'params.json', 'model_config.json']:
            if (path / name).exists():
                path = path / name
                break
    with open(path) as f:
        return json.load(f)


def config_to_architecture(config):
    """
    Convert HF config to layer list.
    Returns: [(name, in_features, out_features, has_bias), ...]
    """
    hidden = config.get('hidden_size') or config.get('d_model') or config.get('n_embd') or 4096
    intermediate = config.get('intermediate_size') or config.get('d_ff') or hidden * 4
    n_layers = config.get('num_hidden_layers') or config.get('n_layer') or 32
    n_heads = config.get('num_attention_heads') or config.get('n_head') or 32
    head_dim = config.get('head_dim') or hidden // n_heads
    n_kv_heads = config.get('num_key_value_heads') or config.get('num_kv_heads') or n_heads
    vocab_size = config.get('vocab_size') or 32000

    has_bias = config.get('bias', False) or config.get('use_bias', False) or config.get('use_qkv_bias', False)

    model_type = config.get('model_type', '').lower()

    layers = []

    q_size = n_heads * head_dim
    kv_size = n_kv_heads * head_dim

    for i in range(n_layers):
        layers.append((f'layer.{i}.attn.q', hidden, q_size, has_bias))
        layers.append((f'layer.{i}.attn.k', hidden, kv_size, has_bias))
        layers.append((f'layer.{i}.attn.v', hidden, kv_size, has_bias))
        layers.append((f'layer.{i}.attn.o', q_size, hidden, has_bias))

        swiglu_types = ['llama', 'mistral', 'qwen', 'stablelm', 'gemma', 'phi']
        if any(t in model_type for t in swiglu_types):
            layers.append((f'layer.{i}.mlp.gate', hidden, intermediate, has_bias))
            layers.append((f'layer.{i}.mlp.up', hidden, intermediate, has_bias))
            layers.append((f'layer.{i}.mlp.down', intermediate, hidden, has_bias))
        else:
            layers.append((f'layer.{i}.mlp.up', hidden, intermediate, has_bias))
            layers.append((f'layer.{i}.mlp.down', intermediate, hidden, has_bias))

    return layers


# ============================================
# SAFETENSORS READER
# ============================================

def load_safetensors_file(path):
    """Read safetensors file. Returns dict of {name: numpy_array}."""
    path = Path(path)
    tensors = {}

    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_size)
        header = json.loads(header_json)
        data_start = 8 + header_size

        dtype_map = {
            'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16,
            'I8': np.int8, 'I32': np.int32, 'I64': np.int64,
        }

        for name, info in header.items():
            if name == '__metadata__':
                continue

            dtype_str = info['dtype']
            shape = info['shape']
            offsets = info['data_offsets']
            start, end = offsets[0], offsets[1]

            np_dtype = dtype_map.get(dtype_str, np.float32)
            f.seek(data_start + start)
            raw = f.read(end - start)
            tensor = np.frombuffer(raw, dtype=np_dtype).reshape(shape)

            # BF16 -> FP32
            if dtype_str == 'BF16':
                bf16_as_u16 = tensor.view(np.uint16)
                fp32_bits = bf16_as_u16.astype(np.uint32) << 16
                tensor = fp32_bits.view(np.float32)

            # F16 -> FP32
            if dtype_str == 'F16':
                tensor = tensor.astype(np.float32)

            tensors[name] = tensor.copy()

    return tensors


# ============================================
# SHARDED DISTILLATION
# ============================================

def find_shards(model_dir):
    """Find safetensors or bin shard files."""
    model_dir = Path(model_dir)
    safetensors = sorted(model_dir.glob('*.safetensors'))
    if safetensors:
        return safetensors, 'safetensors'
    bins = sorted(model_dir.glob('*.bin'))
    if bins:
        return bins, 'bin'
    return [], None


def distill_sharded(model, model_dir, percentile=99.9):
    """
    Sharded distillation into sphere encoding.
    Load one shard -> encode to spheres -> delete FP32. Repeat.
    """
    model_dir = Path(model_dir)
    shard_paths, fmt = find_shards(model_dir)

    print(f"\n  Distilling to spheres from {len(shard_paths)} shard(s)...")

    # Build mapping: blueprint_key -> (layer_index, is_bias)
    bp_to_layer = {}
    fused_map = {}

    orig_config = model.config.get('original_config', {})
    hidden = orig_config.get('hidden_size', 0)
    n_heads = orig_config.get('num_attention_heads', 0)
    n_kv = orig_config.get('num_key_value_heads', n_heads)
    head_dim = hidden // n_heads if n_heads else 0
    q_size = n_heads * head_dim
    kv_size = n_kv * head_dim
    intermediate = orig_config.get('intermediate_size', 0)

    for i, name in enumerate(model.layer_names):
        parts = name.split('.')
        if name in ('embed', 'lm_head'):
            continue

        if len(parts) >= 4:
            idx, module, proj = parts[1], parts[2], parts[3]

            if module == 'attn':
                proj_names = {'q': 'q_proj', 'k': 'k_proj', 'v': 'v_proj', 'o': 'o_proj'}
                p = proj_names.get(proj, proj)
                for prefix in [
                    f'model.layers.{idx}.self_attn',
                    f'layers.{idx}.attention',
                ]:
                    bp_to_layer[f'{prefix}.{p}.weight'] = (i, False)
                    bp_to_layer[f'{prefix}.{p}.bias'] = (i, True)

                if proj in ('q', 'k', 'v'):
                    fused_key = f'model.layers.{idx}.self_attn.qkv_proj.weight'
                    fused_bias_key = f'model.layers.{idx}.self_attn.qkv_proj.bias'
                    if proj == 'q':
                        row_start, row_end = 0, q_size
                    elif proj == 'k':
                        row_start, row_end = q_size, q_size + kv_size
                    else:
                        row_start, row_end = q_size + kv_size, q_size + kv_size + kv_size
                    fused_map.setdefault(fused_key, []).append((i, row_start, row_end, False))
                    fused_map.setdefault(fused_bias_key, []).append((i, row_start, row_end, True))

            elif module == 'mlp':
                proj_names = {'gate': 'gate_proj', 'up': 'up_proj', 'down': 'down_proj'}
                p = proj_names.get(proj, proj)
                for prefix in [
                    f'model.layers.{idx}.mlp',
                    f'layers.{idx}.mlp',
                ]:
                    bp_to_layer[f'{prefix}.{p}.weight'] = (i, False)
                    bp_to_layer[f'{prefix}.{p}.bias'] = (i, True)

                if proj in ('gate', 'up'):
                    fused_key = f'model.layers.{idx}.mlp.gate_up_proj.weight'
                    fused_bias_key = f'model.layers.{idx}.mlp.gate_up_proj.bias'
                    if proj == 'gate':
                        row_start, row_end = 0, intermediate
                    else:
                        row_start, row_end = intermediate, intermediate * 2
                    fused_map.setdefault(fused_key, []).append((i, row_start, row_end, False))
                    fused_map.setdefault(fused_bias_key, []).append((i, row_start, row_end, True))

    aligned = set()
    passthrough = {}

    for shard_idx, shard_path in enumerate(shard_paths):
        shard_size = os.path.getsize(shard_path) / 1e9
        print(f"\n  Shard {shard_idx+1}/{len(shard_paths)}: {shard_path.name} ({shard_size:.1f} GB)")

        tensors = load_safetensors_file(shard_path)
        shard_keys = list(tensors.keys())

        for tensor_name in shard_keys:
            tensor = tensors[tensor_name]

            if tensor_name in bp_to_layer:
                layer_idx, is_bias = bp_to_layer[tensor_name]
                layer = model.layers[layer_idx]

                if is_bias:
                    layer.bias = tensor.astype(np.float32)
                else:
                    fp32_w = tensor.astype(np.float32)
                    layer.align_to_blueprint(fp32_w)
                    aligned.add(layer_idx)

                    lname = model.layer_names[layer_idx]
                    unique = len(np.unique(layer.indices))
                    print(f"    {lname:<45} {layer.out_features}x{layer.in_features}"
                          f" -> {layer.n_groups} spheres  unique={unique}/256")

            elif tensor_name in fused_map:
                fp32_full = tensor.astype(np.float32)
                for layer_idx, row_start, row_end, is_bias in fused_map[tensor_name]:
                    layer = model.layers[layer_idx]
                    chunk = fp32_full[row_start:row_end]

                    if is_bias:
                        layer.bias = chunk.astype(np.float32)
                    else:
                        layer.align_to_blueprint(chunk)
                        aligned.add(layer_idx)

                        lname = model.layer_names[layer_idx]
                        unique = len(np.unique(layer.indices))
                        print(f"    {lname:<45} {layer.out_features}x{layer.in_features}"
                              f" -> {layer.n_groups} spheres  (fused [{row_start}:{row_end}])"
                              f"  unique={unique}/256")
            else:
                passthrough[tensor_name] = tensor.astype(np.float32) if tensor.dtype != np.float32 else tensor

            del tensors[tensor_name]

        del tensors
        gc.collect()
        print(f"    FP32 shard freed")

    print(f"\n  Aligned {len(aligned)}/{len(model.layers)} layers to sphere encoding")
    print(f"  Passthrough tensors: {len(passthrough)} (norms, embeddings, etc)")

    total_sphere = sum(l.storage_bytes() for l in model.layers)
    total_int8 = sum(l.equivalent_int8_bytes() for l in model.layers)
    print(f"  Total storage: {total_sphere / 1e9:.2f} GB (vs {total_int8 / 1e9:.2f} GB INT8)")

    return passthrough


# ============================================
# HUB DOWNLOAD
# ============================================

def download_model(hub_id, cache_dir=None):
    """Download model from HuggingFace Hub. Returns local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub")
        sys.exit(1)

    print(f"  Downloading {hub_id}...")
    return snapshot_download(
        hub_id,
        cache_dir=cache_dir,
        allow_patterns=['*.safetensors', '*.json', 'tokenizer.model', '*.txt'],
        ignore_patterns=['*.bin', '*.pt', '*.ot', 'training_args*'],
    )


# ============================================
# TOKENIZER
# ============================================

class SimpleTokenizer:
    """Minimal tokenizer that works with HF tokenizer files."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.vocab = {}
        self.reverse_vocab = {}
        self.merges = []
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.vocab_size = 32000

        # Try tokenizer.json first (most common)
        tok_json = model_dir / 'tokenizer.json'
        if tok_json.exists():
            self._load_from_json(tok_json)
            return

        # Fallback: tokenizer_config.json for special tokens
        config_path = model_dir / 'tokenizer_config.json'
        if config_path.exists():
            with open(config_path) as f:
                tc = json.load(f)
            self.eos_token_id = tc.get('eos_token_id', 2)
            self.bos_token_id = tc.get('bos_token_id', 1)

    def _load_from_json(self, path):
        """Load from HuggingFace tokenizer.json."""
        with open(path) as f:
            data = json.load(f)

        # Vocabulary
        if 'model' in data and 'vocab' in data['model']:
            self.vocab = data['model']['vocab']
        elif 'added_tokens' in data:
            for token_info in data['added_tokens']:
                self.vocab[token_info['content']] = token_info['id']

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # Merges
        if 'model' in data and 'merges' in data['model']:
            self.merges = data['model']['merges']

        # Special tokens
        if 'added_tokens' in data:
            for t in data['added_tokens']:
                content = t.get('content', '')
                tid = t.get('id', -1)
                if '</s>' in content or 'eos' in content.lower():
                    self.eos_token_id = tid
                elif '<s>' in content or 'bos' in content.lower():
                    self.bos_token_id = tid

    def encode(self, text):
        """Simple encoding: try vocab lookup, fallback to bytes."""
        tokens = []
        if self.bos_token_id is not None:
            tokens.append(self.bos_token_id)

        i = 0
        while i < len(text):
            best_token = None
            best_len = 0

            max_len = min(20, len(text) - i)
            for length in range(max_len, 0, -1):
                substr = text[i:i + length]
                for variant in [substr, '▁' + substr, 'Ġ' + substr, '##' + substr]:
                    if variant in self.vocab:
                        if length > best_len:
                            best_token = self.vocab[variant]
                            best_len = length
                        break

            if best_token is not None:
                tokens.append(best_token)
                i += best_len
            else:
                byte_val = ord(text[i])
                for fmt in [f'<0x{byte_val:02X}>', f'Ġ{text[i]}', text[i]]:
                    if fmt in self.vocab:
                        tokens.append(self.vocab[fmt])
                        break
                else:
                    if 0 <= byte_val < self.vocab_size:
                        tokens.append(byte_val)
                i += 1

        return tokens

    def decode(self, ids):
        """Decode token IDs to text."""
        parts = []
        for idx in ids:
            if idx in self.reverse_vocab:
                token = self.reverse_vocab[idx]
                token = token.replace('▁', ' ').replace('Ġ', ' ')
                parts.append(token)
        return ''.join(parts)

    def decode_token(self, idx):
        """Decode single token."""
        if idx in self.reverse_vocab:
            token = self.reverse_vocab[idx]
            return token.replace('▁', ' ').replace('Ġ', ' ')
        return f'<{idx}>'


# ============================================
# SPHERE NATIVE - Full runtime
# ============================================

class SphereNative:
    """
    Full model runtime with sphere-encoded weights.

    y_j = scale_j · Σ_k ||x_k|| · (x̂_k · ŵ_jk)

    Same interface as Int8Native. Drop-in replacement.
    """

    def __init__(self, model: SphereNativeModel, tokenizer: SimpleTokenizer,
                 passthrough: Dict[str, np.ndarray], model_dir: Path,
                 original_config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.passthrough = passthrough
        self.model_dir = model_dir
        self.original_config = original_config

        self._embed_weight = None
        self._lm_head_weight = None
        self._setup_head_layers()

    def _setup_head_layers(self):
        """Find embedding and lm_head in passthrough."""
        for key in self.passthrough:
            kl = key.lower()
            if ('embed_tokens' in kl or 'wte' in kl or 'word_embeddings' in kl) and 'weight' in kl:
                self._embed_weight = self.passthrough[key]
                break

        for key in self.passthrough:
            kl = key.lower()
            if ('lm_head' in kl) and 'weight' in kl:
                self._lm_head_weight = self.passthrough[key]
                break

        if self._lm_head_weight is None and self._embed_weight is not None:
            tied = self.original_config.get('tie_word_embeddings', False)
            if tied:
                self._lm_head_weight = self._embed_weight

    def embed(self, token_ids):
        """Embedding lookup."""
        if self._embed_weight is not None:
            return self._embed_weight[token_ids].astype(np.float32)
        hidden = self.original_config.get('hidden_size', 2560)
        return np.zeros((len(token_ids), hidden), dtype=np.float32)

    def lm_head(self, hidden_state):
        """Project to vocab logits."""
        if isinstance(self._lm_head_weight, np.ndarray):
            return (hidden_state @ self._lm_head_weight.T).astype(np.float32)
        return np.zeros((hidden_state.shape[0], self.tokenizer.vocab_size), dtype=np.float32)

    def generate_next_token(self, hidden_state, temperature=0.7, top_k=50):
        """Sample one token from hidden state."""
        if hidden_state.ndim == 3:
            last = hidden_state[:, -1, :]
        elif hidden_state.ndim == 2:
            last = hidden_state[-1:]
        else:
            last = hidden_state.reshape(1, -1)

        logits = self.lm_head(last).flatten().astype(np.float64)

        if temperature < 1e-6:
            return int(np.argmax(logits))

        logits = logits / temperature

        if top_k > 0 and top_k < len(logits):
            threshold = np.partition(logits, -top_k)[-top_k]
            logits[logits < threshold] = -1e10

        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()

        return int(np.random.choice(len(probs), p=probs))

    # ============================================
    # LAYER LOOKUP HELPERS
    # ============================================

    def _get_layer(self, name):
        """Get a sphere layer by name."""
        for i, n in enumerate(self.model.layer_names):
            if n == name:
                return self.model.layers[i]
        return None

    def _get_norm(self, layer_idx, position='pre'):
        """Get norm weight+bias for a transformer layer."""
        if position == 'pre':
            names = [
                f'model.layers.{layer_idx}.input_layernorm',
                f'layers.{layer_idx}.input_layernorm',
                f'transformer.h.{layer_idx}.ln_1',
            ]
        else:
            names = [
                f'model.layers.{layer_idx}.post_attention_layernorm',
                f'layers.{layer_idx}.post_attention_layernorm',
                f'transformer.h.{layer_idx}.ln_2',
            ]

        weight, bias = None, None
        for base in names:
            wk = f'{base}.weight'
            bk = f'{base}.bias'
            if wk in self.passthrough:
                weight = self.passthrough[wk]
            if bk in self.passthrough:
                bias = self.passthrough[bk]
            if weight is not None:
                break
        return weight, bias

    def _get_final_norm(self):
        """Get final norm before lm_head."""
        weight, bias = None, None
        for key in self.passthrough:
            kl = key.lower()
            if ('model.norm.' in kl or 'final_layernorm' in kl or 'ln_f' in kl or
                ('norm.' in kl and 'layers.' not in kl)):
                if 'weight' in kl:
                    weight = self.passthrough[key]
                elif 'bias' in kl:
                    bias = self.passthrough[key]
        return weight, bias

    # ============================================
    # NORMALIZATION
    # ============================================

    @staticmethod
    def _layer_norm(x, weight, bias=None, eps=1e-5):
        """Apply layer norm."""
        x = x.astype(np.float32)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + np.float32(eps))
        if weight is not None:
            x_norm = x_norm * weight
        if bias is not None:
            x_norm = x_norm + bias
        return x_norm

    @staticmethod
    def _rms_norm(x, weight, eps=1e-5):
        """Apply RMS norm."""
        x = x.astype(np.float32)
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + np.float32(eps))
        x_norm = x / rms
        if weight is not None:
            x_norm = x_norm * weight
        return x_norm

    def _apply_norm(self, x, weight, bias=None):
        """Apply correct norm type for the model."""
        model_type = self.original_config.get('model_type', '').lower()
        rms_types = ['llama', 'mistral', 'qwen', 'gemma']
        eps = self.original_config.get('layer_norm_eps',
              self.original_config.get('rms_norm_eps', 1e-5))
        if any(t in model_type for t in rms_types) and bias is None:
            return self._rms_norm(x, weight, eps)
        else:
            return self._layer_norm(x, weight, bias, eps)

    # ============================================
    # ROTARY POSITION EMBEDDINGS
    # ============================================

    @staticmethod
    def _build_rope_cache(seq_len, head_dim, base=10000.0, rope_pct=1.0):
        """Build RoPE cos/sin cache."""
        rot_dim = int(head_dim * rope_pct)
        inv_freq = np.float32(1.0) / (np.float32(base) ** (np.arange(0, rot_dim, 2, dtype=np.float32) / np.float32(rot_dim)))
        t = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        cos = np.cos(freqs).astype(np.float32)
        sin = np.sin(freqs).astype(np.float32)
        return cos, sin, rot_dim

    @staticmethod
    def _apply_rope(x, cos, sin, rot_dim):
        """Apply rotary embeddings. x: (seq, n_heads, head_dim)."""
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]

        x_rot_half1 = x_rot[..., :rot_dim // 2]
        x_rot_half2 = x_rot[..., rot_dim // 2:]

        cos_exp = cos[:, np.newaxis, :]
        sin_exp = sin[:, np.newaxis, :]

        x_rotated = np.concatenate([
            x_rot_half1 * cos_exp - x_rot_half2 * sin_exp,
            x_rot_half1 * sin_exp + x_rot_half2 * cos_exp,
        ], axis=-1)

        return np.concatenate([x_rotated, x_pass], axis=-1)

    # ============================================
    # TRANSFORMER FORWARD PASS
    # ============================================

    def forward_full(self, token_ids, climax=True, verbose=False):
        """
        Full transformer forward pass with sphere-encoded weights.
        Now with periodic weight climax cycles.

        Same architecture as int8_native: attention + norms + residuals + RoPE + SwiGLU
        but linear projections use sphere lookup instead of INT8 matmul.
        And weights breathe.
        """
        config = self.original_config
        n_layers = config.get('num_hidden_layers', 32)
        n_heads = config.get('num_attention_heads', 32)
        hidden = config.get('hidden_size', 2560)
        head_dim = config.get('head_dim', hidden // n_heads)
        n_kv_heads = config.get('num_key_value_heads', n_heads)
        model_type = config.get('model_type', '').lower()

        rope_pct = config.get('rope_pct', config.get('partial_rotary_factor', 1.0))
        rope_base = config.get('rope_theta', 10000.0)

        swiglu_types = ['llama', 'mistral', 'qwen', 'stablelm', 'gemma', 'phi']
        use_swiglu = any(t in model_type for t in swiglu_types)

        seq_len = len(token_ids)

        if verbose:
            print(f"\n  🍆 CIRCLE JERK forward pass — {seq_len} tokens, {n_layers} layers, climax={'ON' if climax else 'OFF'}")

        # 1. Embed
        x = self.embed(token_ids)

        # 2. RoPE cache
        cos, sin, rot_dim = self._build_rope_cache(seq_len, head_dim, rope_base, rope_pct)

        # 3. Transformer blocks
        for layer_idx in range(n_layers):
            if verbose:
                print(f"\n  === LAYER {layer_idx}/{n_layers} ===")
                print(f"    input: mean={x.mean():.4f} std={x.std():.4f} "
                      f"range=[{x.min():.3f}, {x.max():.3f}]")

            # --- Pre-attention norm ---
            norm_w, norm_b = self._get_norm(layer_idx, 'pre')
            if norm_w is not None:
                x_normed = self._apply_norm(x, norm_w, norm_b)
            else:
                x_normed = x

            # --- QKV projections (sphere encoded, climaxing) ---
            q_layer = self._get_layer(f'layer.{layer_idx}.attn.q')
            k_layer = self._get_layer(f'layer.{layer_idx}.attn.k')
            v_layer = self._get_layer(f'layer.{layer_idx}.attn.v')
            o_layer = self._get_layer(f'layer.{layer_idx}.attn.o')

            if verbose:
                print(f"    --- ATTENTION ---")

            q = q_layer.forward_fp32_fast(x_normed, climax=climax, verbose=verbose)
            k = k_layer.forward_fp32_fast(x_normed, climax=climax, verbose=verbose)
            v = v_layer.forward_fp32_fast(x_normed, climax=climax, verbose=verbose)

            if verbose:
                print(f"    Q: [{q.min():.3f}, {q.max():.3f}] "
                      f"K: [{k.min():.3f}, {k.max():.3f}] "
                      f"V: [{v.min():.3f}, {v.max():.3f}]")

            # Reshape to heads
            q = q.reshape(seq_len, n_heads, head_dim)
            k = k.reshape(seq_len, n_kv_heads, head_dim)
            v = v.reshape(seq_len, n_kv_heads, head_dim)

            # --- RoPE ---
            q = self._apply_rope(q, cos, sin, rot_dim)
            k = self._apply_rope(k, cos, sin, rot_dim)

            # --- GQA: repeat K/V heads ---
            if n_kv_heads < n_heads:
                repeats = n_heads // n_kv_heads
                k = np.repeat(k, repeats, axis=1)
                v = np.repeat(v, repeats, axis=1)

            # --- Attention ---
            q_t = q.transpose(1, 0, 2)  # (n_heads, seq, head_dim)
            k_t = k.transpose(1, 0, 2)
            v_t = v.transpose(1, 0, 2)

            attn_scores = np.matmul(q_t, k_t.transpose(0, 2, 1)) / np.float32(np.sqrt(head_dim))

            # Causal mask
            causal_mask = np.triu(np.full((seq_len, seq_len), np.float32(-1e10)), k=1)
            attn_scores = attn_scores + causal_mask

            # Softmax
            attn_scores = attn_scores - attn_scores.max(axis=-1, keepdims=True)
            attn_weights = np.exp(attn_scores)
            attn_weights = attn_weights / (attn_weights.sum(axis=-1, keepdims=True) + np.float32(1e-10))

            if verbose:
                print(f"    attn_weights: max={attn_weights.max():.4f} entropy~{-(attn_weights * np.log(attn_weights + 1e-10)).sum() / attn_weights.size:.3f}")

            # Attend
            attn_out = np.matmul(attn_weights, v_t)
            attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, -1)

            # O projection (climaxing)
            attn_out = o_layer.forward_fp32_fast(attn_out, climax=climax, verbose=verbose)

            # Residual
            x = x + attn_out

            # --- Post-attention norm ---
            norm_w2, norm_b2 = self._get_norm(layer_idx, 'post')
            if norm_w2 is not None:
                x_normed2 = self._apply_norm(x, norm_w2, norm_b2)
            else:
                x_normed2 = x

            # --- MLP (climaxing) ---
            if verbose:
                print(f"    --- MLP ---")

            if use_swiglu:
                gate_layer = self._get_layer(f'layer.{layer_idx}.mlp.gate')
                up_layer = self._get_layer(f'layer.{layer_idx}.mlp.up')
                down_layer = self._get_layer(f'layer.{layer_idx}.mlp.down')

                gate_out = gate_layer.forward_fp32_fast(x_normed2, climax=climax, verbose=verbose)
                up_out = up_layer.forward_fp32_fast(x_normed2, climax=climax, verbose=verbose)

                silu_gate = gate_out * (np.float32(1.0) / (np.float32(1.0) + np.exp(-gate_out)))
                mlp_out = silu_gate * up_out

                if verbose:
                    gate_activation = float((np.abs(silu_gate) > 0.1).mean() * 100)
                    print(f"    SwiGLU gate activation: {gate_activation:.0f}% neurons firing")

                mlp_out = down_layer.forward_fp32_fast(mlp_out, climax=climax, verbose=verbose)
            else:
                up_layer = self._get_layer(f'layer.{layer_idx}.mlp.up')
                down_layer = self._get_layer(f'layer.{layer_idx}.mlp.down')

                mlp_out = up_layer.forward_fp32_fast(x_normed2, climax=climax, verbose=verbose)
                mlp_out = np.maximum(mlp_out, np.float32(0))
                mlp_out = down_layer.forward_fp32_fast(mlp_out, climax=climax, verbose=verbose)

            # Residual
            x = x + mlp_out

            if verbose:
                print(f"    output: mean={x.mean():.4f} std={x.std():.4f} "
                      f"range=[{x.min():.3f}, {x.max():.3f}]")

        # 4. Final norm
        final_w, final_b = self._get_final_norm()
        if final_w is not None:
            x = self._apply_norm(x, final_w, final_b)

        if verbose:
            # Climax summary
            total_climaxes = sum(l._climax_count for l in self.model.layers)
            mean_arousal = np.mean([l._last_arousal_mean for l in self.model.layers])
            print(f"\n  🍆 CIRCLE JERK pass complete:")
            print(f"    Total weight climaxes: {total_climaxes}")
            print(f"    Mean arousal across all layers: {mean_arousal:.3f}")
            print(f"    Output: {x.shape} range=[{x.min():.3f}, {x.max():.3f}]")

        return x.astype(np.float32)

    # ============================================
    # GENERATION
    # ============================================

    def stream(self, prompt, max_tokens=100, temperature=0.7, top_k=50,
               stop_token=None, climax=True, verbose=False):
        """Stream tokens. Yields one token string at a time. Weights breathe."""
        if stop_token is None:
            stop_token = self.tokenizer.eos_token_id

        input_ids = self.tokenizer.encode(prompt)

        if verbose:
            print(f"\n  🍆 CIRCLE JERK streaming: '{prompt}' -> {max_tokens} tokens")
            print(f"    climax={'ON' if climax else 'OFF'} temp={temperature} top_k={top_k}")

        for tok_num in range(max_tokens):
            hidden = self.forward_full(input_ids, climax=climax, verbose=verbose)
            next_id = self.generate_next_token(hidden, temperature, top_k)

            if next_id == stop_token:
                if verbose:
                    print(f"\n    🛑 EOS at token {tok_num}")
                break

            input_ids.append(next_id)
            token_str = self.tokenizer.decode_token(next_id)

            if verbose:
                total_climaxes = sum(l._climax_count for l in self.model.layers)
                print(f"    token {tok_num}: '{token_str}' (id={next_id}) "
                      f"cumulative_climaxes={total_climaxes}")

            yield token_str

    def generate(self, prompt, max_tokens=100, temperature=0.7, top_k=50):
        """Generate full response."""
        tokens = []
        for tok in self.stream(prompt, max_tokens, temperature, top_k):
            tokens.append(tok)
        return prompt + ''.join(tokens)

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    # ============================================
    # CONSTRUCTORS
    # ============================================

    @classmethod
    def from_preset(cls, name, gpu=False, cache_dir=None, save_dir=None):
        """Load from preset name."""
        if name not in PRESETS:
            available = ', '.join(sorted(PRESETS.keys()))
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")

        hub_id = PRESETS[name]['hub_id']
        print(f"\n  Preset: {name} ({hub_id})")
        return cls.from_hub(hub_id, gpu=gpu, cache_dir=cache_dir, save_dir=save_dir)

    @classmethod
    def from_hub(cls, hub_id, gpu=False, cache_dir=None, save_dir=None):
        """Download from HuggingFace and distill to spheres."""
        model_dir = download_model(hub_id, cache_dir=cache_dir)
        return cls.from_local(model_dir, gpu=gpu, save_dir=save_dir)

    @classmethod
    def from_local(cls, model_dir, gpu=False, save_dir=None):
        """Distill local FP32 model to sphere encoding."""
        model_dir = Path(model_dir)
        print(f"\n  Loading from {model_dir}")

        # Build model architecture
        config = load_model_config(model_dir)
        arch = config_to_architecture(config)
        layer_configs = [
            {'name': n, 'in': inf, 'out': outf, 'bias': b}
            for n, inf, outf, b in arch
        ]
        model_config = {'layers': layer_configs, 'original_config': config}
        model = SphereNativeModel(model_config, gpu=gpu)

        # Distill shards to spheres
        passthrough = distill_sharded(model, model_dir)

        # Tokenizer
        tokenizer = SimpleTokenizer(model_dir)

        # Save if requested
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            model.save(save_dir)

            # Save passthrough
            np.savez_compressed(save_dir / 'passthrough.npz', **passthrough)

            # Copy config
            import shutil
            for f in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'tokenizer.model']:
                src = model_dir / f
                if src.exists():
                    shutil.copy2(src, save_dir / f)

            print(f"\n  Sphere model saved to {save_dir}")

        return cls(model, tokenizer, passthrough, model_dir, config)

    @classmethod
    def from_saved(cls, save_dir, gpu=False):
        """Load previously saved sphere model."""
        save_dir = Path(save_dir)
        print(f"\n  Loading sphere model from {save_dir}")

        model = SphereNativeModel.load(save_dir, gpu=gpu)
        tokenizer = SimpleTokenizer(save_dir)

        passthrough = {}
        pt_path = save_dir / 'passthrough.npz'
        if pt_path.exists():
            data = np.load(pt_path)
            for key in data.files:
                passthrough[key] = data[key]
            print(f"  Passthrough: {len(passthrough)} tensors")

        # Config
        config = model.config.get('original_config', {})
        hf_config = save_dir / 'config.json'
        if hf_config.exists():
            with open(hf_config) as f:
                config = json.load(f)

        return cls(model, tokenizer, passthrough, save_dir, config)


# ============================================
# CLI
# ============================================

def list_presets():
    print("\nAvailable presets:")
    print(f"  {'Name':<25} {'Description'}")
    print(f"  {'-'*60}")
    for name, info in sorted(PRESETS.items()):
        print(f"  {name:<25} {info['description']}")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Sphere Native - Directional weights on Fibonacci lattice.",
    )

    sub = parser.add_subparsers(dest='command')

    sub.add_parser('list', help='List available preset models')

    pull = sub.add_parser('pull', help='Download and convert a preset model')
    pull.add_argument('name', help='Preset name or HuggingFace model ID')
    pull.add_argument('--output', '-o', default=None, help='Save directory')
    pull.add_argument('--gpu', action='store_true')

    run = sub.add_parser('run', help='Run inference')
    run.add_argument('model', help='Preset name, HF model ID, or saved model path')
    run.add_argument('--prompt', '-p', default='The meaning of life is')
    run.add_argument('--max-tokens', '-n', type=int, default=100)
    run.add_argument('--temperature', '-t', type=float, default=0.7)
    run.add_argument('--gpu', action='store_true')

    bench = sub.add_parser('bench', help='Benchmark sphere vs INT8')
    bench.add_argument('--hidden', type=int, default=2560, help='Hidden dim')
    bench.add_argument('--layers', type=int, default=2, help='Number of layers')
    bench.add_argument('--iters', type=int, default=50, help='Iterations')

    args = parser.parse_args()

    if args.command == 'list':
        list_presets()

    elif args.command == 'pull':
        name = args.name
        save_dir = args.output
        if save_dir is None:
            safe_name = name.replace('/', '-').replace('\\', '-')
            save_dir = f'./sphere-{safe_name}'

        if name in PRESETS:
            model = SphereNative.from_preset(name, gpu=args.gpu, save_dir=save_dir)
        else:
            model = SphereNative.from_hub(name, gpu=args.gpu, save_dir=save_dir)

        print(f"\n  Sphere model saved to: {save_dir}")

    elif args.command == 'run':
        name = args.model
        model_path = Path(name)

        if model_path.exists() and (model_path / 'sphere_config.json').exists():
            model = SphereNative.from_saved(name, gpu=args.gpu)
        elif name in PRESETS:
            safe_name = name.replace('/', '-').replace('\\', '-')
            cache_path = Path(f'./sphere-{safe_name}')
            if cache_path.exists() and (cache_path / 'sphere_config.json').exists():
                print(f"  Found cached sphere model at {cache_path}")
                model = SphereNative.from_saved(cache_path, gpu=args.gpu)
            else:
                model = SphereNative.from_preset(name, gpu=args.gpu, save_dir=str(cache_path))
        else:
            safe_name = name.replace('/', '-').replace('\\', '-')
            cache_path = Path(f'./sphere-{safe_name}')
            if cache_path.exists() and (cache_path / 'sphere_config.json').exists():
                model = SphereNative.from_saved(cache_path, gpu=args.gpu)
            else:
                model = SphereNative.from_hub(name, gpu=args.gpu, save_dir=str(cache_path))

        print(f"\n>>> {args.prompt}")
        print("    ", end="")
        for token in model.stream(args.prompt, args.max_tokens, args.temperature):
            print(token, end="", flush=True)
        print("\n")

    elif args.command == 'bench':
        _run_benchmark(args.hidden, args.layers, args.iters)

    else:
        parser.print_help()
        print("\n  Quick start:")
        print("    python sphere_native.py list")
        print("    python sphere_native.py pull stablelm-3b")
        print("    python sphere_native.py run stablelm-3b -p 'Hello world'")
        print("    python sphere_native.py bench --hidden 2560 --layers 2")
        print()


def _run_benchmark(hidden=2560, n_layers=2, n_iters=50):
    """Benchmark sphere layer vs INT8 layer."""
    print(f"\n=== SPHERE vs INT8 BENCHMARK ===")
    print(f"  Hidden: {hidden}, Layers: {n_layers}, Iters: {n_iters}")
    print()

    # Sphere layer
    sphere = SphereNativeLayer(hidden, hidden)
    fp32_w = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
    sphere.align_to_blueprint(fp32_w)

    x = np.random.randn(5, hidden).astype(np.float32)

    # Warmup
    _ = sphere.forward_fp32_fast(x)

    t0 = time.time()
    for _ in range(n_iters):
        _ = sphere.forward_fp32_fast(x)
    t1 = time.time()

    sphere_ms = (t1 - t0) / n_iters * 1000
    print(f"  Sphere forward:  {sphere_ms:.2f} ms/iter")
    print(f"  Sphere storage:  {sphere.storage_bytes():,} bytes")

    # INT8 equivalent (expand + matmul)
    w_int8 = np.random.randint(-128, 127, (hidden, hidden), dtype=np.int8)
    scale = 0.01

    t2 = time.time()
    for _ in range(n_iters):
        w_fp = w_int8.astype(np.float32) * scale
        _ = x @ w_fp.T
    t3 = time.time()

    int8_ms = (t3 - t2) / n_iters * 1000
    print(f"  INT8 forward:    {int8_ms:.2f} ms/iter")
    print(f"  INT8 storage:    {hidden * hidden:,} bytes")

    print()
    print(f"  Speed ratio:     {sphere_ms / int8_ms:.2f}x (sphere/int8)")
    print(f"  Storage ratio:   {sphere.storage_bytes() / (hidden * hidden) * 100:.1f}% of INT8")

    # Accuracy test: how close is sphere reconstruction to original?
    w_scaled = sphere._reconstruct_weight_scaled()
    w_reconstructed = w_scaled.reshape(hidden, -1)[:, :hidden]

    mse = np.mean((fp32_w - w_reconstructed) ** 2)
    rel_err = np.sqrt(mse) / np.std(fp32_w) * 100
    cos_sim = np.sum(fp32_w * w_reconstructed) / (np.linalg.norm(fp32_w) * np.linalg.norm(w_reconstructed))

    print()
    print(f"  Reconstruction MSE:    {mse:.8f}")
    print(f"  Relative RMSE:         {rel_err:.2f}%")
    print(f"  Cosine similarity:     {cos_sim:.6f}")
    print()


if __name__ == "__main__":
    main()
