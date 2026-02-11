#!/usr/bin/env python3
"""
Molecular Native - Bonded Weight Networks

Atoms don't exist alone. They bond.
- Covalent: share electrons (weights reinforce)
- Antibonding: repel (weights oppose)
- Ionic: transfer (one weight dominates)

Each weight triplet is an "atom".
Bonds connect adjacent atoms with signed strength.

    out = Σ atom_contribution + Σ bond_contribution

    bond_contribution = bond_strength * (atom_i · atom_j)

    positive bond: atoms reinforce when aligned
    negative bond: atoms cancel when aligned (antibonding)

Storage:
    - atom_idx: uint8 (atomic state, 256 options)
    - atom_mag: uint8 (magnitude)
    - bond_strength: int8 (-128 to +127, signed!)

    3 bytes per atom-bond pair vs 3 bytes for INT8 triplet
    Same storage, but with STRUCTURE.
"""

# GPU-native backend
try:
    from backend import cp, GPU, ensure_cpu, ensure_gpu, sync, PI, get_xp
except ImportError:
    import numpy as cp
    GPU = False
    PI = 3.141592653589793
    def ensure_cpu(x): return x
    def ensure_gpu(x): return x
    def sync(): pass
    def get_xp(arr=None): return cp


# ============================================
# ATOMIC BASIS (reuse from atomic_native)
# ============================================

def build_p_orbital_basis():
    """px, py, pz - the natural 3D basis."""
    return cp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=cp.float32)


P_ORBITALS = build_p_orbital_basis()


def build_atomic_directions(n=256):
    """
    Build direction lookup table from p-orbital combinations.
    Simplified: Fibonacci sphere for uniform coverage.
    """
    golden = (1 + cp.sqrt(5)) / 2
    indices = cp.arange(n, dtype=cp.float32)

    theta = 2 * PI * indices / golden
    phi = cp.arccos(1 - 2 * (indices + 0.5) / n)

    directions = cp.stack([
        cp.sin(phi) * cp.cos(theta),
        cp.sin(phi) * cp.sin(theta),
        cp.cos(phi),
    ], axis=1).astype(cp.float32)

    return directions


ATOM_DIRS = build_atomic_directions(256)


# ============================================
# BOND TYPES
# ============================================

# Bond character from strength
# Strong positive: covalent (sharing, reinforcement)
# Weak positive: van der Waals (weak attraction)
# Zero: non-bonded (independent)
# Weak negative: steric repulsion (mild opposition)
# Strong negative: antibonding (destructive interference)

def bond_character(strength):
    """Classify bond type from int8 strength."""
    if strength > 64:
        return "covalent"
    elif strength > 16:
        return "polar"
    elif strength > -16:
        return "non-bonded"
    elif strength > -64:
        return "repulsive"
    else:
        return "antibonding"


# ============================================
# MOLECULAR LAYER
# ============================================

class MolecularLayer:
    """
    Weights as bonded atoms.

    Each weight triplet = atom with:
        - atomic_idx: direction on unit sphere (uint8)
        - atomic_mag: magnitude (uint8)

    Between adjacent atoms:
        - bond_strength: int8 (-128 to +127)

    Forward pass:
        1. Compute atom contributions (like atomic_native)
        2. Compute bond contributions (atom_i · atom_j * bond_strength)
        3. Sum both

    Bonds create NON-LOCAL effects: atom_i's output depends on atom_j.
    """

    def __init__(self, in_features, out_features, has_bias=False):
        self.in_features = in_features
        self.out_features = out_features

        self.n_groups = (in_features + 2) // 3
        self.padded_in = self.n_groups * 3

        # Atomic state
        self.atomic_idx = cp.zeros((out_features, self.n_groups), dtype=cp.uint8)
        self.atomic_mag = cp.zeros((out_features, self.n_groups), dtype=cp.uint8)

        # Bonds: between adjacent atoms in each row
        # n_groups atoms → n_groups-1 bonds per row
        self.n_bonds = max(self.n_groups - 1, 0)
        self.bond_strength = cp.zeros((out_features, self.n_bonds), dtype=cp.int8)

        # Row scale
        self.row_scale = cp.ones(out_features, dtype=cp.float32)

        self.bias = cp.zeros(out_features, dtype=cp.float32) if has_bias else None

        # Precomputed
        self._w_fused = None

    def align_to_blueprint(self, fp32_weight, fp32_bias=None):
        """
        Distill FP32 weights to molecular encoding.

        1. Encode each triplet as atom (direction + magnitude)
        2. Compute bonds from correlation between adjacent triplets
           - High positive correlation → covalent bond
           - High negative correlation → antibonding
           - Low correlation → non-bonded
        """
        out_f, in_f = fp32_weight.shape

        if in_f % 3 != 0:
            pad_width = 3 - (in_f % 3)
            fp32_weight = cp.pad(fp32_weight, ((0, 0), (0, pad_width)), mode='constant')

        # Row normalization
        row_max = cp.abs(fp32_weight).max(axis=1, keepdims=True)
        row_max = cp.maximum(row_max, cp.float32(1e-10))
        self.row_scale = row_max.flatten()

        w_norm = fp32_weight / row_max
        w_trips = w_norm.reshape(out_f, -1, 3)

        # === ATOMS ===
        mags = cp.linalg.norm(w_trips, axis=-1)
        safe_mags = cp.maximum(mags, 1e-10)[..., cp.newaxis]
        dirs = w_trips / safe_mags

        # Find nearest atomic direction
        flat_dirs = dirs.reshape(-1, 3)
        dots = flat_dirs @ ATOM_DIRS.T
        self.atomic_idx = cp.argmax(dots, axis=1).astype(cp.uint8).reshape(out_f, self.n_groups)

        # Magnitude quantization
        mag_max = cp.maximum(mags.max(axis=1, keepdims=True), 1e-10)
        self.atomic_mag = cp.clip(cp.round(mags / mag_max * 255), 0, 255).astype(cp.uint8)
        self._mag_scale = mag_max.flatten()

        # === BONDS ===
        # Correlation between adjacent triplets determines bond strength
        # dot(triplet_i, triplet_j) → positive = aligned, negative = opposed
        if self.n_bonds > 0:
            for i in range(self.n_bonds):
                # Correlation: dot product of adjacent triplet directions
                # Weighted by geometric mean of their magnitudes
                corr = cp.sum(dirs[:, i] * dirs[:, i+1], axis=-1)  # (out_f,)
                mag_weight = cp.sqrt(mags[:, i] * mags[:, i+1])

                # Scale to int8 range: corr in [-1, 1], mag_weight in [0, ~1.7]
                # Combined: corr * mag_weight in [-1.7, 1.7]
                bond_raw = corr * mag_weight * 75  # scale to use int8 range
                self.bond_strength[:, i] = cp.clip(cp.round(bond_raw), -128, 127).astype(cp.int8)

        if fp32_bias is not None and self.bias is not None:
            self.bias = fp32_bias.astype(cp.float32)

        self._precompute()

    def _precompute(self):
        """
        Precompute fused weight matrix.

        The bond contribution can be folded into the weight matrix!

        For atom i with bond to atom i+1:
            effective_weight[i] += bond_strength * atom_dir[i+1]

        This way forward pass stays as single matmul.
        """
        # Dequantize atomic magnitudes
        mag_fp = self.atomic_mag.astype(cp.float32) / 255.0
        if hasattr(self, '_mag_scale'):
            mag_fp = mag_fp * self._mag_scale[:, cp.newaxis]
        mag_fp = mag_fp * self.row_scale[:, cp.newaxis]

        # Atomic directions
        w_dir = ATOM_DIRS[self.atomic_idx]  # (out, n_groups, 3)

        # Base weights from atoms
        w_fused = w_dir * mag_fp[..., cp.newaxis]  # (out, n_groups, 3)

        # Add bond contributions
        # Bond between atom i and i+1 adds cross-term to both
        if self.n_bonds > 0:
            bond_fp = self.bond_strength.astype(cp.float32) / 127.0  # normalize to [-1, 1]

            for i in range(self.n_bonds):
                # Bond affects how atom i responds to atom i+1's direction
                # Positive bond: atom i also responds to i+1's input
                # Negative bond: atom i opposes i+1's input

                bond_weight = bond_fp[:, i:i+1, cp.newaxis] * 0.25  # (out, 1, 1), scaled down

                # Add neighbor's direction influence, weighted by bond
                # This creates non-local receptive field
                w_fused[:, i, :] += bond_weight[:, 0, 0:1] * w_dir[:, i+1, :]
                w_fused[:, i+1, :] += bond_weight[:, 0, 0:1] * w_dir[:, i, :]

        # Apply 2/π normalization
        w_fused = w_fused * cp.float32(2.0 / PI)

        self._w_fused = w_fused.reshape(self.out_features, -1).astype(cp.float32)

    def forward(self, x_fp32):
        """Forward pass: single matmul (bonds folded into weights)."""
        seq_len = x_fp32.shape[0]
        in_f = x_fp32.shape[1]

        if in_f < self.padded_in:
            x_fp32 = cp.pad(x_fp32, ((0, 0), (0, self.padded_in - in_f)), mode='constant')

        x_flat = x_fp32[:, :self.padded_in]
        out = x_flat @ self._w_fused.T

        if self.bias is not None:
            out = out + self.bias

        return out.astype(cp.float32)

    def storage_bytes(self):
        """
        Storage:
            atomic_idx: 1 byte per group
            atomic_mag: 1 byte per group
            bond_strength: 1 byte per bond (n_groups - 1)

        Total: ~3 bytes per group = same as INT8 triplet
        But with structure!
        """
        total = self.atomic_idx.nbytes + self.atomic_mag.nbytes
        total += self.bond_strength.nbytes
        total += self.row_scale.nbytes
        if self.bias is not None:
            total += self.bias.nbytes
        return total

    def equivalent_int8_bytes(self):
        return self.out_features * self.in_features

    def bond_statistics(self):
        """Analyze bond distribution."""
        if self.n_bonds == 0:
            return {}

        flat_bonds = self.bond_strength.flatten()

        covalent = cp.sum(flat_bonds > 64)
        polar = cp.sum((flat_bonds > 16) & (flat_bonds <= 64))
        nonbonded = cp.sum((flat_bonds >= -16) & (flat_bonds <= 16))
        repulsive = cp.sum((flat_bonds < -16) & (flat_bonds >= -64))
        antibonding = cp.sum(flat_bonds < -64)

        total = len(flat_bonds)

        return {
            'covalent': (covalent, covalent/total*100),
            'polar': (polar, polar/total*100),
            'non-bonded': (nonbonded, nonbonded/total*100),
            'repulsive': (repulsive, repulsive/total*100),
            'antibonding': (antibonding, antibonding/total*100),
            'mean_strength': float(flat_bonds.mean()),
            'std_strength': float(flat_bonds.std()),
        }


# ============================================
# TESTS
# ============================================

if __name__ == "__main__":
    import time

    print("=== ATOMIC DIRECTIONS ===")
    print(f"Shape: {ATOM_DIRS.shape}")
    print(f"Unit vectors: {cp.allclose(cp.linalg.norm(ATOM_DIRS, axis=1), 1.0)}")
    print()

    print("=== BOND CHARACTERS ===")
    for s in [100, 50, 0, -50, -100]:
        print(f"  strength {s:>4}: {bond_character(s)}")
    print()

    print("=== MOLECULAR LAYER (256x256) ===")
    layer = MolecularLayer(256, 256)
    fp32_w = cp.random.randn(256, 256).astype(cp.float32) * 0.02
    layer.align_to_blueprint(fp32_w)

    print(f"atomic_idx: {layer.atomic_idx.shape} max={layer.atomic_idx.max()}")
    print(f"atomic_mag: {layer.atomic_mag.shape} max={layer.atomic_mag.max()}")
    print(f"bond_strength: {layer.bond_strength.shape} range=[{layer.bond_strength.min()}, {layer.bond_strength.max()}]")

    sb = layer.storage_bytes()
    ib = layer.equivalent_int8_bytes()
    print(f"Storage: {sb:,} bytes vs INT8 {ib:,} = {sb/ib*100:.1f}%")
    print()

    print("=== BOND STATISTICS ===")
    stats = layer.bond_statistics()
    for bond_type in ['covalent', 'polar', 'non-bonded', 'repulsive', 'antibonding']:
        count, pct = stats[bond_type]
        print(f"  {bond_type:>12}: {count:>6} ({pct:>5.1f}%)")
    print(f"  mean strength: {stats['mean_strength']:>+.1f}")
    print(f"  std strength:  {stats['std_strength']:>.1f}")
    print()

    print("=== FORWARD PASS ===")
    x = cp.random.randn(10, 256).astype(cp.float32)

    t0 = time.time()
    y = layer.forward(x)
    t1 = time.time()

    print(f"Output: {y.shape}")
    print(f"Range:  [{y.min():.4f}, {y.max():.4f}]")
    print(f"NaN:    {cp.any(cp.isnan(y))}")
    print(f"Time:   {(t1-t0)*1000:.2f}ms")
    print()

    print("=== BOND INFLUENCE TEST ===")
    # Compare with vs without bonds
    layer_bonded = MolecularLayer(256, 256)
    layer_bonded.align_to_blueprint(fp32_w)

    # Zero out bonds
    layer_nobond = MolecularLayer(256, 256)
    layer_nobond.align_to_blueprint(fp32_w)
    layer_nobond.bond_strength[:] = 0
    layer_nobond._precompute()

    y_bonded = layer_bonded.forward(x)
    y_nobond = layer_nobond.forward(x)

    diff = cp.abs(y_bonded - y_nobond)
    print(f"Bonded vs non-bonded:")
    print(f"  max diff:  {diff.max():.4f}")
    print(f"  mean diff: {diff.mean():.4f}")
    print(f"  % changed: {(diff > 1e-6).mean()*100:.1f}%")
    print()

    print("=== CORRELATED WEIGHTS → STRONG BONDS ===")
    # Create weights where adjacent triplets are correlated
    w_corr = cp.random.randn(256, 258).astype(cp.float32) * 0.02  # pad to multiple of 3
    # Make adjacent triplets similar
    for i in range(0, 258-6, 3):
        w_corr[:, i+3:i+6] = w_corr[:, i:i+3] * 0.8 + cp.random.randn(256, 3).astype(cp.float32) * 0.004
    w_corr = w_corr[:, :256]  # trim back

    layer_corr = MolecularLayer(256, 256)
    layer_corr.align_to_blueprint(w_corr)

    stats_corr = layer_corr.bond_statistics()
    print(f"Correlated weights:")
    print(f"  covalent:    {stats_corr['covalent'][1]:.1f}%")
    print(f"  antibonding: {stats_corr['antibonding'][1]:.1f}%")
    print(f"  mean strength: {stats_corr['mean_strength']:+.1f}")
    print()

    print("=== ANTICORRELATED WEIGHTS → ANTIBONDS ===")
    # Make adjacent triplets negatively correlated
    w_anti = cp.random.randn(256, 258).astype(cp.float32) * 0.02
    for i in range(0, 258-6, 3):
        w_anti[:, i+3:i+6] = -w_anti[:, i:i+3] * 0.8 + cp.random.randn(256, 3).astype(cp.float32) * 0.004
    w_anti = w_anti[:, :256]

    layer_anti = MolecularLayer(256, 256)
    layer_anti.align_to_blueprint(w_anti)

    stats_anti = layer_anti.bond_statistics()
    print(f"Anticorrelated weights:")
    print(f"  covalent:    {stats_anti['covalent'][1]:.1f}%")
    print(f"  antibonding: {stats_anti['antibonding'][1]:.1f}%")
    print(f"  mean strength: {stats_anti['mean_strength']:+.1f}")
    print()

    print("Bonds emerge from weight structure.")
    print("Correlation → attraction. Opposition → repulsion.")
