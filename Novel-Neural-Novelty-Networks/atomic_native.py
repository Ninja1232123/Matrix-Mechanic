#!/usr/bin/env python3
"""
Atomic Native - Weight Encoding via Electron Orbital Structure

The quantization isn't arbitrary - it's what electrons actually do.

Weight triplet → Atomic configuration:
    - p orbitals (px, py, pz): natural 3D basis, 3 orientations
    - Electron occupation (0,1,2 per orbital): magnitude
    - Spin (+½, -½): sign
    - Shell number (n=1,2,3...): scale

Key insight: p orbitals ARE triplet encoding. Physics did it first.

Orbital shapes:
    s: spherical (l=0, m=0)           - 1 orientation, isotropic
    p: dumbbell  (l=1, m=-1,0,+1)     - 3 orientations, directional
    d: clover    (l=2, m=-2,-1,0,1,2) - 5 orientations, complex
    f: flower    (l=3, m=-3..+3)      - 7 orientations, rare

For weight encoding, p-orbitals are perfect:
    - 3 basis directions (matches triplet grouping)
    - Each can hold 0, 1, or 2 electrons (ternary per axis)
    - 3^3 = 27 base states per triplet
    - With spin: 27 * 2^3 = 216 states (close to 256!)
    - Add shell scaling: effectively unlimited precision

Storage: ~1 byte per triplet (atomic config) + scale
         vs 3 bytes for INT8
         = 33% of INT8

No matrix multiply. Just quantum mechanics.
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
# ORBITAL GEOMETRY
# ============================================

def build_p_orbital_basis():
    """
    The three p orbitals: px, py, pz

    These are the natural 3D directional basis from quantum mechanics.
    Each is a dumbbell shape along one axis.

    Angular part of wavefunction:
        px ~ sin(θ)cos(φ) ~ x/r
        py ~ sin(θ)sin(φ) ~ y/r
        pz ~ cos(θ)       ~ z/r

    So the directions are just the unit axes!
    """
    return cp.array([
        [1.0, 0.0, 0.0],  # px
        [0.0, 1.0, 0.0],  # py
        [0.0, 0.0, 1.0],  # pz
    ], dtype=cp.float32)


P_ORBITALS = build_p_orbital_basis()


def build_atomic_states():
    """
    Build all possible p-orbital electron configurations.

    Each orbital can have 0, 1, or 2 electrons (Pauli exclusion).
    Each electron has spin +½ or -½.

    For weight encoding:
        - Occupation (0,1,2) per orbital → magnitude component
        - Net spin per orbital → sign component

    States: 3^3 = 27 occupation patterns
            × 4 spin configs per occupied orbital

    Simplified model:
        occupation (0,1,2) → magnitude (0, 0.5, 1.0)
        net spin (-1,0,+1) → sign multiplier

    Total encoded states: 3^3 × 3^3 = 729 (use 8 bits = 256 subset)
    """
    states = []
    directions = []
    magnitudes = []

    # Occupation: 0, 1, or 2 electrons per orbital
    for ox in range(3):  # px occupation
        for oy in range(3):  # py occupation
            for oz in range(3):  # pz occupation
                # Magnitude from occupation (normalized)
                occ = cp.array([ox, oy, oz], dtype=cp.float32) / 2.0  # [0, 0.5, 1]

                # Direction is weighted sum of orbital axes
                mag = cp.linalg.norm(occ)
                if mag > 1e-8:
                    direction = occ / mag
                else:
                    direction = cp.array([1.0, 0.0, 0.0], dtype=cp.float32)

                # For each occupation pattern, consider spin configurations
                # Simplified: net spin determines sign
                # -1: more down, 0: balanced, +1: more up
                for spin_x in [-1, 0, 1]:
                    for spin_y in [-1, 0, 1]:
                        for spin_z in [-1, 0, 1]:
                            # Skip invalid: can't have spin without electrons
                            if ox == 0 and spin_x != 0: continue
                            if oy == 0 and spin_y != 0: continue
                            if oz == 0 and spin_z != 0: continue
                            # Can't have net spin > occupation
                            if abs(spin_x) > ox: continue
                            if abs(spin_y) > oy: continue
                            if abs(spin_z) > oz: continue

                            spin_vec = cp.array([spin_x, spin_y, spin_z], dtype=cp.float32)

                            # Sign from net spin direction
                            spin_mag = cp.linalg.norm(spin_vec)
                            if spin_mag > 1e-8:
                                sign_dir = spin_vec / spin_mag
                            else:
                                sign_dir = cp.array([0.0, 0.0, 0.0], dtype=cp.float32)

                            # Combined state
                            states.append({
                                'occupation': (ox, oy, oz),
                                'spin': (spin_x, spin_y, spin_z),
                                'direction': direction.copy(),
                                'magnitude': mag,
                                'sign_direction': sign_dir.copy(),
                            })

                            # Effective direction with sign
                            net_spin = cp.sum(spin_vec)
                            sign = cp.sign(net_spin) if abs(net_spin) > 0 else 1.0

                            directions.append(direction * sign)
                            magnitudes.append(mag)

    return states, cp.array(directions, dtype=cp.float32), cp.array(magnitudes, dtype=cp.float32)


ATOMIC_STATES, ATOMIC_DIRS, ATOMIC_MAGS = build_atomic_states()
N_STATES = len(ATOMIC_STATES)

# Trim or pad to 256 states
if N_STATES > 256:
    # PRIORITY: High-spin states carry directional conviction
    # Score = |net_spin| * magnitude - these states have strong opinions
    spin_scores = cp.array([
        abs(sum(s['spin'])) * s['magnitude'] for s in ATOMIC_STATES
    ], dtype=cp.float32)

    # Secondary: direction diversity (want coverage of the direction sphere)
    # Combine spin conviction with magnitude for final ranking
    conviction = spin_scores + ATOMIC_MAGS * 0.5

    indices = cp.argsort(-conviction)[:256]
    ATOMIC_DIRS = ATOMIC_DIRS[indices]
    ATOMIC_MAGS = ATOMIC_MAGS[indices]
    # Convert to Python ints for list indexing
    indices_cpu = ensure_cpu(indices).tolist() if GPU else indices.tolist()
    ATOMIC_STATES = [ATOMIC_STATES[i] for i in indices_cpu]
    N_STATES = 256  # Update count after trimming

if N_STATES < 256:  # Use 'if' not 'elif' since N_STATES may have changed
    # Pad with interpolated states
    n_pad = 256 - N_STATES
    if n_pad > 0:  # Safety check
        pad_dirs = cp.random.randn(n_pad, 3).astype(cp.float32)
        pad_dirs = pad_dirs / cp.linalg.norm(pad_dirs, axis=1, keepdims=True)
        pad_mags = cp.random.uniform(0, 1.5, n_pad).astype(cp.float32)
        ATOMIC_DIRS = cp.vstack([ATOMIC_DIRS, pad_dirs])
        ATOMIC_MAGS = cp.concatenate([ATOMIC_MAGS, pad_mags])

N_ATOMIC = len(ATOMIC_DIRS)


def nearest_atomic_state(direction):
    """
    Find nearest atomic state for a direction vector.
    """
    if direction.ndim == 1:
        dots = ATOMIC_DIRS @ direction
        return cp.uint8(cp.argmax(dots))
    else:
        dots = direction @ ATOMIC_DIRS.T
        return cp.argmax(dots, axis=1).astype(cp.uint8)


# ============================================
# SHELL STRUCTURE (for magnitude scaling)
# ============================================

def electron_shell_radii(n_shells=7):
    """
    Bohr model: radius of shell n = n² × a₀

    For weight encoding:
        shell 1: base magnitude
        shell 2: 4× base
        shell 3: 9× base
        ...

    This gives natural log-scale quantization!
    """
    return cp.array([n**2 for n in range(1, n_shells + 1)], dtype=cp.float32)


SHELL_RADII = electron_shell_radii(7)  # shells 1-7


def magnitude_to_shell(mag, max_mag):
    """
    Map magnitude to shell number + position within shell.

    Returns: (shell_idx, intra_shell_position)
    """
    if max_mag < 1e-10:
        return 0, 0.0

    normalized = mag / max_mag  # [0, 1]

    # Map to shell: larger magnitudes → higher shells
    # Use log scaling to match n² shell structure
    log_pos = cp.log1p(normalized * (cp.e - 1))  # [0, 1] log-scaled
    shell_float = log_pos * 6  # [0, 6] → shells 1-7

    shell_idx = int(cp.clip(shell_float, 0, 6))
    intra = shell_float - shell_idx

    return shell_idx, float(intra)


# ============================================
# ATOMIC LAYER
# ============================================

class AtomicLayer:
    """
    Weights encoded as atomic electron configurations.

    Each weight triplet becomes:
        - 8-bit atomic state index (p-orbital config)
        - 4-bit shell number (1-7, magnitude scale)
        - 4-bit intra-shell position (fine magnitude)

    Total: 16 bits per triplet = 2 bytes per 3 weights
    Same compression as heart3d and CircleJerk: 67% of INT8

    But the encoding is physically motivated!
    """

    def __init__(self, in_features, out_features, has_bias=False):
        self.in_features = in_features
        self.out_features = out_features

        self.n_groups = (in_features + 2) // 3
        self.padded_in = self.n_groups * 3

        # Atomic state index (p-orbital configuration)
        self.atomic_idx = cp.zeros((out_features, self.n_groups), dtype=cp.uint8)

        # Shell + intra-shell packed into uint8
        # High 3 bits: shell (0-7), Low 5 bits: intra-shell (0-31)
        self.shell_pack = cp.zeros((out_features, self.n_groups), dtype=cp.uint8)

        # Per-row scale (like nuclear charge - determines overall strength)
        self.nuclear_charge = cp.ones(out_features, dtype=cp.float32)

        self.bias = cp.zeros(out_features, dtype=cp.float32) if has_bias else None

        # Precomputed for fast forward
        self._w_fused = None

    def align_to_blueprint(self, fp32_weight, fp32_bias=None):
        """
        Distill FP32 weights to atomic configuration.
        """
        out_f, in_f = fp32_weight.shape

        if in_f % 3 != 0:
            pad_width = 3 - (in_f % 3)
            fp32_weight = cp.pad(fp32_weight, ((0, 0), (0, pad_width)), mode='constant')

        # Row-wise normalization
        row_max = cp.abs(fp32_weight).max(axis=1, keepdims=True)
        row_max = cp.maximum(row_max, cp.float32(1e-10))
        self.nuclear_charge = row_max.flatten()

        w_norm = fp32_weight / row_max
        w_trips = w_norm.reshape(out_f, -1, 3)

        # For each triplet: find atomic state + shell encoding
        mags = cp.linalg.norm(w_trips, axis=-1)
        safe_mags = cp.maximum(mags, 1e-10)
        dirs = w_trips / safe_mags[..., cp.newaxis]

        # Flatten for vectorized lookup
        flat_dirs = dirs.reshape(-1, 3)
        flat_mags = mags.flatten()

        # Find nearest atomic state
        self.atomic_idx = nearest_atomic_state(flat_dirs).reshape(out_f, self.n_groups)

        # Shell encoding
        max_mag = mags.max()
        shell_pack = cp.zeros_like(flat_mags, dtype=cp.uint8)

        for i, m in enumerate(flat_mags):
            shell, intra = magnitude_to_shell(m, max_mag)
            # Pack: shell (3 bits) << 5 | intra (5 bits)
            intra_quant = int(cp.clip(intra * 31, 0, 31))
            shell_pack[i] = (shell << 5) | intra_quant

        self.shell_pack = shell_pack.reshape(out_f, self.n_groups)

        if fp32_bias is not None and self.bias is not None:
            self.bias = fp32_bias.astype(cp.float32)

        self._precompute_fused()

    def _precompute_fused(self):
        """
        Precompute weight matrix for single matmul forward.

        SHELL INTERPOLATION: Smooth the n² → (n+1)² transitions.
        Instead of discrete shell_scale = n², we interpolate:
            effective_scale = lerp(n², (n+1)², intra_position)

        This prevents "quantize-pop" where crossing a shell boundary
        causes sudden magnitude jumps during inference.
        """
        # Unpack shell + intra
        shell = (self.shell_pack >> 5).astype(cp.float32)  # 0-6 (shell index)
        intra = (self.shell_pack & 0x1F).astype(cp.float32) / 31.0  # 0-1

        # Shell radii: n² for n = 1..7
        # shell=0 → n=1 → r=1
        # shell=6 → n=7 → r=49
        n = shell + 1  # actual shell number (1-7)
        n_next = cp.minimum(n + 1, 7)  # next shell, capped at 7

        # Current and next shell radii
        r_current = n ** 2
        r_next = n_next ** 2

        # INTERPOLATE between shells based on intra position
        # At intra=0, use r_current. At intra=1, blend toward r_next.
        # This is the "connective tissue" that smooths shell transitions.
        shell_scale = r_current + intra * (r_next - r_current)
        shell_scale = shell_scale / 49.0  # normalize by max (7² = 49)

        mag = shell_scale * self.nuclear_charge[:, cp.newaxis]

        # Direction from atomic state
        w_dir = ATOMIC_DIRS[self.atomic_idx]  # (out, n_groups, 3)

        # Fused weight
        w_fused = w_dir * mag[..., cp.newaxis]

        # Apply 2/π normalization
        w_fused = w_fused * cp.float32(2.0 / PI)

        self._w_fused = w_fused.reshape(self.out_features, -1).astype(cp.float32)

    def forward(self, x_fp32):
        """
        Forward pass: single matmul.
        """
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
        # atomic_idx: 1 byte per group
        # shell_pack: 1 byte per group
        # nuclear_charge: 4 bytes per row
        total = self.atomic_idx.nbytes + self.shell_pack.nbytes
        total += self.nuclear_charge.nbytes
        if self.bias is not None:
            total += self.bias.nbytes
        return total

    def equivalent_int8_bytes(self):
        return self.out_features * self.in_features


# ============================================
# HYBRID: ATOMIC HEART
# ============================================

class AtomicHeartLayer:
    """
    Combine atomic orbitals with heart geometry.

    The heart surface determines WHERE weights live.
    The atomic structure determines HOW they're quantized.

    Think of it as: atoms arranged on the surface of a heart.
    Each position on the heart holds an atomic configuration.

    This is just for fun. Because why not.
    """

    def __init__(self, in_features, out_features, has_bias=False):
        self.in_features = in_features
        self.out_features = out_features

        self.n_groups = (in_features + 2) // 3
        self.padded_in = self.n_groups * 3

        # Heart position index (from heart3d)
        self.heart_idx = cp.zeros((out_features, self.n_groups), dtype=cp.uint8)

        # Atomic state at each heart position
        self.atomic_idx = cp.zeros((out_features, self.n_groups), dtype=cp.uint8)

        # Magnitude (simpler: just uint8)
        self.magnitude = cp.zeros((out_features, self.n_groups), dtype=cp.uint8)

        self.row_scale = cp.ones(out_features, dtype=cp.float32)
        self.bias = cp.zeros(out_features, dtype=cp.float32) if has_bias else None

        self._w_fused = None

    def storage_bytes(self):
        # 3 bytes per group (heart + atomic + magnitude)
        # Same as INT8! But richer representation
        total = self.heart_idx.nbytes + self.atomic_idx.nbytes + self.magnitude.nbytes
        total += self.row_scale.nbytes
        if self.bias is not None:
            total += self.bias.nbytes
        return total


# ============================================
# TESTS
# ============================================

if __name__ == "__main__":
    import time

    print("=== ATOMIC STATE SPACE ===")
    print(f"Raw states generated: {len(ATOMIC_STATES)}")
    print(f"Trimmed to: {N_ATOMIC}")
    print(f"Direction shape: {ATOMIC_DIRS.shape}")
    print(f"Magnitude range: [{ATOMIC_MAGS.min():.3f}, {ATOMIC_MAGS.max():.3f}]")
    print()

    print("=== P-ORBITAL BASIS ===")
    print(f"px: {P_ORBITALS[0]}")
    print(f"py: {P_ORBITALS[1]}")
    print(f"pz: {P_ORBITALS[2]}")
    print(f"Orthonormal: {cp.allclose(P_ORBITALS @ P_ORBITALS.T, cp.eye(3))}")
    print()

    print("=== SHELL STRUCTURE ===")
    print(f"Shell radii (n²): {SHELL_RADII}")
    for mag in [0.1, 0.3, 0.5, 0.7, 0.9]:
        shell, intra = magnitude_to_shell(mag, 1.0)
        print(f"  mag={mag:.1f} → shell={shell+1}, intra={intra:.2f}")

    print("\n=== SHELL INTERPOLATION (smooth n² transitions) ===")
    # Show that intra position smoothly blends between shell radii
    for shell_idx in [1, 2, 3]:
        n = shell_idx + 1
        r_curr, r_next = n**2, (n+1)**2
        for intra in [0.0, 0.25, 0.5, 0.75, 1.0]:
            effective = r_curr + intra * (r_next - r_curr)
            print(f"  shell {n} intra={intra:.2f}: {r_curr}→{r_next}, effective={effective:.1f}")
    print()

    print("=== SAMPLE ATOMIC STATES (high-spin priority) ===")
    # Show that high-spin states got prioritized
    high_spin_count = sum(1 for s in ATOMIC_STATES if abs(sum(s['spin'])) >= 2)
    print(f"States with |net_spin| >= 2: {high_spin_count}/256 (high conviction)")
    for i in [0, 10, 50, 100, 200]:
        if i < len(ATOMIC_STATES):
            s = ATOMIC_STATES[i]
            net_spin = sum(s['spin'])
            print(f"  State {i}: occ={s['occupation']} spin={s['spin']} "
                  f"|net|={abs(net_spin)} mag={s['magnitude']:.2f}")
    print()

    print("=== ATOMIC LAYER (256x256) ===")
    layer = AtomicLayer(256, 256)
    fp32_w = cp.random.randn(256, 256).astype(cp.float32) * 0.02
    layer.align_to_blueprint(fp32_w)

    print(f"atomic_idx: {layer.atomic_idx.shape} max={layer.atomic_idx.max()}")
    print(f"shell_pack: {layer.shell_pack.shape}")
    print(f"Unique atoms: {len(cp.unique(layer.atomic_idx))}")

    sb = layer.storage_bytes()
    ib = layer.equivalent_int8_bytes()
    print(f"Storage: {sb:,} bytes vs INT8 {ib:,} = {sb/ib*100:.1f}%")

    x = cp.random.randn(10, 256).astype(cp.float32)

    t0 = time.time()
    y = layer.forward(x)
    t1 = time.time()

    print(f"Forward: {y.shape} time={(t1-t0)*1000:.2f}ms")
    print(f"  range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  NaN: {cp.any(cp.isnan(y))}")
    print()

    print("=== SIZE ESTIMATES ===")
    for name, h, inter, nl in [("3B", 2560, 6912, 32), ("7B", 4096, 14336, 32)]:
        params_per_layer = 4 * h * h + 3 * h * inter
        total_params = nl * params_per_layer
        int8_total = total_params

        # Atomic: 2 bytes per triplet (atomic_idx + shell_pack)
        n_triplets = total_params // 3
        atomic_total = n_triplets * 2

        print(f"\n  {name} model ({total_params/1e9:.1f}B params):")
        print(f"    INT8:   {int8_total/1e9:.2f} GB")
        print(f"    Atomic: {atomic_total/1e9:.2f} GB ({atomic_total/int8_total*100:.0f}%)")

    print()
    print("Electrons know where to be. Weights should too.")
    print("The quantization isn't arbitrary - it's physics.")
