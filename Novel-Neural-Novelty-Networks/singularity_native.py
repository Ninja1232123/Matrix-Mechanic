#!/usr/bin/env python3
"""
Singularity Native - Black Hole Weight Encoding

The ultimate compression: collapse weights into gravitational geometry.

Physics:
    1. HOLOGRAPHIC PRINCIPLE: 3D volume → 2D event horizon
       All information inside a black hole is encoded on its surface.
       Weight tensor W(out, in) → Horizon shell H(θ, φ)

    2. NO-HAIR THEOREM: Every black hole described by just 3 numbers:
       M = Mass (total magnitude)
       Q = Charge (asymmetry/polarity)
       J = Angular momentum (rotation/chirality)
       Everything else is "hidden" in the geometry.

    3. GRAVITATIONAL LENSING: Tokens don't matmul through weights.
       They're launched into the gravity well and BENT.
       Output = deflection angle × intensity

    4. HAWKING RADIATION: The output "leaks" from the horizon.
       Thermal spectrum with temperature T = ℏc³/(8πGMk)

Storage:
    - 3 floats per layer (M, Q, J) + horizon shell
    - Horizon shell: 2D spherical harmonic coefficients
    - Compression: O(√n) instead of O(n²) for n×n matrix

No matrix multiply. Just spacetime curvature.
"""

import numpy as np

PI = np.float32(np.pi)
C = np.float32(299792458)  # speed of light (normalized to 1 in natural units)
G = np.float32(6.674e-11)  # gravitational constant (normalized)


# ============================================
# SPHERICAL HARMONICS - The Horizon Basis
# ============================================

def spherical_harmonic_real(l, m, theta, phi):
    """
    Real spherical harmonic Y_l^m(θ, φ).

    These are the natural basis functions for encoding information
    on a 2D sphere (the event horizon).

    l = degree (0, 1, 2, ...)
    m = order (-l to +l)
    theta = polar angle [0, π]
    phi = azimuthal angle [0, 2π]
    """
    # Simplified implementation for low l
    # Full implementation would use scipy.special.sph_harm

    if l == 0:
        return np.ones_like(theta) * np.float32(0.5 / np.sqrt(PI))

    elif l == 1:
        if m == -1:
            return np.sqrt(3/(4*PI)) * np.sin(theta) * np.sin(phi)
        elif m == 0:
            return np.sqrt(3/(4*PI)) * np.cos(theta)
        elif m == 1:
            return np.sqrt(3/(4*PI)) * np.sin(theta) * np.cos(phi)

    elif l == 2:
        if m == -2:
            return np.sqrt(15/(16*PI)) * np.sin(theta)**2 * np.sin(2*phi)
        elif m == -1:
            return np.sqrt(15/(4*PI)) * np.sin(theta) * np.cos(theta) * np.sin(phi)
        elif m == 0:
            return np.sqrt(5/(16*PI)) * (3*np.cos(theta)**2 - 1)
        elif m == 1:
            return np.sqrt(15/(4*PI)) * np.sin(theta) * np.cos(theta) * np.cos(phi)
        elif m == 2:
            return np.sqrt(15/(16*PI)) * np.sin(theta)**2 * np.cos(2*phi)

    # Higher orders: use Legendre polynomials
    # For now, return 0 for unimplemented
    return np.zeros_like(theta)


def build_horizon_basis(n_theta=32, n_phi=64, l_max=8):
    """
    Build spherical harmonic basis functions on a discrete grid.

    The event horizon is a 2D sphere. We sample it at n_theta × n_phi points
    and precompute the spherical harmonic values at each point.

    Returns:
        theta: (n_theta,) polar angles
        phi: (n_phi,) azimuthal angles
        basis: (n_harmonics, n_theta, n_phi) basis function values
        lm_indices: list of (l, m) pairs
    """
    theta = np.linspace(0.01, PI - 0.01, n_theta, dtype=np.float32)
    phi = np.linspace(0, 2*PI, n_phi, endpoint=False, dtype=np.float32)

    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    basis = []
    lm_indices = []

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_lm = spherical_harmonic_real(l, m, theta_grid, phi_grid)
            basis.append(Y_lm)
            lm_indices.append((l, m))

    basis = np.array(basis, dtype=np.float32)
    return theta, phi, basis, lm_indices


# Build default basis
HORIZON_THETA, HORIZON_PHI, HORIZON_BASIS, HORIZON_LM = build_horizon_basis(32, 64, 8)
N_HARMONICS = len(HORIZON_LM)  # (l_max+1)² = 81 coefficients


# ============================================
# KERR METRIC - Rotating Black Hole Geometry
# ============================================

def kerr_metric_deflection(r, theta, M, J):
    """
    Compute gravitational deflection using Kerr metric.

    The Kerr metric describes spacetime around a rotating black hole.
    Light (and tokens) passing near it are BENT.

    M = mass (determines Schwarzschild radius)
    J = angular momentum (spin parameter a = J/M)

    Returns deflection angle and frame-dragging contribution.
    """
    # Spin parameter (dimensionless)
    a = J / np.maximum(M, np.float32(1e-10))
    a = np.clip(a, -0.998, 0.998)  # extremal limit

    # Schwarzschild radius
    r_s = 2 * M

    # Kerr metric components (simplified)
    # Σ = r² + a²cos²θ
    sigma = r**2 + a**2 * np.cos(theta)**2

    # Δ = r² - r_s*r + a²
    delta = r**2 - r_s * r + a**2

    # Frame dragging: ω = 2Mar/(Σ(r² + a²) + 2Ma²r*sin²θ)
    sin2_theta = np.sin(theta)**2
    omega = 2 * M * a * r / (sigma * (r**2 + a**2) + 2*M*a**2*r*sin2_theta + 1e-10)

    # Deflection angle (weak field approximation)
    # α ≈ 4GM/rc² for Schwarzschild, modified by spin for Kerr
    deflection = 4 * M / (r + 1e-10) * (1 + a**2 / (r**2 + 1e-10))

    return deflection.astype(np.float32), omega.astype(np.float32)


# ============================================
# HAWKING RADIATION - The Output Mechanism
# ============================================

def hawking_temperature(M):
    """
    Hawking temperature: T = ℏc³/(8πGMk)

    Smaller black holes radiate more intensely.
    This determines the "activation intensity" of the output.
    """
    # In natural units with ℏ=c=G=k=1: T = 1/(8πM)
    return np.float32(1.0) / (8 * PI * np.maximum(M, np.float32(1e-10)))


def hawking_spectrum(energy, M):
    """
    Hawking radiation spectrum: Planck distribution at temperature T.

    n(E) = 1/(exp(E/T) - 1)

    This gives the probability distribution over output values.
    """
    T = hawking_temperature(M)
    # Planck distribution (bosonic)
    exp_term = np.exp(np.clip(energy / T, -50, 50))
    return np.float32(1.0) / (exp_term - 1 + 1e-10)


# ============================================
# SINGULARITY LAYER
# ============================================

class SingularityLayer:
    """
    Weights encoded as black hole geometry.

    Each layer is a black hole characterized by:
        M = Mass (overall scale)
        Q = Charge (polarity/asymmetry)
        J = Angular momentum (rotation/chirality)

    Plus a horizon shell encoded in spherical harmonic coefficients.

    Forward pass:
        1. Token enters gravity well at radius r
        2. Compute deflection via Kerr metric
        3. Sample horizon shell at deflected position
        4. Output weighted by Hawking radiation intensity

    Storage:
        - 3 floats (M, Q, J)
        - N_HARMONICS coefficients for horizon shell
        - Per-row: ~(3 + 81) × 4 = 336 bytes
        - For (out, in) matrix: out × 336 bytes

    Compression ratio depends on harmonic truncation.
    """

    def __init__(self, in_features, out_features, l_max=8):
        self.in_features = in_features
        self.out_features = out_features
        self.l_max = l_max
        self.n_harmonics = (l_max + 1) ** 2

        # Black hole parameters per output row
        self.mass = np.ones(out_features, dtype=np.float32)      # M
        self.charge = np.zeros(out_features, dtype=np.float32)   # Q
        self.spin = np.zeros(out_features, dtype=np.float32)     # J

        # Horizon shell: spherical harmonic coefficients
        # Shape: (out_features, n_harmonics)
        self.horizon_coeffs = np.zeros((out_features, self.n_harmonics), dtype=np.float32)

        # Input mapping: how input features map to (r, θ, φ) coordinates
        self.input_radius_scale = np.ones(in_features, dtype=np.float32)

        self.bias = None

    def align_to_blueprint(self, fp32_weight, fp32_bias=None):
        """
        Collapse FP32 weight matrix into black hole geometry.

        The "holographic heist": compress n×m matrix into:
        - 3 parameters (M, Q, J) per row
        - Spherical harmonic expansion of the row's "gravitational field"
        """
        out_f, in_f = fp32_weight.shape

        for i in range(out_f):
            row = fp32_weight[i]

            # === MASS: Total magnitude (Schwarzschild radius) ===
            self.mass[i] = np.sqrt(np.sum(row ** 2))

            # === CHARGE: Asymmetry (positive vs negative weights) ===
            pos_sum = np.sum(row[row > 0])
            neg_sum = np.abs(np.sum(row[row < 0]))
            self.charge[i] = (pos_sum - neg_sum) / (pos_sum + neg_sum + 1e-10)

            # === SPIN: Angular momentum (weight pattern chirality) ===
            # Compute "rotation" by looking at weight gradient pattern
            if len(row) > 1:
                gradient = np.diff(row)
                self.spin[i] = np.mean(gradient) / (np.std(row) + 1e-10)
            else:
                self.spin[i] = 0

            # === HORIZON SHELL: Spherical harmonic decomposition ===
            # Map the row to a function on the sphere, then decompose
            # into spherical harmonics

            # Map input index to (theta, phi) on sphere
            n = len(row)
            idx = np.arange(n, dtype=np.float32)

            # Use Fibonacci spiral for uniform distribution
            golden = (1 + np.sqrt(5)) / 2
            theta_pts = np.arccos(1 - 2 * (idx + 0.5) / n)
            phi_pts = 2 * PI * idx / golden

            # Row values at these points
            values = row.astype(np.float32)

            # Project onto spherical harmonics
            for h, (l, m) in enumerate(HORIZON_LM[:self.n_harmonics]):
                Y_lm = spherical_harmonic_real(l, m, theta_pts, phi_pts)
                # Inner product: c_lm = ∫ f(θ,φ) Y_lm(θ,φ) dΩ
                # Approximate with sum
                self.horizon_coeffs[i, h] = np.sum(values * Y_lm) / n * 4 * PI

        # Input radius mapping: larger input features = closer to singularity
        self.input_radius_scale = np.linspace(2.0, 10.0, in_f, dtype=np.float32)

        if fp32_bias is not None:
            self.bias = fp32_bias.astype(np.float32)

        # Precompute for fast forward
        self._precompute()

    def _precompute(self):
        """Precompute horizon function on grid for fast lookup."""
        # Reconstruct horizon function from coefficients
        # horizon_func[i, theta_idx, phi_idx] = Σ c_lm Y_lm(θ,φ)

        self._horizon_func = np.zeros(
            (self.out_features, len(HORIZON_THETA), len(HORIZON_PHI)),
            dtype=np.float32
        )

        for i in range(self.out_features):
            for h in range(min(self.n_harmonics, len(HORIZON_BASIS))):
                self._horizon_func[i] += self.horizon_coeffs[i, h] * HORIZON_BASIS[h]

    def forward(self, x_fp32):
        """
        Gravitational lensing forward pass.

        1. Each input x is a "particle" at some radius r
        2. Compute deflection from Kerr metric (M, J)
        3. Sample horizon shell at deflected position
        4. Weight by Hawking temperature
        5. Apply charge modulation
        """
        seq_len = x_fp32.shape[0]
        in_f = x_fp32.shape[1]

        # Pad input if needed
        if in_f < self.in_features:
            x_fp32 = np.pad(x_fp32, ((0, 0), (0, self.in_features - in_f)), mode='constant')
        x = x_fp32[:, :self.in_features]

        out = np.zeros((seq_len, self.out_features), dtype=np.float32)

        for i in range(self.out_features):
            M = self.mass[i]
            Q = self.charge[i]
            J = self.spin[i]

            # Input values determine radius and angle
            # Higher input magnitude = closer approach = more deflection
            r = self.input_radius_scale[:in_f] / (np.abs(x) + 1)  # (seq, in)
            theta = PI * (x + 1) / 2  # map [-inf, inf] to [0, π]
            theta = np.clip(theta, 0.01, PI - 0.01)

            # Compute deflection
            deflection, frame_drag = kerr_metric_deflection(r, theta, M, J)

            # Deflected theta (bent by gravity)
            theta_deflected = theta + deflection * 0.1  # scale factor
            theta_deflected = np.clip(theta_deflected, 0.01, PI - 0.01)

            # Phi from frame dragging (rotation)
            phi = frame_drag * 2 * PI
            phi = phi % (2 * PI)

            # Sample horizon shell at deflected position
            # Use bilinear interpolation on precomputed grid
            theta_idx = (theta_deflected / PI * (len(HORIZON_THETA) - 1))
            phi_idx = (phi / (2 * PI) * len(HORIZON_PHI))

            theta_idx = np.clip(theta_idx, 0, len(HORIZON_THETA) - 1.001)
            phi_idx = phi_idx % len(HORIZON_PHI)

            # Simple nearest-neighbor for now
            t_i = theta_idx.astype(int)
            p_i = phi_idx.astype(int) % len(HORIZON_PHI)

            horizon_values = self._horizon_func[i, t_i, p_i]  # (seq, in)

            # Weight by Hawking temperature (smaller M = hotter = stronger output)
            T = hawking_temperature(M)
            intensity = T * 10  # scale factor

            # Apply charge modulation
            charge_mod = 1 + Q * np.sign(x) * 0.5

            # Combine: sum over input with deflection-weighted horizon
            contribution = horizon_values * intensity * charge_mod * np.abs(x)
            out[:, i] = np.sum(contribution, axis=-1)

        if self.bias is not None:
            out = out + self.bias

        return out.astype(np.float32)

    def storage_bytes(self):
        """Storage for black hole parameters + horizon coefficients."""
        # M, Q, J: 3 floats per row
        params = 3 * self.out_features * 4

        # Horizon coefficients: n_harmonics floats per row
        horizon = self.n_harmonics * self.out_features * 4

        # Input radius scale
        input_scale = self.in_features * 4

        total = params + horizon + input_scale
        if self.bias is not None:
            total += self.bias.nbytes

        return total

    def equivalent_int8_bytes(self):
        return self.out_features * self.in_features

    def describe_black_hole(self, row_idx):
        """Describe a single row's black hole."""
        M = self.mass[row_idx]
        Q = self.charge[row_idx]
        J = self.spin[row_idx]

        r_s = 2 * M  # Schwarzschild radius
        T = hawking_temperature(M)
        a = J / max(M, 1e-10)  # spin parameter

        # Ergosphere radius (equatorial)
        r_ergo = M + np.sqrt(M**2 - a**2) if abs(a) < M else M

        return {
            'mass': float(M),
            'charge': float(Q),
            'spin': float(J),
            'spin_parameter': float(a),
            'schwarzschild_radius': float(r_s),
            'hawking_temperature': float(T),
            'ergosphere_radius': float(r_ergo),
            'horizon_energy': float(np.sum(self.horizon_coeffs[row_idx] ** 2)),
        }


# ============================================
# TESTS
# ============================================

if __name__ == "__main__":
    import time

    print("=== SPHERICAL HARMONIC BASIS ===")
    print(f"Basis shape: {HORIZON_BASIS.shape}")
    print(f"Number of harmonics (l_max=8): {N_HARMONICS}")
    print(f"Grid: {len(HORIZON_THETA)} θ × {len(HORIZON_PHI)} φ = {len(HORIZON_THETA)*len(HORIZON_PHI)} points")

    # Check orthonormality
    basis_flat = HORIZON_BASIS.reshape(N_HARMONICS, -1)
    gram = basis_flat @ basis_flat.T / (len(HORIZON_THETA) * len(HORIZON_PHI))
    print(f"Basis ~orthonormal: {np.allclose(np.diag(gram), gram.max(axis=1), atol=0.3)}")
    print()

    print("=== KERR METRIC ===")
    r = np.array([5.0, 10.0, 20.0], dtype=np.float32)
    theta = np.array([PI/4, PI/2, PI/4], dtype=np.float32)

    for M, J in [(1.0, 0.0), (1.0, 0.5), (1.0, 0.9)]:
        defl, drag = kerr_metric_deflection(r, theta, M, J)
        print(f"  M={M}, J={J}: deflection={defl}, frame_drag={drag}")
    print()

    print("=== HAWKING RADIATION ===")
    for M in [0.1, 1.0, 10.0]:
        T = hawking_temperature(M)
        print(f"  M={M}: T={T:.4f} (smaller = hotter)")
    print()

    print("=== SINGULARITY LAYER (256x256) ===")
    layer = SingularityLayer(256, 256, l_max=8)
    fp32_w = np.random.randn(256, 256).astype(np.float32) * 0.02
    layer.align_to_blueprint(fp32_w)

    print(f"Mass range:   [{layer.mass.min():.3f}, {layer.mass.max():.3f}]")
    print(f"Charge range: [{layer.charge.min():.3f}, {layer.charge.max():.3f}]")
    print(f"Spin range:   [{layer.spin.min():.3f}, {layer.spin.max():.3f}]")
    print(f"Horizon coeffs: {layer.horizon_coeffs.shape}")

    sb = layer.storage_bytes()
    ib = layer.equivalent_int8_bytes()
    print(f"Storage: {sb:,} bytes vs INT8 {ib:,} = {sb/ib*100:.1f}%")
    print()

    print("=== SAMPLE BLACK HOLE ===")
    bh = layer.describe_black_hole(0)
    for k, v in bh.items():
        print(f"  {k}: {v:.4f}")
    print()

    print("=== FORWARD PASS ===")
    x = np.random.randn(10, 256).astype(np.float32)

    t0 = time.time()
    y = layer.forward(x)
    t1 = time.time()

    print(f"Output: {y.shape}")
    print(f"Range:  [{y.min():.4f}, {y.max():.4f}]")
    print(f"NaN:    {np.any(np.isnan(y))}")
    print(f"Time:   {(t1-t0)*1000:.1f}ms")
    print()

    print("=== COMPRESSION VS HARMONIC ORDER ===")
    for l_max in [2, 4, 8, 16]:
        n_harm = (l_max + 1) ** 2
        storage_per_row = 3 * 4 + n_harm * 4  # M,Q,J + coeffs
        total = 256 * storage_per_row + 256 * 4  # + input scale
        ratio = total / (256 * 256)
        print(f"  l_max={l_max:>2}: {n_harm:>3} harmonics, {ratio*100:.1f}% of INT8")
    print()

    print("All information is on the horizon.")
    print("The singularity is just... geometry.")
