"""
Sphere Stream - Python interface via ctypes

No PyTorch. No frameworks. Just raw CUDA.
Handles GPU memory allocation and transfer with cudart.

Compile:
    nvcc -O3 -arch=sm_61 --shared -Xcompiler -fPIC \
         sphere_stream.cu -o sphere_stream.so

Usage:
    from sphere_stream_raw import SphereStream
    stream = SphereStream("./sphere_stream.so")
    stream.upload_table(table_int8)
    output = stream.forward(tokens, weights, n_outputs)
"""

import ctypes
import ctypes.util
import numpy as np
import os


_cudart = None

def _load_cudart():
    global _cudart
    if _cudart is not None:
        return _cudart
    for name in [
        "libcudart.so", "libcudart.so.12", "libcudart.so.11.0",
        ctypes.util.find_library("cudart"),
    ]:
        if name is None:
            continue
        try:
            _cudart = ctypes.cdll.LoadLibrary(name)
            return _cudart
        except:
            continue
    for prefix in ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"]:
        for lib in ["libcudart.so.12", "libcudart.so.11.0", "libcudart.so"]:
            path = os.path.join(prefix, lib)
            if os.path.exists(path):
                try:
                    _cudart = ctypes.cdll.LoadLibrary(path)
                    return _cudart
                except:
                    continue
    raise RuntimeError("Cannot find libcudart.so - is CUDA installed?")


class GPU:
    """Thin wrapper around cudaMalloc / cudaMemcpy / cudaFree."""

    H2D = 1
    D2H = 2

    def __init__(self):
        self.rt = _load_cudart()
        self.rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.rt.cudaMalloc.restype = ctypes.c_int
        self.rt.cudaFree.argtypes = [ctypes.c_void_p]
        self.rt.cudaFree.restype = ctypes.c_int
        self.rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.rt.cudaMemcpy.restype = ctypes.c_int
        self.rt.cudaDeviceSynchronize.argtypes = []
        self.rt.cudaDeviceSynchronize.restype = ctypes.c_int

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        err = self.rt.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        if err != 0:
            raise RuntimeError(f"cudaMalloc({nbytes}) failed: error {err}")
        return ptr

    def free(self, ptr):
        self.rt.cudaFree(ptr)

    def to_gpu(self, arr):
        arr = np.ascontiguousarray(arr)
        ptr = self.malloc(arr.nbytes)
        err = self.rt.cudaMemcpy(ptr, arr.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_size_t(arr.nbytes), self.H2D)
        if err != 0:
            raise RuntimeError(f"cudaMemcpy H2D failed: error {err}")
        return ptr

    def to_cpu(self, ptr, dtype, shape):
        arr = np.empty(shape, dtype=dtype)
        err = self.rt.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), ptr,
                                  ctypes.c_size_t(arr.nbytes), self.D2H)
        if err != 0:
            raise RuntimeError(f"cudaMemcpy D2H failed: error {err}")
        return arr

    def sync(self):
        self.rt.cudaDeviceSynchronize()


class SphereStream:
    def __init__(self, lib_path="./sphere_stream.so"):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"{lib_path} not found. Compile with:\n"
                f"  nvcc -O3 -arch=sm_61 --shared -Xcompiler -fPIC "
                f"sphere_stream.cu -o sphere_stream.so")

        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.gpu = GPU()
        self._setup()
        self._rng = 42
        self._weight_cache = {}

    def _setup(self):
        VP = ctypes.c_void_p
        I32 = ctypes.c_int32
        U32 = ctypes.c_uint32
        U8 = ctypes.c_uint8

        self.lib.upload_sphere_table.argtypes = [ctypes.POINTER(ctypes.c_int8)]
        self.lib.upload_sphere_table.restype = None

        self.lib.launch_sphere_stream_forward.argtypes = [
            VP, VP, VP, I32, I32, I32, I32, U32, VP]
        self.lib.launch_sphere_stream_forward.restype = None

        self.lib.launch_sphere_stream_collide.argtypes = [
            VP, VP, VP, I32, I32, I32, I32, I32, VP, U32, VP]
        self.lib.launch_sphere_stream_collide.restype = None

        self.lib.launch_sphere_stream_earthquake.argtypes = [
            VP, I32, U8, U32, VP]
        self.lib.launch_sphere_stream_earthquake.restype = None

    def _seed(self):
        self._rng += 1
        return self._rng

    def upload_table(self, table_i8):
        assert table_i8.shape == (256, 3) and table_i8.dtype == np.int8
        flat = np.ascontiguousarray(table_i8.flatten())
        self.lib.upload_sphere_table(flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))

    def upload_weights(self, name, weights_u8):
        """Pre-upload weights to GPU. Call once per layer. Stays in VRAM."""
        if name in self._weight_cache:
            self.gpu.free(self._weight_cache[name])
        self._weight_cache[name] = self.gpu.to_gpu(weights_u8)

    def forward(self, tokens_i8, weights, n_outputs, out_shift=8):
        """
        tokens_i8:  numpy (n_tokens, n_triplets*3) int8
        weights:    numpy uint8 OR string name of pre-uploaded weights
        n_outputs:  int
        Returns:    numpy (n_tokens, n_outputs) int8
        """
        n_tokens = tokens_i8.shape[0]
        n_triplets = tokens_i8.shape[1] // 3

        d_tok = self.gpu.to_gpu(tokens_i8)

        if isinstance(weights, str):
            d_w = self._weight_cache[weights]
            free_w = False
        else:
            d_w = self.gpu.to_gpu(weights)
            free_w = True

        d_out = self.gpu.malloc(n_tokens * n_outputs)

        self.lib.launch_sphere_stream_forward(
            d_tok, d_w, d_out,
            ctypes.c_int32(n_tokens), ctypes.c_int32(n_triplets),
            ctypes.c_int32(n_outputs), ctypes.c_int32(out_shift),
            ctypes.c_uint32(self._seed()), None)

        self.gpu.sync()
        out = self.gpu.to_cpu(d_out, np.int8, (n_tokens, n_outputs))

        self.gpu.free(d_tok)
        self.gpu.free(d_out)
        if free_w:
            self.gpu.free(d_w)

        return out

    def collide(self, tokens_i8, weights_u8, n_outputs, out_shift=8, swap_threshold=32):
        """Forward + collision. Mutates weights. Returns (output, n_swaps)."""
        n_tokens = tokens_i8.shape[0]
        n_triplets = tokens_i8.shape[1] // 3

        d_tok = self.gpu.to_gpu(tokens_i8)
        d_w = self.gpu.to_gpu(weights_u8)
        d_out = self.gpu.malloc(n_tokens * n_outputs)
        d_swap = self.gpu.to_gpu(np.zeros(1, dtype=np.uint32))

        self.lib.launch_sphere_stream_collide(
            d_tok, d_w, d_out,
            ctypes.c_int32(n_tokens), ctypes.c_int32(n_triplets),
            ctypes.c_int32(n_outputs), ctypes.c_int32(out_shift),
            ctypes.c_int32(swap_threshold), d_swap,
            ctypes.c_uint32(self._seed()), None)

        self.gpu.sync()
        out = self.gpu.to_cpu(d_out, np.int8, (n_tokens, n_outputs))
        np.copyto(weights_u8, self.gpu.to_cpu(d_w, np.uint8, weights_u8.shape))
        swaps = self.gpu.to_cpu(d_swap, np.uint32, (1,))

        self.gpu.free(d_tok)
        self.gpu.free(d_w)
        self.gpu.free(d_out)
        self.gpu.free(d_swap)
        return out, int(swaps[0])

    def earthquake(self, weights_u8, magnitude=25):
        d_w = self.gpu.to_gpu(weights_u8)
        self.lib.launch_sphere_stream_earthquake(
            d_w, ctypes.c_int32(len(weights_u8)),
            ctypes.c_uint8(magnitude),
            ctypes.c_uint32(self._seed()), None)
        self.gpu.sync()
        np.copyto(weights_u8, self.gpu.to_cpu(d_w, np.uint8, weights_u8.shape))
        self.gpu.free(d_w)

    def free_cache(self):
        for ptr in self._weight_cache.values():
            self.gpu.free(ptr)
        self._weight_cache.clear()


def build_sphere_table_int8(n=256):
    PI = 3.141592653589793
    GOLDEN = (1 + 5**0.5) / 2
    half = n // 2
    table = np.empty((n, 3), dtype=np.float64)
    for i in range(half):
        phi_a = np.arccos(1.0 - 2.0 * (i + 0.5) / half)
        theta_a = 2.0 * PI * i / GOLDEN
        table[i*2] = [np.cos(theta_a)*np.sin(phi_a), np.sin(theta_a)*np.sin(phi_a), np.cos(phi_a)]
        phi_b = np.arccos(1.0 - 2.0 * (i + 1.0) / half)
        theta_b = 2.0 * PI * (i + 0.5) / GOLDEN
        table[i*2+1] = [np.cos(theta_b)*np.sin(phi_b), np.sin(theta_b)*np.sin(phi_b), np.cos(phi_b)]
    return np.round(table * 127.0).clip(-127, 127).astype(np.int8)
