"""Vector-space text modeling using deterministic hashing."""

import re
import hashlib
import numpy as np
from typing import Tuple, List


class HashSketch:
    """
    CountSketch-like hashing for text -> R^D.
    Deterministic (SHA1). No vocab, no training.
    """

    def __init__(
        self, dim: int = 256, ngram: Tuple[int, int] = (1, 2), lowercase: bool = True
    ):
        self.dim, self.ngram, self.lowercase = dim, ngram, lowercase

    def _hash(self, s: str, salt: str) -> int:
        h = hashlib.sha1((salt + s).encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little", signed=False)

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        if self.lowercase:
            text = text.lower()
        toks = re.findall(r"[a-zA-Z0-9_\-/\.]+", text)
        nmin, nmax = self.ngram
        out = []
        for n in range(nmin, nmax + 1):
            for i in range(0, len(toks) - n + 1):
                out.append(" ".join(toks[i : i + n]))
        return out

    def encode(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokenize(text):
            i = self._hash(tok, "i") % self.dim
            s = 1.0 if (self._hash(tok, "s") & 1) == 0 else -1.0
            v[i] += s
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v


def _sliding_window_view(
    x: np.ndarray, window: int, step: int = 1, pad: int = 0
) -> np.ndarray:
    """Windows along axis 0 with zero padding on both sides."""
    T = x.shape[0]
    if pad > 0:
        left = np.zeros((pad,) + x.shape[1:], dtype=x.dtype)
        right = np.zeros((pad,) + x.shape[1:], dtype=x.dtype)
        xx = np.concatenate([left, x, right], axis=0)
    else:
        xx = x
    T2 = xx.shape[0]
    T_out = (T2 - window) // step + 1
    s0 = xx.strides[0]
    shape = (T_out, window) + x.shape[1:]
    strides = (step * s0, s0) + x.strides[1:]
    return np.lib.stride_tricks.as_strided(
        xx, shape=shape, strides=strides, writeable=False
    )


def conv1d_text(
    X: np.ndarray, K: np.ndarray, stride: int = 1, padding: str = "same"
) -> np.ndarray:
    """
    Cross-correlation: sum_{i,j} X[t+i,j]*K[i,j]
    X: (T,D), K: (W,D) -> y: (T_out,)
    padding: 'same' (pad=W//2) or 'valid'
    """
    W = K.shape[0]
    pad = W // 2 if padding == "same" else 0
    windows = _sliding_window_view(X, W, step=stride, pad=pad)
    y = np.tensordot(windows, K, axes=([1, 2], [0, 1]))  # (T_out,)
    return y.astype(np.float32)


def make_kernel_from_phrases(
    phrases: List[str], sketch: HashSketch, width: int
) -> np.ndarray:
    """
    Kernel = (triangular envelope) ⊗ (avg phrase vector).
    Width controls temporal patterning; envelope gives central emphasis.
    """
    v = np.zeros(sketch.dim, dtype=np.float32)
    for p in phrases:
        v += sketch.encode(p)
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    W = width
    if W <= 1:
        return v[np.newaxis, :].astype(np.float32)
    env = np.array(
        [1.0 - abs(2 * i / (W - 1) - 1.0) for i in range(W)], dtype=np.float32
    )
    env /= env.max() if env.max() > 0 else 1.0
    return np.outer(env, v).astype(np.float32)
