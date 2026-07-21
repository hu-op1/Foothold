"""Layer builder registry for computation graph construction.

Each attention/FFN type is a callable (ctx, layer_type, hw) -> list[OpSpec].
Register via @register_attention("name") / @register_ffn("name").
"""

ATTENTION_REGISTRY: dict[str, callable] = {}
FFN_REGISTRY: dict[str, callable] = {}


def register_attention(name: str):
    def decorator(fn):
        ATTENTION_REGISTRY[name] = fn
        return fn
    return decorator


def register_ffn(name: str):
    def decorator(fn):
        FFN_REGISTRY[name] = fn
        return fn
    return decorator


# Import builders so decorators fire at module load time.
from sim.layers.common import (     # noqa: E402, F401
    build_fused_residual_norm,
    build_rmsnorm,
    build_rope,
    build_swiglu,
)
from sim.layers.standard_attention import build_standard_attention       # noqa: E402, F401
from sim.layers.dense_ffn import build_dense_ffn                         # noqa: E402, F401
from sim.layers.moe_ffn import build_moe_ffn                             # noqa: E402, F401
from sim.layers.linear_attention import build_linear_attention           # noqa: E402, F401
from sim.layers.head import build_lm_head                                # noqa: E402, F401
