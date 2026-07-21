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
