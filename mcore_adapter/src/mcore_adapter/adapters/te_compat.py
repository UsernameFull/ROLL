class _UnavailableTE:
    pass


try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEGroupedLinear,
        TELayerNormColumnParallelLinear,
        TELinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    HAVE_MEGATRON_TE = True
except ImportError:
    TEColumnParallelGroupedLinear = _UnavailableTE
    TEColumnParallelLinear = _UnavailableTE
    TEGroupedLinear = _UnavailableTE
    TELayerNormColumnParallelLinear = _UnavailableTE
    TELinear = _UnavailableTE
    TERowParallelGroupedLinear = _UnavailableTE
    TERowParallelLinear = _UnavailableTE
    HAVE_MEGATRON_TE = False
