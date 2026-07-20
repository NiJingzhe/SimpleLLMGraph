"""Provider-neutral context representation.

Currently exposes the :mod:`SimpleLLMFunc.context.ir` intermediate
representation. A higher-level, more semantic Event layer will sit above
the IR in future revisions.
"""

from SimpleLLMFunc.context import ir

__all__ = ["ir"]
