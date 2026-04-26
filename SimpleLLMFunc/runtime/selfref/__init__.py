"""Self-reference runtime package.

This package groups two concerns under one namespace:

- ``state``: the shared ``SelfReference`` state object and memory/fork logic
- ``primitives``: the runtime primitive pack that exposes ``selfref.*`` calls
"""

from .primitives import (
    DEFAULT_SELF_REFERENCE_BACKEND_NAME,
    SELF_REFERENCE_PACK_GUIDANCE,
    build_self_reference_pack,
    register_self_reference_primitives,
)
from .session import SelfRefSession, resolve_self_reference_key
from .state import (
    SELF_REFERENCE_FORK_TASK_TEMPLATE_PARAM,
    SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM,
    SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM,
    SelfReference,
    SelfReferenceInstanceHandle,
    SelfReferenceMemoryHandle,
    SelfReferenceMemoryProxy,
)

__all__ = [
    "SelfReference",
    "SelfReferenceMemoryHandle",
    "SelfReferenceMemoryProxy",
    "SelfReferenceInstanceHandle",
    "SelfRefSession",
    "resolve_self_reference_key",
    "SELF_REFERENCE_KEY_OVERRIDE_TEMPLATE_PARAM",
    "SELF_REFERENCE_TOOLKIT_OVERRIDE_TEMPLATE_PARAM",
    "SELF_REFERENCE_FORK_TASK_TEMPLATE_PARAM",
    "DEFAULT_SELF_REFERENCE_BACKEND_NAME",
    "SELF_REFERENCE_PACK_GUIDANCE",
    "register_self_reference_primitives",
    "build_self_reference_pack",
]
