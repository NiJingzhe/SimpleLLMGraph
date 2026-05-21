# Primitive Developer API Plan

<!-- DOC_SUMMARY_START -->
This document summarizes the current runtime primitive authoring and registration flow in SimpleLLMFunc, identifies the main developer-experience issues, and proposes a new API design for custom primitive packs, backend binding, and registration. It is intended as the implementation plan for `feat/better-primitive-dev`.
<!-- DOC_SUMMARY_END -->

<!-- DOC_MAP_START -->
## Document Map

- [Primitive Developer API Plan](#primitive-developer-api-plan)
  - [Document Maintenance](#document-maintenance)
    - [When to Update](#when-to-update)
    - [Editing Rules](#editing-rules)
    - [Related Files](#related-files)
  - [Current Surface](#current-surface)
    - [Python-Side APIs](#python-side-apis)
    - [In-REPL Runtime APIs](#in-repl-runtime-apis)
    - [Current Authoring Flow](#current-authoring-flow)
  - [Observed Problems](#observed-problems)
    - [DX Problems](#dx-problems)
    - [Functional Gaps](#functional-gaps)
  - [Design Goals](#design-goals)
  - [Proposed API](#proposed-api)
    - [Core Types](#core-types)
    - [Pack Installation](#pack-installation)
    - [Decorator Sugar](#decorator-sugar)
    - [Context Improvements](#context-improvements)
    - [Contract Metadata](#contract-metadata)
    - [Compatibility Rules](#compatibility-rules)
  - [Before And After](#before-and-after)
  - [Migration Plan](#migration-plan)
  - [Implementation Slices](#implementation-slices)
  - [Open Questions](#open-questions)
<!-- DOC_MAP_END -->

<!-- DOC_META_START -->
## Document Maintenance

### When to Update
- When the runtime primitive authoring API changes.
- When the pack installation / cloning strategy changes.
- When compatibility or migration rules change.

### Editing Rules
- Keep the recommended API examples in sync with the real public surface.
- Preserve the distinction between additive changes and breaking changes.
- Update migration steps when implementation order changes.

### Related Files
- This document: `spec/primitive-dev-api-plan.md`
- Related spec: `spec/meta.md`
- Related code: `SimpleLLMFunc/runtime/primitives.py`, `SimpleLLMFunc/builtin/pyrepl_primitive_host.py`, `SimpleLLMFunc/runtime/selfref/primitives.py`, `SimpleLLMFunc/llm_decorator/llm_chat_decorator.py`
- Related docs: `mintlify_docs/pyrepl.mdx`, `mintlify_docs/detailed_guide/llm_chat.mdx`
<!-- DOC_META_END -->

## Current Surface {#current-surface}

### Python-Side APIs {#python-side-apis}

The current public surface is split across `SimpleLLMFunc.runtime` and `SimpleLLMFunc.builtin.PyRepl`.

Current runtime exports:

- `PrimitiveCallContext`
- `PrimitiveHandler`
- `PrimitiveParameterSpec`
- `PrimitiveRegistry`
- `PrimitiveSpec`
- `primitive`
- `primitive_spec`
- `register_self_reference_primitives`

Current `PyRepl` authoring APIs:

- `register_runtime_backend(name, backend, replace=False)`
- `unregister_runtime_backend(name)`
- `get_runtime_backend(name)`
- `list_runtime_backends()`
- `register_primitive(name, handler, ..., replace=False)`
- `unregister_primitive(name)`
- `list_primitives()`
- `list_primitive_specs(..., format="xml")`
- `get_primitive_spec(name, format="xml")`
- `register_primitive_pack_installer(pack_name, installer, replace=False)`
- `install_primitive_pack(pack_name, **options)`

### In-REPL Runtime APIs {#in-repl-runtime-apis}

Inside `execute_code`, worker-side code uses a dynamic `runtime` proxy:

- `runtime.namespace.name(...)`
- `runtime.call(name, *args, **kwargs)`
- `runtime.list_primitives(prefix=None, contains=None)`
- `runtime.list_primitive_specs(..., format="xml")`
- `runtime.get_primitive_spec(name, format="xml")`
- `runtime.list_backends()`

This dotted runtime call surface is good and should be preserved.

### Current Authoring Flow {#current-authoring-flow}

Today a custom primitive typically looks like this:

```python
repl.register_runtime_backend("constants", {"project": "SimpleLLMFunc"})


def constants_get(_ctx, key: str):
    backend = repl.get_runtime_backend("constants")
    if not isinstance(backend, dict):
        raise RuntimeError("runtime backend 'constants' must be a dict")
    return backend.get(key)


repl.register_primitive(
    "constants.get",
    constants_get,
    description="Read one value from constants backend.",
    replace=True,
)
```

For built-in packs such as selfref, pack installers manually register many primitives against closures that capture backend getters.

## Observed Problems {#observed-problems}

### DX Problems {#dx-problems}

1. Metadata is split across too many places.
   - `primitive()` docstring parsing handles `Use`, `Input`, `Output`, `Parse`, `Parameters`, `Best Practices`.
   - decorator kwargs handle `parameters`, `best_practices`, `next_steps`.
   - `PrimitiveRegistry.register(...)` handles a partly overlapping set.
   - `PyRepl.register_primitive(...)` exposes an even smaller subset.

2. Backend access is awkward.
   - Primitive handlers often close over the concrete `repl` instance.
   - `PrimitiveCallContext` does not expose a backend accessor.
   - The author has to manually call `repl.get_runtime_backend(...)` inside the handler.

3. Pack registration is too low-level.
   - Developers manually register backends, then primitives, then optional installers.
   - There is no first-class pack object that groups namespace, backend, primitives, and clone behavior.

4. The public authoring API is under-documented.
   - The source code exposes a real runtime authoring layer in `SimpleLLMFunc.runtime`.
   - The docs mostly teach the lower-level `PyRepl.register_*` flow.

5. Python-facing spec queries are easy to misuse.
   - `list_primitive_specs()` and `get_primitive_spec()` default to XML.
   - This is good for model prompting, but surprising for Python authors who expect dict payloads.

6. Handler signature rules are implicit.
   - Runtime handlers are called as `handler(ctx, *args, **kwargs)`.
   - Registration does not eagerly validate the required context parameter.

### Functional Gaps {#functional-gaps}

1. Custom primitives do not appear to be cloned into forked `PyRepl` instances.
   - `llm_chat` clones mounted `PyRepl` instances for forked child agents.
   - The clone path preserves runtime backends and special-cases selfref.
   - It does not preserve arbitrary custom registered primitives or custom pack installers.

2. Current examples are already showing drift.
   - `examples/runtime_primitives_basic_example.py` uses `runtime.list_primitive_specs()` as if it returned dicts.
   - That no longer matches the current XML default.

## Design Goals {#design-goals}

The new developer API should:

1. Preserve the worker-side runtime call surface:
   - `runtime.namespace.name(...)`

2. Make custom primitive authoring obvious without source reading.

3. Give primitive handlers direct access to their backend without `repl` closure glue.

4. Introduce a first-class pack abstraction so that custom extensions can be cloned into forked `PyRepl` instances.

5. Keep current APIs working during migration.

6. Keep XML contract output as the default model-facing format for backward compatibility, while adding friendlier dict-native helpers for Python authors.

## Proposed API {#proposed-api}

### Core Types {#core-types}

Add two new public runtime types.

#### `PrimitiveContract`

A typed metadata object that becomes the single source of truth for primitive contract data.

Suggested shape:

```python
@dataclass(frozen=True)
class PrimitiveContract:
    description: str = ""
    input_type: str = ""
    output_type: str = ""
    output_parsing: str = ""
    parameters: tuple[PrimitiveParameterSpec, ...] = ()
    best_practices: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()
```

Role:

- replaces the current fragmented metadata story
- can be built from docstrings, decorator kwargs, or explicit objects
- can be reused across `PrimitiveRegistry`, `PyRepl`, and pack objects

#### `PrimitivePack`

A first-class runtime extension unit.

Suggested shape:

```python
class PrimitivePack:
    name: str
    backend_name: str
    backend: Any

    def primitive(self, name: str, ..., contract: PrimitiveContract | None = None): ...
    def install(self, repl: PyRepl, *, replace: bool = False) -> None: ...
    def clone_for_repl(self, repl: PyRepl, *, self_reference_override: Any = None): ...
```

Role:

- groups namespace, backend, registered primitives, and install logic
- becomes the cloneable unit when `PyRepl` is copied for selfref forks
- replaces manual closure-heavy pack installers for most user code

### Pack Installation {#pack-installation}

Add a direct installation API on `PyRepl`:

```python
repl.install_pack(pack, replace=False)
```

Recommended author flow:

```python
from SimpleLLMFunc.runtime import PrimitivePack, primitive

constants = PrimitivePack(
    name="constants",
    backend_name="constants",
    backend={"project": "SimpleLLMFunc"},
)


@constants.primitive("get")
def constants_get(ctx, key: str) -> str | None:
    return ctx.backend.get(key)


repl.install_pack(constants)
```

Behavior:

- installs backend under `backend_name`
- registers every primitive in the pack under `name.segment`
- records the installed pack so that child `PyRepl` clones can reproduce it

### Decorator Sugar {#decorator-sugar}

Add low-friction sugar on `PyRepl` for small custom extensions.

Suggested API:

```python
@repl.primitive(
    "constants.get",
    backend="constants",
    description="Read one value from the constants backend.",
)
def constants_get(ctx, key: str) -> str | None:
    return ctx.backend.get(key)
```

Suggested companion builder:

```python
constants = repl.pack("constants", backend={"project": "SimpleLLMFunc"})


@constants.primitive("get")
def constants_get(ctx, key: str) -> str | None:
    return ctx.backend.get(key)
```

The sugar should compile down to the same internal `PrimitivePack` representation.

### Context Improvements {#context-improvements}

Extend `PrimitiveCallContext` with backend-aware helpers.

Suggested additions:

```python
@dataclass
class PrimitiveCallContext:
    primitive_name: str
    call_id: str
    execution_id: str
    event_emitter: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    repl: Any = None
    registry: PrimitiveRegistry | None = None
    backend_name: str | None = None
    backend: Any = None

    def get_backend(self, name: str) -> Any: ...
```

Benefits:

- custom primitives no longer need to close over `repl`
- pack-local backend access becomes straightforward: `ctx.backend`
- cross-backend access remains possible: `ctx.get_backend("other")`

### Contract Metadata {#contract-metadata}

Keep docstring authoring, but make it one input into a typed contract rather than the whole system.

Recommended rules:

1. `primitive(...)` may still parse the docstring.
2. Explicit decorator kwargs override docstring-derived fields.
3. `PrimitiveContract` becomes the resolved representation stored in the registry.
4. `PyRepl.register_primitive(...)` should accept the full contract surface, including:
   - `output_parsing`
   - `next_steps`

Also add dict-native inspection helpers for Python authors:

- `PyRepl.list_primitive_contracts(...)`
- `PyRepl.get_primitive_contract(name)`
- `runtime.list_primitive_specs_dict(...)`
- `runtime.get_primitive_spec_dict(name)`

These should be additive helpers. Existing XML defaults should stay unchanged.

### Compatibility Rules {#compatibility-rules}

Keep these existing surfaces working:

- `register_runtime_backend(...)`
- `register_primitive(...)`
- `register_primitive_pack_installer(...)`
- `install_primitive_pack(...)`
- `runtime.list_primitives(...)`
- `runtime.get_primitive_spec(...)`
- `runtime.list_primitive_specs(...)`
- all existing dotted runtime names such as `runtime.selfref.context.inspect()`

Compatibility strategy:

1. Legacy `register_primitive(...)` writes into an internal legacy pack.
2. Legacy pack installers are adapted into installable `PrimitivePack` objects.
3. Built-in selfref is migrated to the new pack model and installed through the same `PrimitivePack` path during `PyRepl` startup.

## Before And After {#before-and-after}

Current style:

```python
repl.register_runtime_backend("constants", {"project": "SimpleLLMFunc"})


def constants_get(_ctx, key: str):
    backend = repl.get_runtime_backend("constants")
    if not isinstance(backend, dict):
        raise RuntimeError("runtime backend 'constants' must be a dict")
    return backend.get(key)


repl.register_primitive(
    "constants.get",
    constants_get,
    description="Read one value from constants backend.",
    replace=True,
)
```

Recommended style:

```python
constants = repl.pack("constants", backend={"project": "SimpleLLMFunc"})


@constants.primitive(
    "get",
    description="Read one value from the constants backend.",
    output_parsing="Use the returned string directly. Handle None when the key is missing.",
)
def constants_get(ctx, key: str) -> str | None:
    return ctx.backend.get(key)
```

Key improvement:

- the namespace is still `runtime.constants.get(...)`
- the handler no longer needs a `repl` closure
- installation and cloning become pack-driven instead of ad hoc

## Migration Plan {#migration-plan}

### Phase 1

Add the new abstractions without breaking anything:

- `PrimitiveContract`
- `PrimitivePack`
- `PyRepl.install_pack(...)`
- `PyRepl.pack(...)`
- `PyRepl.primitive(...)`
- context backend helpers

### Phase 2

Bridge the old APIs onto the new model:

- make `register_primitive(...)` write to an internal legacy pack
- make `register_primitive_pack_installer(...)` use compatibility adapters
- start `PyRepl` with builtin selfref already installed through `install_pack(...)`

### Phase 3

Fix fork cloning:

- clone installed packs, not just raw backends
- preserve custom primitives in selfref child REPLs
- remove the current selfref special-case after the builtin pack path is in place

### Phase 4

Add Python-friendly contract helpers:

- dict-native helpers for programmatic inspection
- keep XML defaults for current runtime methods

### Phase 5

Update docs, examples, and tests:

- add a dedicated runtime-extension authoring guide
- update `examples/runtime_primitives_basic_example.py` to use dict-native inspection where appropriate
- add tests for custom pack cloning and custom primitive registration

## Implementation Slices {#implementation-slices}

Recommended implementation order for this branch:

1. Add new core types in `SimpleLLMFunc/runtime/primitives.py`.
2. Add pack installation tracking to `SimpleLLMFunc/builtin/pyrepl_primitive_host.py`.
3. Update `PrimitiveCallContext` and registry call plumbing to expose backend access.
4. Refactor builtin selfref pack onto the new internal pack model.
5. Update `SimpleLLMFunc/llm_decorator/llm_chat_decorator.py` clone logic to clone installed packs.
6. Add tests for custom pack installation and fork propagation.
7. Refresh docs/examples once the final API shape is stable.

## Open Questions {#open-questions}

1. Should `PrimitivePack` support explicit uninstall hooks, or is install-only sufficient for now?
2. Should dict-native contract helpers live under the existing method names with `format="dict"`, or be exposed as new explicit helpers to avoid ambiguity?
3. Should we validate that handler first parameter is named `ctx`, or only validate the positional contract?
4. Should pack cloning support custom backend clone callbacks, or default to shallow reuse except for special stateful builtins such as selfref and PyRepl?
