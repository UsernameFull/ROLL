# Safe Megatron Model Logging Design

## Goal

Prevent Megatron strategy initialization from failing while logging models
whose third-party submodules have broken `extra_repr` implementations.

## Design

Replace recursive full-model representations with a compact summary containing
the number of virtual-pipeline model chunks and each chunk's class name. The
summary reads only `type(model).__name__`, so it never invokes model or child
module `__repr__` methods. Both inference and training initialization paths use
the same helper.

## Validation

A unit test uses objects whose `__repr__` raises and verifies that the expected
summary is still produced. Python compilation and diff validation cover the
modified runtime files.
