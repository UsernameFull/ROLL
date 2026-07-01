from contextlib import AbstractContextManager

import torch


def copy_parameter_metadata(param: torch.nn.Parameter, source) -> None:
    if source is None or not isinstance(source, torch.nn.Parameter):
        return

    base_param_attrs = set(dir(torch.nn.Parameter))
    for attr in dir(source):
        if attr in base_param_attrs or attr.startswith("__"):
            continue
        try:
            setattr(param, attr, getattr(source, attr))
        except (AttributeError, RuntimeError):
            pass

    subclass_type = getattr(source, "subclass_type", type(source))
    if subclass_type is not torch.nn.Parameter:
        param.subclass_type = subclass_type


def parameter_from_subclass_attributes(custom_param) -> torch.nn.Parameter:
    param = torch.nn.Parameter(custom_param.data, requires_grad=False)
    copy_parameter_metadata(param, custom_param)
    if type(custom_param) is not torch.nn.Parameter:
        param.subclass_type = type(custom_param)
    return param


def parameter_from_data_and_source(data: torch.Tensor, source) -> torch.nn.Parameter:
    param = torch.nn.Parameter(data, requires_grad=False)
    copy_parameter_metadata(param, source)
    return param


def parameter_from_runtime_ref(ref: torch.Tensor, source) -> torch.nn.Parameter:
    return parameter_from_data_and_source(ref, source)


def runtime_value_from_ref(ref: torch.Tensor, source):
    if isinstance(source, torch.nn.Parameter):
        return parameter_from_runtime_ref(ref, source)
    return ref


def replace_parameter_preserve_metadata(layer: torch.nn.Module, param_name: str, new_data: torch.Tensor | None) -> None:
    if new_data is None:
        setattr(layer, param_name, None)
        return

    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data

    old_param = getattr(layer, param_name, None)
    setattr(layer, param_name, parameter_from_data_and_source(new_data, old_param))


def restore_layer_parameter_metadata(layer: torch.nn.Module, old_params: dict[str, torch.nn.Parameter]) -> None:
    for name, old_param in old_params.items():
        new_param = getattr(layer, name, None)
        if isinstance(new_param, torch.nn.Parameter):
            copy_parameter_metadata(new_param, old_param)


class TemporaryParameterSubclassTypes(AbstractContextManager):
    """Temporarily restore vLLM custom Parameter subclasses for load_weights."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._patched_params = []

    def __enter__(self):
        for _name, param in self.model.named_parameters():
            subclass_type = getattr(param, "subclass_type", None)
            if subclass_type is None or param.__class__ is subclass_type:
                continue
            self._patched_params.append((param, param.__class__))
            param.__class__ = subclass_type
        return self

    def __exit__(self, exc_type, exc, tb):
        for param, original_type in reversed(self._patched_params):
            param.__class__ = original_type
        self._patched_params = []
        return False
