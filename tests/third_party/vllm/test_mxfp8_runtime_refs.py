from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from roll.third_party.vllm import worker


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer._mxfp8_original_shapes = {"weight": (2, 2)}
        self.layer.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.layer.weight_scale = torch.nn.Parameter(torch.ones(2))


def _iter_test_modules(self):
    yield "layer", self.model.layer


def test_mxfp8_runtime_refs_keep_storage_and_copy_new_values(monkeypatch):
    model = _Model()
    monkeypatch.setattr(worker.Mxfp8WeightLifecycle, "iter_runtime_modules", _iter_test_modules)

    original_weight = model.layer.weight.data
    original_scale = model.layer.weight_scale.data
    new_loader = object()

    assert worker._record_mxfp8_runtime_refs(model) == 2

    model.layer.weight = torch.nn.Parameter(torch.full((2, 2), 7.0))
    model.layer.weight.weight_loader = new_loader
    model.layer.weight_scale = torch.nn.Parameter(torch.full((2,), 3.0))

    assert worker._restore_mxfp8_runtime_refs(model) == 2

    assert model.layer.weight.data.data_ptr() == original_weight.data_ptr()
    assert model.layer.weight_scale.data.data_ptr() == original_scale.data_ptr()
    assert torch.equal(model.layer.weight, torch.full((2, 2), 7.0))
    assert torch.equal(model.layer.weight_scale, torch.full((2,), 3.0))
    assert model.layer.weight.weight_loader is new_loader


def test_mxfp8_runtime_refs_reject_metadata_changes(monkeypatch):
    model = _Model()
    monkeypatch.setattr(worker.Mxfp8WeightLifecycle, "iter_runtime_modules", _iter_test_modules)

    worker._record_mxfp8_runtime_refs(model)
    model.layer.weight = torch.nn.Parameter(torch.ones(3, 2))

    with pytest.raises(RuntimeError, match="graph-safe reload failed"):
        worker._restore_mxfp8_runtime_refs(model)


def test_mxfp8_transform_wrapper_reports_graph_safe_update(monkeypatch):
    model = _Model()
    monkeypatch.setattr(worker.Mxfp8WeightLifecycle, "iter_runtime_modules", _iter_test_modules)

    worker._record_mxfp8_runtime_refs(model)
    original_weight = model.layer.weight.data

    def _replace_transformed_tensors(_model):
        _model.layer.weight = torch.nn.Parameter(torch.full((2, 2), 9.0))
        _model.layer.weight_scale = torch.nn.Parameter(torch.full((2,), 4.0))

    monkeypatch.setattr(worker, "apply_mxfp8_transformation_after_loading", _replace_transformed_tensors)

    assert worker._apply_mxfp8_transformation_after_loading(model, preserve_runtime_refs=True) is True
    assert model.layer.weight.data.data_ptr() == original_weight.data_ptr()
    assert torch.equal(model.layer.weight, torch.full((2, 2), 9.0))


def test_bucketed_model_update_restores_and_finalizes_once(monkeypatch):
    model = _Model()
    loaded = []
    model.load_weights = lambda weights: loaded.append(list(weights))

    instance = object.__new__(worker.WorkerBase)
    instance.model_runner = SimpleNamespace(model=model)
    instance._is_mxfp8_model = True
    instance._mxfp8_transformation_applied = True
    instance._model_update_in_progress = False
    instance.weight_loaded = True

    calls = {"restore": 0, "finalize": 0}

    def restore(_model):
        calls["restore"] += 1

    def finalize(_model):
        calls["finalize"] += 1
        instance._mxfp8_transformation_applied = True

    monkeypatch.setattr(worker, "restore_mxfp8_weights_for_loading", restore)
    monkeypatch.setattr(worker, "_temporary_parameter_subclass_types", lambda _model: nullcontext())
    instance._finalize_weight_loading = finalize

    instance.begin_model_update()
    assert instance._model_update_in_progress
    assert calls == {"restore": 1, "finalize": 0}

    instance._load_weights_internal(model, [("first", torch.ones(1))])
    instance._load_weights_internal(model, [("second", torch.ones(1))])
    assert len(loaded) == 2
    assert calls == {"restore": 1, "finalize": 0}

    with pytest.raises(RuntimeError, match="already in progress"):
        instance.begin_model_update()

    instance.process_weights_after_loading()
    assert calls == {"restore": 1, "finalize": 1}
    assert not instance._model_update_in_progress
