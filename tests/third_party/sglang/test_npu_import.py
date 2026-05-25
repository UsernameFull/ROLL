import asyncio
import gc
import importlib.util
import inspect
import os
import sys
import uuid

import pytest

from roll.platforms import current_platform


def _require_module(module_name: str) -> None:
    try:
        module_spec = importlib.util.find_spec(module_name)
    except ValueError:
        module_spec = None

    available = module_spec is not None or module_name in sys.modules
    if not available and not current_platform.is_npu():
        pytest.skip(f"{module_name} is not installed in this environment.")
    assert available, f"{module_name} must be installed for NPU SGLang tests."


def test_sglang_import_available():
    _require_module("sglang")
    import sglang

    assert sglang.__version__


async def _generate_sglang_chunks(model, request):
    generator = model.tokenizer_manager.generate_request(request, None)
    chunks = None
    async for chunks in generator:
        pass
    if chunks is None:
        return []
    return chunks if isinstance(chunks, list) else [chunks]


async def _shutdown_sglang_engine(model):
    for method_name in ("shutdown", "close"):
        method = getattr(model, method_name, None)
        if method is None:
            continue
        result = method()
        if inspect.isawaitable(result):
            await result
        return


async def _run_npu_sglang_abort_smoke():
    _require_module("sglang")
    _require_module("sgl_kernel_npu")

    from sglang.srt.managers.io_struct import GenerateReqInput
    from transformers import AutoTokenizer

    from roll.distributed.strategy.sglang_strategy import shutdown as shutdown_sglang_processes
    from roll.third_party.sglang import patch as sglang_patch
    from roll.utils import checkpoint_manager

    model = None
    try:
        model_name_or_path = os.environ.get("ROLL_NPU_SGLANG_SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        model_path = checkpoint_manager.download_model(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        input_ids = tokenizer(["Count upward and keep going."])["input_ids"][0]

        model = sglang_patch.engine.engine_module.Engine(
            model_path=model_path,
            enable_memory_saver=False,
            skip_tokenizer_init=False,
            dtype="bfloat16",
            tp_size=1,
            mem_fraction_static=0.35,
            max_total_tokens=1024,
            max_running_requests=1,
            disable_custom_all_reduce=True,
            trust_remote_code=True,
        )
        auto_create_handle_loop = getattr(model.tokenizer_manager, "auto_create_handle_loop", None)
        if auto_create_handle_loop is not None:
            auto_create_handle_loop()

        request_id = uuid.uuid4().hex
        request = GenerateReqInput(
            input_ids=input_ids,
            sampling_params={
                "temperature": 0.0,
                "min_new_tokens": 512,
                "max_new_tokens": 512,
                "n": 1,
            },
            rid=request_id,
            return_logprob=False,
        )

        task = asyncio.create_task(_generate_sglang_chunks(model, request))
        await asyncio.sleep(float(os.environ.get("ROLL_NPU_SGLANG_ABORT_DELAY", "0.2")))
        result = model.tokenizer_manager.abort_request(request_id)
        if inspect.isawaitable(result):
            await result

        chunks = await asyncio.wait_for(task, timeout=120)
        assert chunks
        finish_reason = chunks[0]["meta_info"]["finish_reason"]
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason["type"]
        assert finish_reason == "abort"
    finally:
        if model is not None:
            try:
                await _shutdown_sglang_engine(model)
            except Exception as e:
                print(f"Failed to shut down SGLang smoke model cleanly: {e}")
        shutdown_sglang_processes()
        checkpoint_manager.shared_storage = None
        gc.collect()
        empty_cache = getattr(current_platform, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()


def test_npu_sglang_abort_smoke():
    if not current_platform.is_npu():
        pytest.skip("NPU SGLang abort smoke only applies on Ascend NPU.")
    if os.environ.get("ROLL_NPU_SGLANG_ABORT_SMOKE", "1") == "0":
        pytest.skip("ROLL_NPU_SGLANG_ABORT_SMOKE=0")

    asyncio.run(_run_npu_sglang_abort_smoke())
