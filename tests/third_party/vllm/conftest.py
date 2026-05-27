import os


if os.environ.get("ROLL_NPU_CI") == "1":
    collect_ignore = [
        "test_abort.py",
        "test_add_requests.py",
        "test_collective_rpc.py",
        "test_fp8.py",
        "test_fp8_perf.py",
        "test_model_update.py",
        "test_sleep_level.py",
        "test_vllm_local.py",
        "test_vllm_local_actor.py",
        "test_vllm_local_async.py",
        "test_vllm_mem_oom.py",
        "vllm_generate_test.py",
    ]
