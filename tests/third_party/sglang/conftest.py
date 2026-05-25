import os


if os.environ.get("ROLL_NPU_CI") == "1":
    collect_ignore = [
        "test_abort.py",
        "test_abort_grpc.py",
        "test_abort_http.py",
        "test_fp8.py",
    ]
