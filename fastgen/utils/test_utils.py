# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test utilities for fastgen, including:
- RunIf: Conditional test skipping based on hardware/software requirements
- Distributed testing infrastructure for multi-GPU FSDP tests
"""

import os
import pickle
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import sys

import pytest
from pytest import MarkDecorator
from packaging.version import Version
from pkg_resources import get_distribution
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# =============================================================================
# Distributed Testing Infrastructure
# =============================================================================


def _init_distributed(rank: int, world_size: int, port: int, nccl_timeout: int = 600):
    """Initialize distributed process group for testing.

    Args:
        rank: Process rank
        world_size: Total number of processes
        port: Port for distributed communication
        nccl_timeout: Timeout in seconds for NCCL operations (default: 600)
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    from datetime import timedelta

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=nccl_timeout),
    )
    torch.cuda.set_device(rank)


def _cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _error_watchdog(error_event: mp.Event, check_interval: float = 0.5):
    """Watchdog thread that monitors for errors from other ranks.

    When the error_event is set by any rank, this thread will terminate
    the current process to prevent hanging on NCCL collectives.

    Args:
        error_event: Shared multiprocessing Event that signals an error occurred
        check_interval: How often to check for errors (seconds)
    """
    while True:
        if error_event.is_set():
            # Another rank failed, force exit to prevent hanging on collectives
            os._exit(1)
        time.sleep(check_interval)


def _run_distributed_test(
    rank: int,
    world_size: int,
    port: int,
    test_fn: Callable,
    result_file: str,
    error_file: str,
    error_event: mp.Event,
    setup_fn: Optional[Callable] = None,
    nccl_timeout: int = 600,
    **kwargs,
):
    """Worker function to run a distributed test.

    Args:
        rank: Process rank
        world_size: Total number of processes
        port: Port for distributed communication
        test_fn: The test function to run with signature fn(rank, world_size, **kwargs)
        result_file: Path to file where rank 0 writes results
        error_file: Path to file where any rank writes error details
        error_event: Shared event to signal errors across ranks
        setup_fn: Optional setup function to run after distributed init (e.g., set_env_vars)
        nccl_timeout: Timeout in seconds for NCCL operations (default: 600)
        **kwargs: Additional arguments to pass to test_fn
    """
    # Start watchdog thread to monitor for errors from other ranks
    watchdog = threading.Thread(target=_error_watchdog, args=(error_event,), daemon=True)
    watchdog.start()

    try:
        _init_distributed(rank, world_size, port, nccl_timeout=nccl_timeout)

        # Run optional setup function (e.g., set_env_vars for library-specific setup)
        if setup_fn is not None:
            setup_fn()

        # Run the test function
        result = test_fn(rank=rank, world_size=world_size, **kwargs)

        # Only rank 0 writes the result to file
        if rank == 0:
            with open(result_file, "wb") as f:
                pickle.dump(("success", result), f)

    except Exception as e:
        import traceback

        error_info = {
            "rank": rank,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

        # Signal error to all other ranks FIRST (before any cleanup)
        error_event.set()

        # Write error details to file (use rank-specific suffix to avoid race conditions)
        try:
            error_file_rank = f"{error_file}.rank{rank}"
            with open(error_file_rank, "wb") as f:
                pickle.dump(error_info, f)
        except Exception:
            pass  # Best effort - error signaling via event is the priority

        # Also write to main result file if rank 0
        if rank == 0:
            with open(result_file, "wb") as f:
                pickle.dump(("error", (str(e), traceback.format_exc())), f)

    finally:
        _cleanup_distributed()


def run_distributed_test(
    test_fn: Callable,
    world_size: int = 2,
    timeout: int = 300,
    setup_fn: Optional[Callable] = None,
    nccl_timeout: Optional[int] = None,
    **kwargs,
) -> Optional[Dict]:
    """Run a test function in a distributed setting using multiprocessing.spawn.

    This utility handles:
    - Finding a free port for NCCL communication
    - Spawning worker processes with proper distributed initialization
    - Collecting results from rank 0 via pickle file (avoiding Queue issues with CUDA)
    - Cross-rank error signaling to prevent hanging when one rank fails
    - Proper cleanup of process groups

    Args:
        test_fn: Test function with signature fn(rank, world_size, **kwargs)
                 Should return a dict with test results from rank 0.
        world_size: Number of processes to spawn (default: 2)
        timeout: Timeout in seconds for each process to complete (default: 300)
        setup_fn: Optional setup function to run after distributed init on each rank.
                  Useful for library-specific setup like set_env_vars().
        nccl_timeout: Timeout in seconds for NCCL operations. Defaults to match `timeout`.
        **kwargs: Additional arguments to pass to test_fn

    Returns:
        Result dict from rank 0 test execution, or None if no result was written

    Raises:
        TimeoutError: If any process exceeds the timeout
        AssertionError: If any process exits with non-zero code or test fails

    Example:
        ```python
        def _test_my_distributed_feature(rank: int, world_size: int, some_arg: int) -> dict:
            # Setup distributed model/training
            ...
            # Run test logic
            ...
            # Return results (only rank 0's result is collected)
            return {"passed": True, "value": computed_value}

        @RunIf(min_gpus=2)
        def test_my_distributed_feature():
            result = run_distributed_test(
                test_fn=_test_my_distributed_feature,
                world_size=2,
                some_arg=42,
            )
            assert result["passed"]
        ```
    """
    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    # Create temp files for results and errors
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        result_file = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".err") as f:
        error_file = f.name

    # Create shared error event for cross-rank signaling
    ctx = mp.get_context("spawn")
    error_event = ctx.Event()

    # Default nccl_timeout to match test timeout
    if nccl_timeout is None:
        nccl_timeout = timeout

    processes = []
    try:
        # Spawn processes
        for rank in range(world_size):
            p = ctx.Process(
                target=_run_distributed_test,
                args=(rank, world_size, port, test_fn, result_file, error_file, error_event),
                kwargs={"setup_fn": setup_fn, "nccl_timeout": nccl_timeout, **kwargs},
            )
            p.start()
            processes.append(p)

        # Wait for processes with periodic error checking
        start_time = time.time()
        while True:
            # Check if error event was set
            if error_event.is_set():
                # Give processes a moment to write error files and exit
                time.sleep(0.5)
                # Terminate any still-running processes
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5)
                break

            # Check if all processes have finished
            all_done = all(not p.is_alive() for p in processes)
            if all_done:
                break

            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5)
                raise TimeoutError(f"Distributed test timed out after {timeout}s")

            # Brief sleep to avoid busy waiting
            time.sleep(0.1)

        # Join all processes to collect exit codes
        for p in processes:
            p.join(timeout=5)

        # Collect error information from all ranks
        error_messages = []
        for rank in range(world_size):
            error_file_rank = f"{error_file}.rank{rank}"
            if os.path.exists(error_file_rank):
                try:
                    with open(error_file_rank, "rb") as f:
                        error_info = pickle.load(f)
                    error_messages.append(
                        f"Rank {error_info['rank']} failed:\n{error_info['error']}\n{error_info['traceback']}"
                    )
                except Exception:
                    error_messages.append(f"Rank {rank} failed (unable to read error details)")

        # If we collected errors, raise them
        if error_messages:
            raise AssertionError(
                f"Distributed test failed on {len(error_messages)} rank(s):\n\n" + "\n---\n".join(error_messages)
            )

        # Check if any process failed with non-zero exit code
        for i, p in enumerate(processes):
            if p.exitcode != 0:
                raise AssertionError(f"Process {i} exited with code {p.exitcode}")

        # Read result from file
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            with open(result_file, "rb") as f:
                status, result = pickle.load(f)
            if status == "error":
                error_msg, traceback_str = result
                raise AssertionError(f"Distributed test failed:\n{error_msg}\n{traceback_str}")
            return result

        return None

    finally:
        # Ensure all processes are terminated
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()

        # Clean up temp files
        for f in [result_file, error_file]:
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception:
                    pass
        # Clean up rank-specific error files
        for rank in range(world_size):
            error_file_rank = f"{error_file}.rank{rank}"
            if os.path.exists(error_file_rank):
                try:
                    os.unlink(error_file_rank)
                except Exception:
                    pass


# =============================================================================
# Conditional Test Skipping
# =============================================================================


class RunIf:
    """RunIf wrapper for conditional skipping of tests.

    Fully compatible with `@pytest.mark`.

    Example:

    ```python
        @RunIf(min_torch="1.8")
        @pytest.mark.parametrize("arg1", [1.0, 2.0])
        def test_wrapper(arg1):
            assert arg1 > 0
    ```
    """

    def __new__(
        cls,
        min_gpus: int = 0,
        min_torch: Optional[str] = None,
        max_torch: Optional[str] = None,
        min_python: Optional[str] = None,
        supported_arch: Optional[List[str]] = None,
        requires_file: Optional[Union[str, List[str]]] = None,
        **kwargs: Dict[Any, Any],
    ) -> MarkDecorator:
        """Creates a new `@RunIf` `MarkDecorator` decorator.

        :param min_gpus: Min number of GPUs required to run test.
        :param min_torch: Minimum pytorch version to run test.
        :param max_torch: Maximum pytorch version to run test.
        :param min_python: Minimum python version required to run test.
        :param requires_file: File or list of files required to run test.
        :param kwargs: Native `pytest.mark.skipif` keyword arguments.
        """
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if min_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) < Version(min_torch))
            reasons.append(f"torch>={min_torch}")

        if max_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) >= Version(max_torch))
            reasons.append(f"torch<{max_torch}")

        if min_python:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if requires_file:
            if isinstance(requires_file, str):
                requires_file = [requires_file]
            conditions.extend([not Path(file).exists() for file in requires_file])
            reasons.append(f"requires file={','.join(requires_file)}")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )


def check_grad_zero(model: torch.nn.Module):
    """check if the gradients of the model are zero"""
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.allclose(param.grad, torch.zeros_like(param.grad))
