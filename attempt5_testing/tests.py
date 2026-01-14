import argparse
from typing import Callable, Tuple, Type
import math
import cuda.bindings.driver as cuda

import torch
from triton import runtime
import functools
import statistics
import multiprocessing as mp

from cutlass import cute
from cutlass.cute.runtime import from_dlpack

import pytest
import io
import sys
import traceback

from gemm import GemmSM90

torch.manual_seed(42)

# I don't have pytest on my sif so I'll try something manual

def _get_tflops(m, n, k, time_ms):
    flops = 2 * m * n * k
    return (flops / (time_ms / 1e3)) / 1e12

def _get_abc(m, n, k):
    a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    return a, b, c

def profile_ms(op, repeats=30):

    clear_cache = functools.partial(
        runtime.driver.active.clear_cache,  # type: ignore[attr-defined]
        runtime.driver.active.get_empty_cache_for_benchmark(),  # type: ignore[attr-defined]
    )
    clear_cache()

    # warmup
    op()
    torch.cuda.synchronize()

    start = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        clear_cache()
        start[i].record()
        op()
        end[i].record()

    torch.cuda.synchronize()
    return statistics.median([s.elapsed_time(e) for s, e in zip(start, end)])

convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )

# just run pytest -s to collect outputs
# TODO this does not work for hung outputs on the GPU, so beware...
def _run_test_impl(gemm, m, n, k, tag):
    a, b, c = _get_abc(m, n, k)
    ref = a @ b.t()
    a_cute, b_cute, c_cute = [convert_from_dlpack(x) for x in (a, b, c)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled_gemm = cute.compile(gemm, a_cute, b_cute, c_cute, current_stream)
    compiled_gemm(a_cute, b_cute, c_cute, current_stream)
    assert torch.allclose(ref, c), f'Incorrect. Max abs diff: {torch.max((c - ref).abs()).item()}\n{ref}\n{c}'
    
    time_ms = profile_ms(lambda: compiled_gemm(a_cute, b_cute, c_cute, current_stream))
    print(f'\n[{tag}] t={time_ms}ms, TFLOPS={_get_tflops(m, n, k, time_ms)}')

def _run_test_impl_torch(m, n, k):
    tag = f'torch m{m}n{n}k{k}'
    a, b, c = _get_abc(m, n, k)
    bt = b.t()

    @torch.compile
    def fn(a_, bt_):
        return a_ @ bt_
    time_ms = profile_ms(lambda: fn(a, bt))
    print(f'\n[{tag}] t={time_ms}ms, TFLOPS={_get_tflops(m, n, k, time_ms)}')


def async_wrapper(queue, gemm, m, n, k, tag):
    buf = io.StringIO()
    old_out, old_error = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf, buf
    try:
        torch.cuda.init()
        _run_test_impl(gemm, m, n, k, tag)
        queue.put(('ok', buf.getvalue(), ""))
    except Exception as e:
        queue.put(('error', str(e), traceback.format_exc()))
    finally:
        sys.stdout, sys.stderr = old_out, old_error
    

def run_test(gemm, m, n, k, tag, timeout=30):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=async_wrapper, args=(q, gemm, m, n, k, tag))
    p.start()

    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        pytest.fail("Kernel hung") # TODO haven't tested this with actual hung gpu kernels, just time.sleep
    
    status, payload, logs = q.get(timeout=30)

    if status == 'error':
        pytest.fail(f"{payload}")
    print(f'{payload}, {logs}')


def test_pytorch():
    _run_test_impl_torch(4096, 4096, 4096)

def test_basic():
    gm = GemmSM90(tile_shape_mn=(128, 256), 
                    epi_tile_mn=(128, 32),
                    cluster_shape_mnk=(2, 1, 1), 
                    atom_layout_mn=(2, 1),
                    ab_stage=3,
                    reuse_ab=False,
                    is_persistent=True,
                    gemm_n_prologue=1)
    run_test(gm, 4096, 4096, 4096, 'gemm')

def test_multicast_dims():
    gm = GemmSM90(
        tile_shape_mn=(128, 256),
        epi_tile_mn=(128, 32),
        cluster_shape_mnk=(2, 2, 1),
        atom_layout_mn=(2, 1),
        ab_stage=3,
        reuse_ab=False,
        is_persistent=True,
        gemm_n_prologue=1,
    )
    run_test(gm, 4096, 4096, 4096, 'gemm_cluster22')

def test_atom_layout_horizontal():
    gm = GemmSM90(
        tile_shape_mn=(128, 256),
        epi_tile_mn=(64, 256), # need to cover entire n dim so we iterate the epi tile downwards
        # otherwise we'd have to redo the epilogue
        cluster_shape_mnk=(1, 1, 1),
        atom_layout_mn=(1, 2),
        ab_stage=3,
        reuse_ab=False,
        is_persistent=True,
        gemm_n_prologue=1,
    )
    run_test(gm, 4096, 4096, 4096, 'gemm_atom12')

def test_gemm_no_prologue():
    gm = GemmSM90(tile_shape_mn=(128, 256), 
                    epi_tile_mn=(128, 32),
                    cluster_shape_mnk=(2, 1, 1), 
                    atom_layout_mn=(2, 1),
                    ab_stage=3,
                    reuse_ab=False,
                    is_persistent=True,
                    gemm_n_prologue=0)
    run_test(gm, 4096, 4096, 4096, 'gemm_no_prologue_mma')

def test_gemm_reuse_ab():
    gm = GemmSM90(tile_shape_mn=(128, 256), 
                    epi_tile_mn=(128, 32),
                    cluster_shape_mnk=(2, 1, 1), 
                    atom_layout_mn=(2, 1),
                    ab_stage=3,
                    reuse_ab=True,
                    is_persistent=False,
                    gemm_n_prologue=0)
    run_test(gm, 4096, 4096, 4096, 'gemm_reuseab_no_persistent')

# run pytest -s tests.py to collect print outputs