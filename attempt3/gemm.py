import argparse
from typing import Callable, Tuple, Type
import math
import cuda.bindings.driver as cuda

import torch
import statistics

import cutlass
from cutlass import Boolean, Int32, const_expr
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait, PipelineState, PipelineUserType
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils

from tile_scheduler import SimpleTileSchedulerArguments, SimpleTileScheduler, RasterOrder
from cute_dsl_utils import ParamsBase
from functools import partial
from my_utils import tma_get_copy_fn, make_smem_layout_epi

THREADS_PER_WG = 128

@cute.jit
def print0(x):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.print_tensor(x)
    else:
        if tidx == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.printf(x)

@cute.jit
def printwg(x):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx%128 == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.print_tensor(x)
    else:
        if tidx%128 == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.printf(x)

class GemmSM90:
    def __init__(
        self,
        tile_shape_mn: Tuple[int, int],
        epi_tile_mn: Tuple[int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        atom_layout_mn: Tuple[int, int],
        ab_stage: int = 2,
        epi_stage: int = 2,
        raster_order: RasterOrder = RasterOrder.AlongN,
        reuse_ab: bool = True,
        is_persistent: bool = False, # dummy for now
        ):
        self.acc_dtype = cutlass.Float32
        self.raster_order = raster_order
        self.scheduler_group_size = Int32(8)
        self.cluster_shape_mnk = cluster_shape_mnk
        self.cluster_layout_mnk = None
        self.cta_tile_shape_mnk = (*tile_shape_mn, -1) # K-dim decided later
        tile_M, tile_N = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]

        # Atom layout
        assert atom_layout_mn[0] in [1, 2, 3] and atom_layout_mn[1] in [1, 2]
        assert atom_layout_mn[0] == 1 or tile_M % (atom_layout_mn[0] * 64) == 0
        assert atom_layout_mn[1] == 1 or tile_N % (atom_layout_mn[1] * 32) == 0
        self.atom_layout_mnk = (*atom_layout_mn, 1)

        # Multicast
        self.mcast_ctas_a = self.cluster_shape_mnk[1] # how many times we need to multicast
        self.mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.mcast_ctas_a > 1
        self.is_b_mcast = self.mcast_ctas_b > 1

        # Kernel config
        self.occupancy = 1
        self.mma_warpgroups = math.prod(self.atom_layout_mnk)
        assert self.mma_warpgroups in [1, 2, 3]
        self.threads_per_cta = (self.mma_warpgroups + 1) * THREADS_PER_WG
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes("sm_90")
        self.ab_load_warp_id = self.mma_warpgroups * 4
        
        # Registers
        regs_per_thread = math.prod(self.cta_tile_shape_mnk[:2]) // (
            math.prod(self.atom_layout_mnk) * THREADS_PER_WG
        ) # C tile shape / num threads. This is just a heuristic
        heavy_register_pressure = regs_per_thread >= 208
        self.num_regs_load, self.num_regs_mma = (40, 232) if not heavy_register_pressure else (24, 140)

        self.ab_stage = ab_stage
        self.epi_stage = epi_stage # NO EPI STAGING

        # runtime stuff
        self.a_dtype, self.b_dtype, self.c_dtype = None, None, None
        self.a_layout, self.b_layout = None, None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024
        self.tma_ab_load_bytes = 0

        # New epilogue
        self.epi_stage = epi_stage
        self.epi_tile_mn = epi_tile_mn
        self.reuse_ab = reuse_ab
        self.epi_smem_layout_staged = None
        self.epi_smem_size = 0

        # Persistent(Future)
        self.is_persistent = is_persistent

        # Checks
        assert not (self.reuse_ab and self.is_persistent), "Persistent kernel can't reuse AB for epilogue"

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda.CUstream):
        # Populate fields
        self.populate_dtypes_and_layouts(a, b, c)
        self.populate_mma_atom()
        self.populate_smem_layouts()
        self.populate_shared_storage()

        # EPILOGUE
        tma_atom_d, tma_tensor_d = self._get_tma_epi_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile_mn,
        )

        # Get TMA tensors and atoms
        self.tma_ab_load_bytes = 0
        tma_atom_a, tma_tensor_a = self._get_tma_load_and_tensors_incr_bytes(a, self.a_smem_layout_staged, (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]), self.mcast_ctas_a, self.a_dtype)
        tma_atom_b, tma_tensor_b = self._get_tma_load_and_tensors_incr_bytes(b, self.b_smem_layout_staged, (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]), self.mcast_ctas_b, self.b_dtype)
        tensor_c = c # no epilogue, so this works

        # Get Grid
        ts_args = self.get_tile_scheduler_args(a, b, c)
        ts_params = SimpleTileScheduler.to_underlying_arguments(ts_args)
        grid = SimpleTileScheduler.get_grid_shape(ts_params)
        # cute.printf(grid)
        self.kernel(
            tma_atom_a, tma_atom_b,
            tma_tensor_a, tma_tensor_b, tensor_c,
            self.tiled_mma,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
            ts_params,
            self.cluster_layout_mnk,
            self.epi_smem_layout_staged,
            tma_atom_d, tma_tensor_d
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], cluster=self.cluster_shape_mnk, stream=stream) # min_blocks_per_mp=1 only if kernel is large
    
    @cute.kernel
    def kernel(self,
               tma_atom_a: cute.CopyAtom,
               tma_atom_b: cute.CopyAtom,
               mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
               tiled_mma: cute.TiledMma,
               a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout,
               tile_sched_params: ParamsBase,
               cluster_layout_mnk: cute.Layout,
               epi_smem_layout: cute.ComposedLayout,
               epi_copy: cute.CopyAtom, # S2G
               epi_mC: cute.Tensor, # GMEM TMA tensor for storing
               ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.ab_load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(storage.mainloop_pipeline_barriers.data_ptr(), cute.make_layout((1, *cluster_layout_mnk.shape)))
        pipeline_init_arrive()
        pipeline_init_wait() # you can move this elsewhere

        # SMEM tensor creation
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

        # Pointer for epilogue
        sD = None
        if cutlass.const_expr(self.reuse_ab):
            sD_ptr = cute.recast_ptr(sA.iterator, epi_smem_layout.inner, dtype=self.c_dtype)
            sD = cute.make_tensor(sD_ptr, epi_smem_layout.outer)
        else:
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)

        tile_scheduler = SimpleTileScheduler.create(tile_sched_params)


        # TODO something with the barrier is wrong
        if warp_idx >= self.ab_load_warp_id: # Producer
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
            if warp_idx == self.ab_load_warp_id: # TMA warp
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)

                # Multicast mask: slices along a mode to specify the CTAs to cast to
                a_mcast_mask = cute.make_layout_image_mask(cluster_layout_mnk, block_in_cluster_coord_mnk, mode=1)
                b_mcast_mask = cute.make_layout_image_mask(cluster_layout_mnk, block_in_cluster_coord_mnk, mode=0)
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

                # I don't need to worry about schedulerWarp
                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
                while work_tile.is_valid_tile:
                    # do work
                    tile_coord_mnkl = work_tile.tile_idx

                    # NOTE this part ignores the L dimension, just do single GEMM for now
                    gA_mk = cute.local_tile(mA, cute.select(self.cta_tile_shape_mnk, [0, 2]), (tile_coord_mnkl[0], None))
                    gB_nk = cute.local_tile(mB, cute.select(self.cta_tile_shape_mnk, [1, 2]), (tile_coord_mnkl[1], None))
                    k_iters = cute.size(gA_mk, mode=[2]) # M, K, restK

                    # there's a good utility function in quack > copy_utils.py > tma_get_copy_fn
                    # TODO I should copy it tbh
                    copy_a, _, _ = tma_get_copy_fn(tma_atom_a, 
                                                   block_in_cluster_coord_mnk[1], # Where is your cluster in the multicast
                                                   cute.make_layout(cute.slice_(cluster_layout_mnk, (0, None, 0)).shape), 
                                                   gA_mk, sA, mcast_mask=a_mcast_mask)
                    copy_b, _, _ = tma_get_copy_fn(tma_atom_b,
                                                   block_in_cluster_coord_mnk[0],
                                                   cute.make_layout(cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape),
                                                   gB_nk, sB, mcast_mask=b_mcast_mask)

                    ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
                    
                    self.produce_mainloop(k_iters, copy_a, copy_b, ab_pipeline, ab_producer_state) # TODO
                    tile_scheduler.fetch_next_work()
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                # TODO: if this was persistent, you may need to advance to next work here
                # ab_pipeline.producer_tail(ab_producer_state) # this is NOT supposed to be here, that was the error

        if warp_idx < self.ab_load_warp_id: # Consumer
            cute.arch.warpgroup_reg_alloc(self.num_regs_mma)
            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx // THREADS_PER_WG)
            
            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage) # TODO

            thr_mma = tiled_mma.get_slice(tidx) # TODO why do they want warp_group_thread_layout(warp_group_idx)? how is it used?

            # TODO work out what make_fragment and partition are doing
            tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
            tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
            acc_shape = tiled_mma.partition_shape_C(
                cute.select(self.cta_tile_shape_mnk, mode=[0, 1])
            )
            accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                tile_coord_mnk = (work_tile.tile_idx[0], work_tile.tile_idx[1], work_tile.tile_idx[2])
                # TODO call mma()
                gA_mk = cute.local_tile(mA, cute.select(self.cta_tile_shape_mnk, [0, 2]), (tile_coord_mnk[0], None))
                k_iters = cute.size(gA_mk, mode=[2]) # m, k, restK

                # You have to return this, it doesn't let you modify a var inside a diff scope or smth
                # SSA -- modifying a value means reassigning it, so you can't go into this fn scope and only modify the value there
                # you have to return it
                ab_consumer_state, tiled_mma = self.consume_mainloop(k_iters, tiled_mma, accumulators, ab_pipeline, ab_consumer_state, tCrA, tCrB)

                # Epilogue ##################################################

                self.epilogue(tiled_mma, epi_mC, epi_copy, sD, accumulators, tile_coord_mnk, tidx, warp_idx)

                # no need to fetch -- producer does that
                tile_scheduler.advance_to_next_work() # this just does some arriving I think, if this was an actual tile scheduler
                work_tile = tile_scheduler.get_current_work()
            
            # no need to producer tail, the producer handles that since it contains the update warp
        return

    # Main stuff
    # -----------------------------
    @cute.jit
    def epilogue(self, tiled_mma, epi_mC, epi_copy, sD, accumulators, tile_coord_mnk, tidx, warp_idx):
        # You can just spam create copy atoms, layouts etc. since it won't be in the final compiled product.

        # NOTE other gemm examples do a cluster arrive/wait, not sure why
        # We use a NamedBarrier since we can't syncthreads(only want to sync consumers)
        epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(1),
            num_threads=self.mma_warpgroups * 4 * cute.arch.WARP_SIZE
        )
        if const_expr(self.reuse_ab):
            epilogue_barrier.arrive_and_wait()
        
        # NOTE: In cutlass example, they have multiple copy atoms and stuff, maybe this helps support alternate dtypes, not sure.
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(),
                4,
            ),
            self.c_dtype,
        )
        tiled_copy_r2s = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

        # gC_mnl stores where our output tile should be
        gC_mnl = cute.local_tile(epi_mC, self.cta_tile_shape_mnk, tile_coord_mnk, proj=(1, 1, None))
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD) # sD follows epi_smem_layout (m, n, stages)?
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)

        # Need to make accumulators that represents one stage of the epilogue, tRS_rAcc is all stages
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sD)) # registers needed for one epi tile
        # register layout, but for one stage of the epilogue
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        # sD has 4 stages, gD has 8 indices
        sepi_for_tma_partition = cute.group_modes(sD, 0, 2)
        tCgC_for_tma_partition = cute.zipped_divide(gC_mnl, self.epi_tile_mn) # this just happens to be the right shape
        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            epi_copy,
            0,
            cute.make_layout(1),
            sepi_for_tma_partition,
            tCgC_for_tma_partition,
        )

        epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
        epi_tile_shape = tCgC_for_tma_partition.shape[1] # the layout of epi tiles
        epi_tile_layout = cute.make_layout(
            epi_tile_shape, stride=(epi_tile_shape[1], 1)
        )

        # this sets up a bulk wait pipeline
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=c_producer_group,
        )

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                # Take a slice of the accumulators
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
            
            # Type conversion
            tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.c_dtype)
            acc_vec = tRS_rD.load()
            tRS_rD_out.store(acc_vec.to(self.c_dtype))

            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            # R2S stmatrix
            cute.copy(
                tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
            )
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            epilogue_barrier.arrive_and_wait() # Make sure stmatrix is done

            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx) # literally like (0, 0) to (7, 0)
            if warp_idx == 0:
                cute.copy(
                    epi_copy,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                c_pipeline.producer_commit() # commit_group
                c_pipeline.producer_acquire() # wait_group(stages-1)
            epilogue_barrier.arrive_and_wait() # Don't start next stmatrix yet
        
        if warp_idx == 0:
            c_pipeline.producer_tail() # wait_group(0)

    @cute.jit
    def produce_mainloop(self, k_iters: Int32, copy_a: Callable, copy_b: Callable, pipe: pipeline.PipelineAsync, state: pipeline.PipelineState):
        for _ in cutlass.range(k_iters, unroll=1, unroll_full=False):
            pipe.producer_acquire(state) # wait empty arrive full
            mbar = pipe.producer_get_barrier(state)
            copy_a(state.count, state.index, tma_bar_ptr=mbar)
            copy_b(state.count, state.index, tma_bar_ptr=mbar)
            pipe.producer_commit(state)
            state.advance()
        pipe.producer_tail(state)
        return state

    @cute.jit
    def consume_mainloop(self, k_iters: Int32, tiled_mma: cute.TiledMma, accumulators: cute.Tensor, pipe: pipeline.PipelineAsync, read_state: pipeline.PipelineState, tCrA: cute.Tensor, tCrB: cute.Tensor):
        # I have to add prologue MMAs if I want to pipeline MMAs
        K_PIPE_MMAS = 1
        release_state = read_state.clone()
        num_prologue_mma = min(K_PIPE_MMAS, k_iters)
        num_k_blocks = cute.size(tCrA, mode=[2])
        # print(tiled_mma)
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        # print(tCrA)
        # print(tCrB)
        for _ in cutlass.range(num_prologue_mma):
            pipe.consumer_wait(read_state)
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (None, None, k_block_idx, read_state.index)
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]
                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            read_state.advance()

        for _ in cutlass.range(k_iters, unroll=1, unroll_full=False):
            pipe.consumer_wait(read_state)
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (None, None, k_block_idx, read_state.index)
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]
                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(K_PIPE_MMAS)
            pipe.consumer_release(release_state) # this can lag behind read_state
            read_state.advance()
            release_state.advance()
        cute.nvgpu.warpgroup.wait_group(0)
        return read_state, tiled_mma

    # More runtime stuff
    # -----------------------------
    def tma_partition(self, cluster_coord, tma_atom: cute.CopyAtom, sMatrix: cute.Tensor, gMatrix: cute.Tensor):
        s_tma = cute.group_modes(sMatrix, 0, 2)
        g_tma = cute.group_modes(gMatrix, 0, 2)

        # (TMA, pipe_stages) and (TMA, k)
        shared_layout, global_layout = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            cluster_coord,
            s_tma,
            g_tma,
        )
        return shared_layout, global_layout

    @cute.jit
    def make_ab_pipeline(self, mbar_ptr: cute.Pointer, cta_layout_vmnk: cute.Layout):
        num_producers = 1
        mcast_size = self.mcast_ctas_a + self.mcast_ctas_b - 1
        num_warps = self.mma_warpgroups * 4
        num_consumers = num_warps * mcast_size # IMPORTANT!!!
        # print(num_consumers)
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_producers)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_consumers)
        # reminder: CTA layout is only used for syncing
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
            num_stages=self.ab_stage,
            tx_count=self.tma_ab_load_bytes,
            producer_group=producer_group,
            consumer_group=consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
        )

    def get_tile_scheduler_args(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        batch_size = mC.shape[2] if cute.rank(mC.layout) == 3 else 1
        problem_shape_ntile_mnl = (
            cute.ceil_div(mA.shape[0], self.cta_tile_shape_mnk[0]),
            cute.ceil_div(mB.shape[0], self.cta_tile_shape_mnk[1]),
            batch_size,
        )
        tile_sched_args = SimpleTileSchedulerArguments(
            problem_shape_ntile_mnl,
            self.raster_order,
            self.scheduler_group_size,
            self.cluster_shape_mnk,
        )
        return tile_sched_args

    def _get_tma_load_and_tensors_incr_bytes(self, global_tensor: cute.Tensor, smem_layout_staged: cute.ComposedLayout, smem_tile: tuple[int, int], mcast_dim: bool, dtype: Type[cutlass.Numeric]) -> tuple[cute.CopyAtom, cute.Tensor]:
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp() if mcast_dim == 1 else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        # TMA tensor is just like a normal tensor but with 1@1, 1@0 etc. so the TMA can consume it
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            global_tensor,
            smem_layout,
            smem_tile, # CTA tiler
            num_multicast=mcast_dim
        )
        self.tma_ab_load_bytes += cute.size_in_bytes(dtype, smem_layout)
        return tma_atom, tma_tensor
    
    @staticmethod
    def _get_tma_epi_atoms_and_tensors(
            tensor_d: cute.Tensor,
            epi_smem_layout_staged: cute.ComposedLayout,
            epi_tile: Tuple[int, int],
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        # ASSUME we're just storing for now
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        d_cta_v_layout = cute.composition(cute.make_identity_layout(tensor_d.shape), epi_tile) # change it to TMA-usable format with 1@0, 1@1 etc.

        op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # Tiles D with d_cta_v_layout, prepares to copy to d
        tma_atom_d, tma_tensor_d = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, tensor_d, epi_smem_layout, d_cta_v_layout
        )
        return tma_atom_d, tma_tensor_d

    # Easy Population Helpers
    # -------------------------------------
    def populate_dtypes_and_layouts(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self.cluster_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
    
    @cute.jit
    def populate_mma_atom(self):
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.cta_tile_shape_mnk[1])
        )
        mma_k = 16
        mma_inst_tile_k = 4
        self.cta_tile_shape_mnk = (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], mma_k * mma_inst_tile_k)
    
    def populate_smem_layouts(self):
        self.a_smem_layout_staged, self.b_smem_layout_staged, self.epi_smem_layout_staged = self._get_smem_layouts(self.cta_tile_shape_mnk, self.a_dtype, self.a_layout, 
                                                                                                                   self.b_dtype, self.b_layout, self.ab_stage,
                                                                                                                   self.c_dtype, self.c_layout, self.epi_tile_mn, self.epi_stage)
        if not self.reuse_ab:
            self.epi_smem_size = cute.cosize(self.epi_smem_layout_staged)

    @staticmethod
    def _get_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: Type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
        d_dtype: Type[cutlass.Numeric], # NEW, for epilogue. Use D to refer to the output matrix
        d_layout: utils.LayoutEnum,
        epi_tile: Tuple[int, int],
        epi_stage: int,
    ):
        a_smem_layout = sm90_utils.make_smem_layout_a(
            a_layout, tile_shape_mnk, a_dtype, ab_stage
        )

        b_smem_layout = sm90_utils.make_smem_layout_b(
            b_layout, tile_shape_mnk, b_dtype, ab_stage
        )

        epi_smem_layout_staged = make_smem_layout_epi(d_dtype, d_layout, epi_tile, epi_stage)
        return a_smem_layout, b_smem_layout, epi_smem_layout_staged
    
    @cute.jit
    def populate_shared_storage(self):
        @cute.struct
        class SharedStorage:
            mainloop_pipeline_barriers: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)], self.buffer_align_bytes]
            sD: cute.struct.Align[cute.struct.MemRange[self.c_dtype, self.epi_smem_size], self.buffer_align_bytes]

        self.shared_storage = SharedStorage

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["debug", "speed", "ncu"])
    args = parser.parse_args()
    IS_NCU = args.mode == 'ncu'
    IS_DEBUG = args.mode == 'debug'
    IS_SPEED = args.mode == 'speed'

    m, n, k = 4096, 4096, 4096
    flops = 2 * m * n * k

    def get_tflops(time_ms):
        return (flops / (time_ms / 1e3)) / 1e12

    a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    ref = a @ b.t()
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    a_cute, b_cute, c_cute = [convert_from_dlpack(x) for x in (a, b, c)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    gemm = GemmSM90(tile_shape_mn=(128, 256), 
                    epi_tile_mn=(128, 32),
                    cluster_shape_mnk=(2, 1, 1), 
                    atom_layout_mn=(2, 1),
                    ab_stage=3,
                    reuse_ab=False)
    compiled_gemm = cute.compile(gemm, a_cute, b_cute, c_cute, current_stream)
    compiled_gemm(a_cute, b_cute, c_cute, current_stream)
    print(ref)
    print(c)

    if IS_DEBUG:
        n_incorrect = c.numel() - ((c - ref).abs() < 0.001).sum()
        print('n_incorrect :', n_incorrect)

    def profile_ms(op, repeats=30):

        clear_cache = torch.cuda.empty_cache
        clear_cache()

        # warmup
        op()
        torch.cuda.synchronize()

        start = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

        for i in range(repeats):
            # clear_cache()
            start[i].record()
            op()
            end[i].record()

        torch.cuda.synchronize()
        return statistics.median([s.elapsed_time(e) for s, e in zip(start, end)])

    @torch.compile
    def torch_gemm():
        return a @ b.t()

    if IS_SPEED:
        my_ms = profile_ms(lambda: compiled_gemm(a_cute, b_cute, c_cute, current_stream))
        other_ms = profile_ms(torch_gemm)
        print(f'{my_ms=}, {other_ms=}')
        my_flops, other_flops = get_tflops(my_ms), get_tflops(other_ms)
        print(f'{my_flops=}, {other_flops=}')