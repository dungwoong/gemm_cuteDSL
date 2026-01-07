# Adding Epilogue

## Changes
- Gemm constructor, add epi tile and epi stage
- Remember, k=64. Let's just make the epi tile (128, 32) with 2 stages for now, and let's make it overlap with other stuff
- Added a reuse_ab flag
- Quick analysis: (128, 64) + (256, 64) * 2(stages) * 2(bytes) is 98KB. Add (128, 32) * 2(stages) * 2(B) = 16384. SM100 is 227KB(look at `cutlass.utils.get_smem_capacity_in_bytes`). PyTorch kernel requests 213KB/block
- Need to change how we allocate SMEM
    - Quack gemm uses has_D and is_persistent as their flags. sD is their epilogue buffer, and `epi_smem_size=0` if they choose to reuse A and B
    - If reusing, they do a recast_ptr and make_tensor like on line 633 in quack gemm (`sD_ptr = cute.recast_ptr(sA.iterator, epi_smem_layout.inner, dtype=self.d_dtype)`)
    - Otherwise, they just use has_D
    - Done, I copied their code in
- Need to allocate epi if not reusing AB
    - Quack does it on line 1888 `quack_sm90_utils.make_smem_layout_epi`
    - Done, and also change `epi_smem_size` based on what goes on
- Added `is_persistent` and startup check for `reuse_AB` based on it
- Added _make_tma_epi_atoms_and_tensors to handle the S2G, 
- TODO figure out how to retile accumulators and handle R2S, create the copy and just print what's in the epi SMEM to make sure things match for now

## Number of epilogue warps
- just all warps that participate in MMA, for pingpong it should just be one WG
- this is used to decide number of threads that should arrive at the epilogue barrier(just a barrier with an id)
- helps all warps wait in case you're reusing AB smem. You also wait before and after doing tma store(why after? do they not use a barrier for stages here?)
- epi store pipeline is created in make_epi_store_pipeline. It's a PipelineTmaStore, and there's **NO CONSUMER AGENT**.
    - You can only arrive/wait at the sync object
- is tma warp is true for warp idx 0(or 4 if pingpong)
- if we look at construction of make the tma store object, look at `_make_sync_object` we see that we get this TmaStoreFence. I'm just gonna assume that is a barrier that's ready if the TMA store is triggered and the data is no longer needed or smth.
- If we actually look at TmaStoreFence(pipeline/helpers.py) we can see that it's just a commit group and wait group object so when we call commit and acquire, it's actually calling the commit group and wait group. Ok that makes sense.
- We can actually see that when TmaStore is created it ignores the producer_group which is the cooperative group, it just expects one thread to call the stuff anyways(in `_make_sync_object`)

So TLDR the PipelineTmaStore is basically an object meant to be called with one thread that just calls tma commit group and wait group. It's just confusing because pipelines are typically setup differently, so this one has no consumer agent, and should implicitly be called with one thread.

```python
def tma_store_fn(src_idx, dst_idx):
    # Fence and barrier to make sure shared memory store is visible to TMA store
    cute.arch.fence_proxy(
        cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
    )
    epilogue_barrier.arrive_and_wait() # sync so we know all data is in smem now
    
    ...
    
    if_generate(is_tma_warp, lambda: epi_store_pipeline.producer_commit()) # commit group
    if_generate(is_tma_warp, lambda: epi_store_pipeline.producer_acquire()) # wait group(so next stage is ready to go, we can move on)
    epilogue_barrier.arrive_and_wait() # sync so other threads don't move on too early
```

## Now we just need the tiles
- [NEED WORK] `epilog_smem_store_and_partition` creates a r2s copy, tRS_rD, tRS_sD
- get smem store op
- they retile acc
- extra work if has_C but we can ignore for now
- WAITING on compute canada

## Dataflow
This is using the CUTLASS example
- trs_rAcc > tRs_rD(single tile of acc)
- then tRs_rD_out(cast dtype)
- then stmatrix to tRS_sD
- reinterpret that as something else and then TMA to GMEM
- not really much pipelining on our end, except that we don't wait for the TMA to finish before starting next one, we wait num_stages - 1

## Epi tile layout
This is using the CUTLASS example
This helps them figure out where we're going in GMEM using get_hier_coord
- actually, they use TMA partition to get bSG_sD and gD
- you then have the `epi_tile_layout` which is just row major I guess

## Making the TMA atom
- `cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionOp.ADD)` helps you do AB+C, so it adds to whatever's already there
- [docs](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor) You can use `.bulk_group` instead of `.mbarrier::...` for stores from shared to global. CuteDSL mentions they use the `.tile` load mode

## Barriers
- We have epi warps. If not persistent, all these epilogue warps must sync up since SMEM for tensor A is reused.
- Otherwise we arrive and wait at a barrier AFTER stmatrix to get ready for TMA store.
- You CANNOT use syncthreads since that expects ALL threads(including producer)