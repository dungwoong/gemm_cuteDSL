577TFLOPS vs Torch is like 780TFLOPs, although the timing mechanism isn't that thorough.

# Kernel
- Warp specialization
- WGMMA and TMA with multicast
- A scheduler setup, although the scheduler does not actually do anything

# Add Clusters
- cta layout mnk is (cluster_shape_mn, 1)
- mcast_ctas_a = number of CTAs along the B dim, which would be the number of multicasts to do
- you make an mcast mask for loads, 0 if no multicast
- when you cute.copy, you add that mcast mask
- when making TMA atoms, you set them to CopyBulkTensorTileG2SMulticastOp if it's multicast

# Gemm attempt issues encountered
- Pipelines, just make sure right number of arrivals and don't add extra calls to e.g. `tail()`
- make sure mcast mask is 0 if you don't use it
- If you modify something inside a function you have to return it because mlir is SSA so every modify is a reassignment. That's why functions will return the state they modify in other examples.
- When multicasting, all warps that we multicast to must arrive at the barrier. There's a bit of math to calculate how many warps that is
- some syntax errors but that's fine