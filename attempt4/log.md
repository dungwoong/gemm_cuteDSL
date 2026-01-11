Add persistent kernel with tile scheduler, no dynamic scheduling though

## Tile Scheduler
- add `is_persistent` arg
- when getting grid shape, if it's persistent you pass in `max_active_clusters`
- map cta coords still just gives `cluster_id` to 2d coords
- add current work linear idx to Tile Scheduler
- added `max_active_clusters` and put that into the gemm

## Debugging
- producer tail comes AFTER the entire persistent kernel, that makes logical sense but don't forget!
- make sure if you run a function that mutates the pipeline state, return that pipeline state so that mutation is actually assigned. Matters a lot in persistent kernels
- I made a dumb mistake where I was initializing pipeline state INSIDE the persistent while loop. That was resetting pipeline state everytime.