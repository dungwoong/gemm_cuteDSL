# Adding tile layout (1, 2) instead of (2, 1)
- If you have (1, 2) your epilogue tile has to be 64x(cta_tile_n) so all warps can store on each step, and you iterate down the epilogue tile. Otherwise, we'd have to modify the epilogue so warps take turns storing or whatever
- After mathing out the stmatrix, I found when we populate mma we do 64x(cta_tile_n), giving that mma size to EACH mma, which is not what we want.
- so when make trivial tiled mma, do `tiler_mn=(64, self.cta_tile_shape_mnk[1] // self.atom_layout_mnk[1])`
- fixed!