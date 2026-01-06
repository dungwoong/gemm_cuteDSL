import cutlass
from cutlass import cute


def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    **kwargs,
):
    src_is_smem = cutlass.const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)

    s, g = cute.nvgpu.cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, cute.rank(smem_tensor) - 1),
        cute.group_modes(gmem_tensor, 0, cute.rank(gmem_tensor) - 1),
    )
    src, dst = (s, g) if src_is_smem else (g, s)

    def copy_tma(src_idx, dst_idx, **kwargs2):
        cute.copy(atom, src[None, src_idx], dst[None, dst_idx], **kwargs2, **kwargs)
    return copy_tma, s, g