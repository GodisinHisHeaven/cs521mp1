import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa



@nki.jit
def conv2d(X, W, bias):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape

    assert in_channels_ == in_channels, "Mismatch in input channels"
    assert bias.shape[0] == out_channels, "Mismatch in out_channels vs. bias"

    # Output dimensions
    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    # Fixed block sizes for all dimensions
    IC_BLOCK = 128
    OC_BLOCK = 128
    OH_BLOCK = 32
    OW_BLOCK = 32

    num_ic_tiles = in_channels // IC_BLOCK
    num_oc_tiles = out_channels // OC_BLOCK
    num_oh_tiles = (out_height + OH_BLOCK - 1) // OH_BLOCK
    num_ow_tiles = (out_width + OW_BLOCK - 1) // OW_BLOCK

    # Allocate output in HBM
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Loop over each batch image
    for b in nl.affine_range(batch_size):
        # Loop over blocks of 128 output channels
        for oc_t in nl.affine_range(num_oc_tiles):
            oc_start = oc_t * OC_BLOCK
            oc_end = (oc_t + 1) * OC_BLOCK  # Add this line to define oc_end

            # Add loops for output height/width tiles
            for oh_t in nl.affine_range(num_oh_tiles):
                oh_start = oh_t * OH_BLOCK
                curr_oh = min(OH_BLOCK, out_height - oh_start)

                for ow_t in nl.affine_range(num_ow_tiles):
                    ow_start = ow_t * OW_BLOCK
                    curr_ow = min(OW_BLOCK, out_width - ow_start)

                    # Always allocate fixed-size buffer
                    out_accum = nl.zeros(
                        shape=(OC_BLOCK, OH_BLOCK, OW_BLOCK),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )

                    # Accumulate partial sums
                    for fh in range(filter_height):
                        for fw in range(filter_width):
                            for ic_t2 in range(num_ic_tiles):
                                ic_start = ic_t2 * IC_BLOCK
                                ic_end = (ic_t2 + 1) * IC_BLOCK

                                # (A) Load entire plane of W => shape (OC_BLOCK, IC_BLOCK, filter_height, filter_width)
                                w_plane_sbuf = nl.ndarray(
                                    (OC_BLOCK, IC_BLOCK, filter_height, filter_width),
                                    dtype=W.dtype,
                                    buffer=nl.sbuf,
                                )
                                w_plane = nl.load(
                                    W[
                                        oc_start:oc_end,
                                        ic_start:ic_end,
                                        0:filter_height,
                                        0:filter_width,
                                    ]
                                )
                                w_plane_sbuf[...] = w_plane

                                # Sub-slice for (fh, fw) => shape (OC_BLOCK, IC_BLOCK)
                                w_slice_sbuf = nl.ndarray(
                                    (OC_BLOCK, IC_BLOCK), dtype=W.dtype, buffer=nl.sbuf
                                )
                                w_slice_sbuf[...] = w_plane_sbuf[:, :, fh, fw]

                                # Adjust output height loop with boundary check
                                for oh in nl.affine_range(
                                    curr_oh
                                ):  # Changed from OH_BLOCK
                                    actual_oh = oh_start + oh
                                    if actual_oh + fh >= input_height:
                                        continue

                                    # Load input row
                                    x_row_sbuf = nl.ndarray(
                                        (IC_BLOCK, input_width),
                                        dtype=X.dtype,
                                        buffer=nl.sbuf,
                                    )
                                    row_load = nl.load(
                                        X[
                                            b,
                                            ic_start:ic_end,
                                            actual_oh + fh,
                                            0:input_width,
                                        ]
                                    )
                                    x_row_sbuf[...] = row_load

                                    # Adjust output width loop with boundary check
                                    for ow in nl.affine_range(
                                        curr_ow
                                    ):  # Changed from OW_BLOCK
                                        actual_ow = ow_start + ow
                                        if actual_ow + fw >= input_width:
                                            continue
                                        # sub-slice (IC_BLOCK,1) from x_row_sbuf
                                        col_sbuf = nl.ndarray(
                                            (IC_BLOCK, 1), dtype=X.dtype, buffer=nl.sbuf
                                        )
                                        col_sbuf[...] = x_row_sbuf[
                                            :, (actual_ow + fw) : (actual_ow + fw + 1)
                                        ]

                                        # Multiply => (OC_BLOCK,1)
                                        prod = nl.matmul(
                                            w_slice_sbuf, col_sbuf, transpose_x=False
                                        )

                                        # Accumulate into out_accum[:, oh, ow]
                                        # We'll do a sub-slice in SBUF: shape (OC_BLOCK,1)
                                        px_out_sbuf = out_accum[
                                            :, oh, ow : ow + 1
                                        ]  # (OC_BLOCK,1)
                                        px_out_sbuf += prod

                    # Add bias for this tile
                    bias_sbuf = nl.load(bias[oc_start:oc_end])
                    for oh in nl.affine_range(OH_BLOCK):
                        for ow in nl.affine_range(OW_BLOCK):
                            px_out_sbuf = out_accum[:, oh, ow : ow + 1]  # (OC_BLOCK,1)
                            # We'll create a (OC_BLOCK,1) tile for bias
                            bcast_bias = nl.ndarray(
                                (OC_BLOCK, 1), dtype=bias.dtype, buffer=nl.sbuf
                            )
                            bcast_bias[...] = (
                                bias_sbuf  # shape(OC_BLOCK,) => assigned into shape(OC_BLOCK,1)
                            )
                            px_out_sbuf += bcast_bias

                    # When storing back, only store the valid region
                    valid_h = min(OH_BLOCK, out_height - oh_start)
                    valid_w = min(OW_BLOCK, out_width - ow_start)

                    nl.store(
                        X_out[
                            b,
                            oc_start : oc_start + OC_BLOCK,
                            oh_start : oh_start + valid_h,
                            ow_start : ow_start + valid_w,
                        ],
                        value=out_accum[:, :valid_h, :valid_w],
                    )

    return X_out
