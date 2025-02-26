import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    assert in_channels == in_channels_

    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Define tiling parameters
    input_chan_block_size = nl.tile_size.pmax
    num_input_chan_tiles = in_channels // input_chan_block_size

    output_chan_block_size = input_chan_block_size
    num_output_chan_tiles = out_channels // output_chan_block_size

    num_output_height_tiles = (
        out_height + 1
    ) // 2  # +1 to ensure we have at least one tile

    tile_height = out_height // num_output_height_tiles

    # reshape input and weights
    X = X.reshape(
        (
            batch_size,
            num_input_chan_tiles,
            input_chan_block_size,
            input_height,
            input_width,
        )
    )
    bias = bias.reshape((num_output_chan_tiles, output_chan_block_size, 1))

    bias_buf = nl.ndarray(
        (num_output_chan_tiles, nl.par_dim(output_chan_block_size), 1),
        dtype=bias.dtype,
        buffer=nl.sbuf,
    )
    for out_chan_tile in nl.affine_range(num_output_chan_tiles):
        bias_buf[out_chan_tile] = nl.load(bias[out_chan_tile])

    W = W.reshape(
        (
            num_output_chan_tiles,
            output_chan_block_size,
            num_input_chan_tiles,
            input_chan_block_size,
            filter_height,
            filter_width,
        )
    )
    weights = nl.ndarray(
        (
            num_output_chan_tiles,
            nl.par_dim(output_chan_block_size),
            num_input_chan_tiles,
            input_chan_block_size,
            filter_height,
            filter_width,
        ),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )
    for out_chan_tile in nl.affine_range(num_output_chan_tiles):
        weights[out_chan_tile] = nl.load(W[out_chan_tile])

    weights_temp = nl.ndarray(
        (
            filter_height,
            filter_width,
            num_output_chan_tiles,
            num_input_chan_tiles,
            nl.par_dim(output_chan_block_size),
            input_chan_block_size,
        ),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )
    weights_transposed = nl.ndarray(
        (
            filter_height,
            filter_width,
            num_output_chan_tiles,
            num_input_chan_tiles,
            nl.par_dim(input_chan_block_size),
            output_chan_block_size,
        ),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )
    for out_chan_tile in nl.affine_range(num_output_chan_tiles):
        for in_chan_tile in nl.affine_range(num_input_chan_tiles):
            for height_idx in nl.affine_range(filter_height):
                for width_idx in nl.affine_range(filter_width):
                    weights_temp[
                        height_idx, width_idx, out_chan_tile, in_chan_tile, :, :
                    ] = nl.copy(
                        weights[
                            out_chan_tile, :, in_chan_tile, :, height_idx, width_idx
                        ]
                    )
                    weights_transposed[
                        height_idx, width_idx, out_chan_tile, in_chan_tile, :, :
                    ] = nisa.nc_transpose(
                        weights_temp[
                            height_idx, width_idx, out_chan_tile, in_chan_tile, :, :
                        ]
                    )

    # Loop over each batch image
    for batch_idx in nl.affine_range(batch_size):
        for height_tile in nl.affine_range(num_output_height_tiles):
            img_sbuf = nl.ndarray(
                (
                    num_input_chan_tiles,
                    nl.par_dim(input_chan_block_size),
                    tile_height + filter_height - 1,
                    input_width,
                ),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )
            for in_chan_tile in nl.affine_range(num_input_chan_tiles):
                img_sbuf[in_chan_tile] = nl.load(
                    X[
                        batch_idx,
                        in_chan_tile,
                        :,
                        height_tile * tile_height : (height_tile + 1) * tile_height
                        + filter_height
                        - 1,
                        :,
                    ]
                )

            for out_chan_tile in nl.affine_range(num_output_chan_tiles):
                curr_output = nl.ndarray(
                    (nl.par_dim(output_chan_block_size), tile_height, out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )

                for height_offset in nl.affine_range(tile_height):
                    accum = nl.zeros(
                        (output_chan_block_size, out_width),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    # Accumulate partial sums
                    for filter_h in nl.affine_range(filter_height):
                        for filter_w in nl.affine_range(filter_width):
                            for in_chan_tile in nl.affine_range(num_input_chan_tiles):
                                accum += nl.matmul(
                                    weights_transposed[
                                        filter_h,
                                        filter_w,
                                        out_chan_tile,
                                        in_chan_tile,
                                        :,
                                        :,
                                    ],
                                    img_sbuf[
                                        in_chan_tile,
                                        :,
                                        height_offset + filter_h,
                                        filter_w : filter_w + out_width,
                                    ],
                                    transpose_x=True,
                                )
                    # Add bias
                    accum = nisa.tensor_scalar(accum, nl.add, bias_buf[out_chan_tile])
                    curr_output[:, height_offset, :] = nl.copy(accum)

                nl.store(
                    X_out[
                        batch_idx,
                        out_chan_tile
                        * output_chan_block_size : (out_chan_tile + 1)
                        * output_chan_block_size,
                        height_tile * tile_height : (height_tile + 1) * tile_height,
                        :,
                    ],
                    value=curr_output[...],
                )

    return X_out
