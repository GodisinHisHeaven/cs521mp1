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

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    assert in_channels % 128 == 0

    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Define tiling parameters
    input_chan_block_size = nl.tile_size.pmax
    num_input_chan_tiles = in_channels // input_chan_block_size

    output_chan_block_size = nl.tile_size.pmax
    num_output_chan_tiles = out_channels // output_chan_block_size

    output_height_block_size = 2
    num_output_height_tiles = (
        out_height + output_height_block_size - 1
    ) // output_height_block_size

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
    for j in nl.affine_range(num_output_chan_tiles):
        bias_buf[j] = nl.load(bias[j])

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
    for i in nl.affine_range(num_output_chan_tiles):
        weights[i] = nl.load(W[i])

    # Loop over each batch image
    for b in nl.affine_range(batch_size):
        for a in nl.affine_range(num_output_height_tiles):
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
            for i in nl.affine_range(num_input_chan_tiles):
                img_sbuf[i] = nl.load(
                    X[
                        b,
                        i,
                        :,
                        a * tile_height : (a + 1) * tile_height + filter_height - 1,
                        :,
                    ]
                )

            for i in nl.affine_range(num_output_chan_tiles):
                curr_output = nl.ndarray(
                    (nl.par_dim(output_chan_block_size), tile_height, out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )

                for curr_row in nl.affine_range(tile_height):
                    accum = nl.zeros(
                        (output_chan_block_size, out_width),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    # Accumulate partial sums
                    for fh in nl.affine_range(filter_height):
                        for fw in nl.affine_range(filter_width):
                            for j in nl.affine_range(num_input_chan_tiles):
                                accum += nl.matmul(
                                    weights[i, :, j, :, fh, fw],
                                    img_sbuf[j, :, curr_row + fh, fw : fw + out_width],
                                    transpose_x=False,
                                )
                    # Add bias
                    accum = nisa.tensor_scalar(accum, nl.add, bias_buf[i])
                    curr_output[:, curr_row, :] = nl.copy(accum)

                nl.store(
                    X_out[
                        b,
                        i * output_chan_block_size : (i + 1) * output_chan_block_size,
                        a * tile_height : (a + 1) * tile_height,
                        :,
                    ],
                    value=curr_output[...],
                )

    return X_out
