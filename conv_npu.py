import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.jit
def conv2d(X, W, bias):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    # Calculate output dimensions
    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    TILE_IC = 128
    TILE_OC = 128
    output_tile_height = 2
    input_tile_height = output_tile_height + filter_height - 1
    n_tiles_c_in = in_channels // TILE_IC
    n_tiles_c_out = out_channels // TILE_OC
    n_tiles_h = out_height // output_tile_height

    W_tiled = nl.ndarray(
        shape=(n_tiles_c_out, nl.par_dim(TILE_OC), n_tiles_c_in, TILE_IC, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf
    )

    for c_out in nl.affine_range(n_tiles_c_out):
        for c_in in nl.affine_range(n_tiles_c_in):
            W_tiled[c_out, :, c_in, :, :, :] = nl.load(W[nl.ds(c_out*TILE_OC, TILE_OC), nl.ds(c_in*TILE_IC, TILE_IC), :, :])

    W_transposed = nl.ndarray(
        shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(TILE_IC), TILE_OC),
        dtype=W.dtype,
        buffer=nl.sbuf
    )

    for fh in nl.affine_range(filter_height):
        for fw in nl.affine_range(filter_width):
            for c_out in nl.affine_range(n_tiles_c_out):
                for c_in in nl.affine_range(n_tiles_c_in):
                    W_slice = nl.copy(W_tiled[c_out, :, c_in, :, fh, fw])
                    W_transposed[fh, fw, c_out, c_in] = nisa.nc_transpose(W_slice)

    for b in nl.affine_range(batch_size):
        for tile_h in nl.affine_range(n_tiles_h):
            X_tile = nl.ndarray(
                shape=(n_tiles_c_in, nl.par_dim(TILE_IC), input_tile_height, input_width),
                dtype=X.dtype,
                buffer=nl.sbuf
            )

            for c_in in nl.affine_range(n_tiles_c_in):
                X_tile[c_in, :, :, :] = nl.load(
                    X[b, nl.ds(c_in*TILE_IC, TILE_IC),
                    nl.ds(tile_h*output_tile_height, input_tile_height),
                    :
                ])

            for c_out in nl.affine_range(n_tiles_c_out):
                output_tile = nl.ndarray(
                    shape=(nl.par_dim(TILE_OC), output_tile_height, out_width),
                    dtype=X_out.dtype,
                    buffer=nl.sbuf
                )

                for out_row in nl.affine_range(output_tile_height):
                    acc_tile = nl.zeros((nl.par_dim(TILE_OC), out_width), np.float32, buffer=nl.psum)

                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            for c_in_tile in nl.affine_range(n_tiles_c_in):
                                x_slice = X_tile[c_in_tile, :, out_row + i, j:j+out_width]
                                w_slice = W_transposed[i, j, c_out, c_in_tile, :, :]
                                acc_tile += nl.matmul(w_slice, x_slice, transpose_x=True)

                    output_tile[:, out_row, :] = acc_tile

                bias_tile = nl.load(bias[nl.ds(c_out*TILE_OC, TILE_OC)]).reshape((TILE_OC, 1))
                output_tile = nisa.tensor_scalar(output_tile, np.add, bias_tile)
                h_start = tile_h * output_tile_height
                nl.store(
                    X_out[b, nl.ds(c_out*TILE_OC, TILE_OC), h_start:h_start+output_tile_height, :],
                    output_tile
                )

    return X_out
