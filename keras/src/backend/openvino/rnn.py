
import numpy as np
import openvino.opset15 as ov_opset
from openvino import Type

from keras.src import tree
from keras.src.backend.common import stateless_scope
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import get_ov_output
from keras.src.backend.openvino.core import while_loop
from keras.src.backend.openvino.core import ov_to_keras_type
from keras.src.backend.openvino.core import convert_to_tensor
from keras.src.backend.openvino.numpy import split as numpy_split


def rnn(
    step_function,
    inputs,
    initial_states,
    go_backwards=False,
    mask=None,
    constants=None,
    unroll=False,
    input_length=None,
    time_major=False,
    zero_output_for_mask=False,
    return_all_outputs=True,
):
    def swap_batch_timestep(input_t):
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        perm = ov_opset.constant(axes, Type.i32).output(0)
        return OpenVINOKerasTensor(ov_opset.transpose(input_t.output, perm).output(0))

    if not time_major:
        inputs = tree.map_structure(swap_batch_timestep, inputs)

    flattened_inputs = tree.flatten(inputs)
    time_steps = flattened_inputs[0].shape[0]

    if mask is not None:
        mask_ov = get_ov_output(mask)
        if mask_ov.get_element_type() != Type.boolean:
            mask = OpenVINOKerasTensor(ov_opset.convert(mask_ov, Type.boolean).output(0))
        if len(mask.shape) == 2:
            mask = OpenVINOKerasTensor(
                ov_opset.unsqueeze(mask.output, ov_opset.constant(-1, Type.i32).output(0)).output(0)
            )
        if not time_major:
            mask = swap_batch_timestep(mask)

    if constants is None:
        constants = []

    def _expand_mask(mask_t, input_t, fixed_dim=1):
        if tree.is_nested(mask_t):
            raise ValueError(
                f"mask_t is expected to be tensor, but got {mask_t}"
            )
        if tree.is_nested(input_t):
            raise ValueError(
                f"input_t is expected to be tensor, but got {input_t}"
            )
        rank_diff = len(input_t.shape) - len(mask_t.shape)
        mask_out = mask_t.output
        for _ in range(rank_diff):
            mask_out = ov_opset.unsqueeze(mask_out, ov_opset.constant(-1, Type.i32).output(0)).output(0)

        multiples = [1] * fixed_dim + list(input_t.shape[fixed_dim:])
        multiples_node = ov_opset.constant(multiples, Type.i32).output(0)
        tiled = ov_opset.tile(mask_out, multiples_node).output(0)
        return OpenVINOKerasTensor(tiled)

    # Note: OpenVINO opset doesn't support complex control flow easily for unrolling
    # like conditional execution within symbolic graph if we want to mimic python logic.
    # But for "unroll=True", we just execute python loop.
    if unroll or time_steps is not None:
        if not time_steps:
            raise ValueError("Unrolling requires a fixed number of timesteps. If time_steps is None, ensure your input has a static shape.")
        states = tuple(initial_states)
        successive_states = []
        successive_outputs = []

        def _process_single_input_t(input_t):
            input_t = unstack(input_t)  # unstack for time_step dim
            if go_backwards:
                input_t.reverse()
            return input_t

        if tree.is_nested(inputs):
            processed_input = tree.map_structure(
                _process_single_input_t, inputs
            )
        else:
            processed_input = (_process_single_input_t(inputs),)

        def _get_input_tensor(time):
            inp = [t_[time] for t_ in processed_input]
            return tree.pack_sequence_as(inputs, inp)

        if mask is not None:
            mask_list = unstack(mask)
            if go_backwards:
                mask_list.reverse()

            for i in range(time_steps):
                inp = _get_input_tensor(i)
                mask_t = mask_list[i]
                output, new_states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                tiled_mask_t = _expand_mask(mask_t, output)

                if not successive_outputs:
                    prev_output = OpenVINOKerasTensor(
                        ov_opset.broadcast(
                            ov_opset.constant(0, output.output.get_element_type()).output(0),
                            ov_opset.shape_of(output.output, Type.i32).output(0)
                        ).output(0)
                    )
                else:
                    prev_output = successive_outputs[-1]

                output = OpenVINOKerasTensor(ov_opset.select(tiled_mask_t.output, output.output, prev_output.output).output(0))

                flat_states = tree.flatten(states)
                flat_new_states = tree.flatten(new_states)
                tiled_mask_t_states = tuple(
                    _expand_mask(mask_t, s) for s in flat_states
                )
                flat_final_states = tuple(
                    OpenVINOKerasTensor(ov_opset.select(m.output, s.output, ps.output).output(0))
                    for m, s, ps in zip(
                        tiled_mask_t_states, flat_new_states, flat_states
                    )
                )
                states = tree.pack_sequence_as(states, flat_final_states)

                if return_all_outputs:
                    successive_outputs.append(output)
                    successive_states.append(states)
                else:
                    successive_outputs = [output]
                    successive_states = [states]
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = _stack(successive_outputs)

        else:  # mask is None
            for i in range(time_steps):
                inp = _get_input_tensor(i)
                output, states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                if return_all_outputs:
                    successive_outputs.append(output)
                    successive_states.append(states)
                else:
                    successive_outputs = [output]
                    successive_states = [states]
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = _stack(successive_outputs)

    else: # Unroll == False and time_steps is None (dynamic)
        # Symbolic loop implementation for OpenVINO is non-trivial here due to:
        # 1. Need to adapt `step_function` (which is Keras logic) into `ov_opset.loop` body.
        # 2. Variable length sequences are supported in OpenVINO but `rnn` abstraction usually implies fixed static graph construction in current backends logic structure unless using `while_loop`.
        # Given we prioritized static shapes (common in inference):
        raise ValueError("OpenVINO backend requires static timesteps for RNN. Please ensure input shapes are static or use `unroll=True` with fixed steps.")

    if not time_major:
        outputs = tree.map_structure(swap_batch_timestep, outputs)

    return last_output, outputs, new_states


def lstm(
    inputs,
    initial_state_h,
    initial_state_c,
    mask,
    kernel,
    recurrent_kernel,
    bias,
    activation,
    recurrent_activation,
    return_sequences=False,
    go_backwards=False,
    unroll=False,
    time_major=False,
):
    # Fallback to generic rnn implementation as `ov_opset.lstm_sequence` has strict requirements
    # and different weight layouts that are complex to map perfectly with Keras's flexible `lstm` arguments
    # (e.g. activation functions matching, separate biases).
    # Since the user requested implementing LSTM layer support in `rnn.py`, ensuring `rnn` works is sufficient
    # for the LSTM layer to function (it calls `rnn` if `cudnn_ok` is false or `lstm` is not implemented/raises).
    # However, to be explicit:
    raise NotImplementedError("`lstm` optimized op is not supported with openvino backend; falling back to generic `rnn`.")


def gru(*args, **kwargs):
    raise NotImplementedError("`gru` is not supported with openvino backend")


def unstack(x, axis=0):
    """Unstack a tensor along a given axis."""
    x_ov = get_ov_output(x)
    shape = x_ov.get_partial_shape()
    rank = shape.rank.get_length()

    if axis < 0:
        axis += rank

    dim = shape[axis]
    if dim.is_dynamic:
         raise ValueError(f"Cannot unstack along a dynamic dimension {axis} of tensor with shape {shape}")

    num_splits = dim.get_length()

    # Split into num_splits parts along axis
    # split returns list of output nodes
    axis_node = ov_opset.constant(axis, Type.i32).output(0)
    splits = ov_opset.split(x_ov, axis_node, num_splits)

    results = []
    squeeze_axis = ov_opset.constant([axis], Type.i32).output(0)
    for i in range(num_splits):
        # split output keeps the dimension as 1, so we need to squeeze it
        squeezed = ov_opset.squeeze(splits.output(i), squeeze_axis).output(0)
        results.append(OpenVINOKerasTensor(squeezed))

    return results


def _stack(tensor_list):
    # Helper to stack list of tensors
    ov_outputs = [t.output for t in tensor_list]
    stacked = ov_opset.concat([ov_opset.unsqueeze(o, ov_opset.constant(0, Type.i32).output(0)).output(0) for o in ov_outputs], 0).output(0)
    return OpenVINOKerasTensor(stacked)


def numpy_scan(f, init, xs, reverse=False, mask=None):
    raise NotImplementedError(
        "`numpy_scan` is not supported with openvino backend"
    )


def cudnn_ok(*args, **kwargs):
    return False
