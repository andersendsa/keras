import openvino.opset15 as ov_opset
from openvino import Type

from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import cast
from keras.src.backend.openvino.core import convert_to_tensor
from keras.src.backend.openvino.core import get_ov_output


def cholesky(a, upper=False):
    raise NotImplementedError(
        "`cholesky` is not supported with openvino backend."
    )


def cholesky_inverse(a, upper=False):
    a = convert_to_tensor(a)
    a_ov = get_ov_output(a)
    if upper:
        # Reconstruct A = U^T @ U, then invert
        reconstructed_matrix = ov_opset.matmul(a_ov, a_ov, True, False).output(
            0
        )
    else:
        # Reconstruct A = L @ L^T, then invert
        reconstructed_matrix = ov_opset.matmul(a_ov, a_ov, False, True).output(
            0
        )
    result = ov_opset.inverse(reconstructed_matrix, adjoint=False).output(0)
    return OpenVINOKerasTensor(result)


def det(a):
    raise NotImplementedError("`det` is not supported with openvino backend")


def _jacobi_eigh(a_ov):
    import numpy as np
    from openvino import Model
    from openvino import PartialShape
    from openvino import Type
    from keras.src.backend.openvino.numpy import sort, argsort
    from keras.src.backend.openvino.numpy import take_along_axis

    pshape = a_ov.get_partial_shape()
    if pshape.rank.is_dynamic:
        raise NotImplementedError("`eigh` requires a static rank.")
    rank = pshape.rank.get_length()
    n_dim = pshape[rank - 1]
    if n_dim.is_dynamic:
        raise NotImplementedError(
            "`eigh` requires a static matrix size for the last two dimensions."
        )
    N = n_dim.get_length()
    original_dtype = a_ov.get_element_type()

    # Workaround for f64 ops failing inside loop in OpenVINO CPU plugin
    if original_dtype == Type.f64:
        a_ov = ov_opset.convert(a_ov, Type.f32).output(0)

    dtype = a_ov.get_element_type()

    V_ov = ov_opset.broadcast(
        ov_opset.constant(np.eye(N, dtype=np.float32), dtype),
        ov_opset.shape_of(a_ov),
    ).output(0)

    p_A = ov_opset.parameter(a_ov.get_partial_shape(), dtype)
    p_V = ov_opset.parameter(V_ov.get_partial_shape(), dtype)
    p_p = ov_opset.parameter(PartialShape([]), Type.i32)
    p_q = ov_opset.parameter(PartialShape([]), Type.i32)

    A_b = p_A.output(0)
    V_b = p_V.output(0)
    p_b = p_p.output(0)
    q_b = p_q.output(0)

    zero = ov_opset.constant(0.0, dtype).output(0)
    one = ov_opset.constant(1.0, dtype).output(0)
    two = ov_opset.constant(2.0, dtype).output(0)
    rank_m1 = ov_opset.constant(rank - 1, Type.i32)
    rank_m2 = ov_opset.constant(rank - 2, Type.i32)

    p_idx = ov_opset.unsqueeze(p_b, ov_opset.constant([0], Type.i32)).output(0)
    q_idx = ov_opset.unsqueeze(q_b, ov_opset.constant([0], Type.i32)).output(0)

    A_row_p = ov_opset.gather(A_b, p_idx, rank_m2).output(0)
    A_row_q = ov_opset.gather(A_b, q_idx, rank_m2).output(0)

    A_pq = ov_opset.gather(A_row_p, q_idx, rank_m1).output(0)
    A_pp = ov_opset.gather(A_row_p, p_idx, rank_m1).output(0)
    A_qq = ov_opset.gather(A_row_q, q_idx, rank_m1).output(0)

    is_zero = ov_opset.equal(A_pq, zero).output(0)

    num = ov_opset.subtract(A_qq, A_pp).output(0)
    den = ov_opset.multiply(two, A_pq).output(0)

    safe_den = ov_opset.select(is_zero, one, den).output(0)
    tau = ov_opset.divide(num, safe_den).output(0)

    tau_sq = ov_opset.multiply(tau, tau).output(0)
    one_plus_tau_sq = ov_opset.add(one, tau_sq).output(0)
    sqrt_part = ov_opset.sqrt(one_plus_tau_sq).output(0)

    abs_tau = ov_opset.abs(tau).output(0)
    t_den = ov_opset.add(abs_tau, sqrt_part).output(0)

    sign_tau = ov_opset.sign(tau).output(0)
    is_tau_zero = ov_opset.equal(tau, zero).output(0)
    sign_tau = ov_opset.select(is_tau_zero, one, sign_tau).output(0)

    t = ov_opset.divide(sign_tau, t_den).output(0)

    t_sq = ov_opset.multiply(t, t).output(0)
    one_plus_t_sq = ov_opset.add(one, t_sq).output(0)
    sqrt_t = ov_opset.sqrt(one_plus_t_sq).output(0)
    c = ov_opset.divide(one, sqrt_t).output(0)

    s = ov_opset.multiply(c, t).output(0)

    C = ov_opset.select(is_zero, one, c).output(0)
    S = ov_opset.select(is_zero, zero, s).output(0)

    C_row_p = ov_opset.multiply(C, A_row_p).output(0)
    S_row_q = ov_opset.multiply(S, A_row_q).output(0)
    S_row_p = ov_opset.multiply(S, A_row_p).output(0)
    C_row_q = ov_opset.multiply(C, A_row_q).output(0)

    new_A_row_p = ov_opset.subtract(C_row_p, S_row_q).output(0)
    new_A_row_q = ov_opset.add(S_row_p, C_row_q).output(0)

    A_b = ov_opset.scatter_update(A_b, p_idx, new_A_row_p, rank_m2).output(0)
    A_b = ov_opset.scatter_update(A_b, q_idx, new_A_row_q, rank_m2).output(0)

    A_col_p = ov_opset.gather(A_b, p_idx, rank_m1).output(0)
    A_col_q = ov_opset.gather(A_b, q_idx, rank_m1).output(0)

    C_col_p = ov_opset.multiply(C, A_col_p).output(0)
    S_col_q = ov_opset.multiply(S, A_col_q).output(0)
    S_col_p = ov_opset.multiply(S, A_col_p).output(0)
    C_col_q = ov_opset.multiply(C, A_col_q).output(0)

    new_A_col_p = ov_opset.subtract(C_col_p, S_col_q).output(0)
    new_A_col_q = ov_opset.add(S_col_p, C_col_q).output(0)

    A_b = ov_opset.scatter_update(A_b, p_idx, new_A_col_p, rank_m1).output(0)
    A_b = ov_opset.scatter_update(A_b, q_idx, new_A_col_q, rank_m1).output(0)

    V_col_p = ov_opset.gather(V_b, p_idx, rank_m1).output(0)
    V_col_q = ov_opset.gather(V_b, q_idx, rank_m1).output(0)

    C_V_col_p = ov_opset.multiply(C, V_col_p).output(0)
    S_V_col_q = ov_opset.multiply(S, V_col_q).output(0)
    S_V_col_p = ov_opset.multiply(S, V_col_p).output(0)
    C_V_col_q = ov_opset.multiply(C, V_col_q).output(0)

    new_V_col_p = ov_opset.subtract(C_V_col_p, S_V_col_q).output(0)
    new_V_col_q = ov_opset.add(S_V_col_p, C_V_col_q).output(0)

    V_b = ov_opset.scatter_update(V_b, p_idx, new_V_col_p, rank_m1).output(0)
    V_b = ov_opset.scatter_update(V_b, q_idx, new_V_col_q, rank_m1).output(0)

    N_const = ov_opset.constant(N, Type.i32).output(0)
    one_i32 = ov_opset.constant(1, Type.i32).output(0)
    zero_i32 = ov_opset.constant(0, Type.i32).output(0)

    q_next = ov_opset.add(q_b, one_i32).output(0)
    is_q_n = ov_opset.equal(q_next, N_const).output(0)

    p_plus_1 = ov_opset.add(p_b, one_i32).output(0)
    p_next = ov_opset.select(is_q_n, p_plus_1, p_b).output(0)
    q_next = ov_opset.select(
        is_q_n, ov_opset.add(p_next, one_i32).output(0), q_next
    ).output(0)

    N_minus_1 = ov_opset.subtract(N_const, one_i32).output(0)
    is_p_n_minus_1 = ov_opset.equal(p_next, N_minus_1).output(0)

    p_next_final = ov_opset.select(is_p_n_minus_1, zero_i32, p_next).output(0)
    q_next_final = ov_opset.select(is_p_n_minus_1, one_i32, q_next).output(0)

    cond_true = ov_opset.constant(True, Type.boolean).output(0)
    body = Model(
        [cond_true, A_b, V_b, p_next_final, q_next_final],
        [p_A, p_V, p_p, p_q],
    )

    max_sweeps = 5
    num_rotations = int(max_sweeps * N * (N - 1) / 2)
    trip_count = ov_opset.constant(num_rotations, Type.i32).output(0)
    exec_cond = ov_opset.constant(True, Type.boolean).output(0)

    loop = ov_opset.loop(trip_count, exec_cond)
    loop.set_function(body)
    loop.set_special_body_ports([-1, 0])

    p_init = ov_opset.constant(0, Type.i32).output(0)
    q_init = ov_opset.constant(1, Type.i32).output(0)

    loop.set_merged_input(p_A, a_ov, A_b)
    loop.set_merged_input(p_V, V_ov, V_b)
    loop.set_merged_input(p_p, p_init, p_next_final)
    loop.set_merged_input(p_q, q_init, q_next_final)

    out_A = loop.get_iter_value(A_b, -1)
    out_V = loop.get_iter_value(V_b, -1)

    I = ov_opset.broadcast(
        ov_opset.constant(np.eye(N, dtype=np.float32), dtype),
        ov_opset.shape_of(out_A),
    ).output(0)
    W = ov_opset.reduce_sum(
        ov_opset.multiply(out_A, I).output(0),
        ov_opset.constant([rank - 1], Type.i32),
        False,
    ).output(0)

    if original_dtype == Type.f64:
        W = ov_opset.convert(W, Type.f64).output(0)
        out_V = ov_opset.convert(out_V, Type.f64).output(0)

    # Sort eigenvalues in ascending order to align with NumPy behavior
    W_tensor = OpenVINOKerasTensor(W)
    out_V_tensor = OpenVINOKerasTensor(out_V)

    indices = argsort(W_tensor, axis=-1)
    W_sorted = take_along_axis(W_tensor, indices, axis=-1)

    # Sort eigenvectors based on sorted eigenvalue indices
    indices_expanded = ov_opset.unsqueeze(
        get_ov_output(indices),
        ov_opset.constant([rank - 2], Type.i32)
    ).output(0)

    # out_V is (..., N, N), we want to sort along the last axis
    N_const = ov_opset.constant(N, Type.i32).output(0)
    indices_broadcast = ov_opset.broadcast(
        indices_expanded,
        ov_opset.shape_of(out_V),
    ).output(0)

    # Use gather_elements to take along axis for eigenvectors
    out_V_sorted = ov_opset.gather_elements(
        out_V,
        indices_broadcast,
        rank - 1
    ).output(0)

    return W_sorted, OpenVINOKerasTensor(out_V_sorted)


def eig(a):
    a = convert_to_tensor(a)
    a_ov = get_ov_output(a)
    return _jacobi_eigh(a_ov)


def eigh(a):
    a = convert_to_tensor(a)
    a_ov = get_ov_output(a)
    return _jacobi_eigh(a_ov)


def inv(a):
    a = convert_to_tensor(a)
    a_ov = get_ov_output(a)
    result = ov_opset.inverse(a_ov, adjoint=False).output(0)
    return OpenVINOKerasTensor(result)


def lu_factor(a):
    raise NotImplementedError(
        "`lu_factor` is not supported with openvino backend"
    )


def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    x_shape = tuple(x.shape)
    ndim = len(x_shape)

    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    if any(a < -ndim or a >= ndim for a in axis):
        raise ValueError(
            "All `axis` values must be in the range [-ndim, ndim). "
            f"Received inputs with ndim={ndim}, while axis={axis}"
        )
    axis = axis[0] if len(axis) == 1 else axis
    num_axes = 1 if isinstance(axis, int) else len(axis)

    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)

    x_ov = get_ov_output(x)

    # Ref: jax.numpy.linalg.norm
    if num_axes == 1:
        if ord is None or ord == 2:
            # L2 norm: sqrt(sum(x * conj(x)))
            x_conj = x_ov
            x_sq = ov_opset.multiply(x_conj, x_conj).output(0)
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            norm_result = ov_opset.reduce_sum(
                x_sq, axis_const, keepdims
            ).output(0)
            norm_result = ov_opset.sqrt(norm_result).output(0)
        elif ord == float("inf"):
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            x_abs = ov_opset.abs(x_ov).output(0)
            norm_result = ov_opset.reduce_max(
                x_abs, axis_const, keepdims
            ).output(0)
        elif ord == float("-inf"):
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            x_abs = ov_opset.abs(x_ov).output(0)
            norm_result = ov_opset.reduce_min(
                x_abs, axis_const, keepdims
            ).output(0)
        elif ord == 0:
            # Count non-zero elements
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            zero = ov_opset.constant(0.0, Type.f32).output(0)
            not_equal = ov_opset.not_equal(x_ov, zero).output(0)
            not_equal_float = ov_opset.convert(not_equal, Type.f32).output(0)
            norm_result = ov_opset.reduce_sum(
                not_equal_float, axis_const, keepdims
            ).output(0)
        elif ord == 1:
            # L1 norm: sum(|x|)
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            x_abs = ov_opset.abs(x_ov).output(0)
            norm_result = ov_opset.reduce_sum(
                x_abs, axis_const, keepdims
            ).output(0)
        elif isinstance(ord, str):
            raise ValueError(
                f"Invalid `ord` argument for vector norm. Received: ord={ord}"
            )
        else:
            # p-norm: (sum(|x|^p))^(1/p)
            ord_tensor = convert_to_tensor(ord, dtype=dtype)
            ord_ov = get_ov_output(ord_tensor)
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            x_abs = ov_opset.abs(x_ov).output(0)
            x_pow = ov_opset.power(x_abs, ord_ov).output(0)
            sum_pow = ov_opset.reduce_sum(x_pow, axis_const, keepdims).output(0)
            one = convert_to_tensor(1.0, dtype=dtype)
            one_ov = get_ov_output(one)
            inv_ord = ov_opset.divide(one_ov, ord_ov).output(0)
            norm_result = ov_opset.power(sum_pow, inv_ord).output(0)

    elif num_axes == 2:
        row_axis, col_axis = axis[0], axis[1]
        row_axis = row_axis + ndim if row_axis < 0 else row_axis
        col_axis = col_axis + ndim if col_axis < 0 else col_axis

        if ord is None or ord == "fro":
            # Frobenius norm: sqrt(sum(x * conj(x)))
            x_sq = ov_opset.multiply(x_ov, x_ov).output(0)
            axis_for_const = list(axis) if isinstance(axis, tuple) else axis
            axis_const = ov_opset.constant(axis_for_const, Type.i32).output(0)
            sum_sq = ov_opset.reduce_sum(x_sq, axis_const, keepdims).output(0)
            norm_result = ov_opset.sqrt(sum_sq).output(0)
        elif ord == 1:
            # Maximum absolute column sum
            if not keepdims and col_axis > row_axis:
                col_axis -= 1
            row_axis_const = ov_opset.constant(row_axis, Type.i32).output(0)
            col_axis_const = ov_opset.constant(col_axis, Type.i32).output(0)

            x_abs = ov_opset.abs(x_ov).output(0)
            col_sum = ov_opset.reduce_sum(
                x_abs, row_axis_const, keep_dims=keepdims
            ).output(0)
            norm_result = ov_opset.reduce_max(
                col_sum, col_axis_const, keep_dims=keepdims
            ).output(0)
        elif ord == -1:
            # Minimum absolute column sum
            if not keepdims and col_axis > row_axis:
                col_axis -= 1
            row_axis_const = ov_opset.constant(row_axis, Type.i32).output(0)
            col_axis_const = ov_opset.constant(col_axis, Type.i32).output(0)

            x_abs = ov_opset.abs(x_ov).output(0)
            col_sum = ov_opset.reduce_sum(
                x_abs, row_axis_const, keep_dims=keepdims
            ).output(0)
            norm_result = ov_opset.reduce_min(
                col_sum, col_axis_const, keep_dims=keepdims
            ).output(0)
        elif ord == float("inf"):
            # Maximum absolute row sum
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            col_axis_const = ov_opset.constant(col_axis, Type.i32).output(0)
            row_axis_const = ov_opset.constant(row_axis, Type.i32).output(0)

            x_abs = ov_opset.abs(x_ov).output(0)
            row_sum = ov_opset.reduce_sum(
                x_abs, col_axis_const, keep_dims=keepdims
            ).output(0)
            norm_result = ov_opset.reduce_max(
                row_sum, row_axis_const, keep_dims=keepdims
            ).output(0)
        elif ord == float("-inf"):
            # Minimum absolute row sum
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            col_axis_const = ov_opset.constant(col_axis, Type.i32).output(0)
            row_axis_const = ov_opset.constant(row_axis, Type.i32).output(0)

            x_abs = ov_opset.abs(x_ov).output(0)
            row_sum = ov_opset.reduce_sum(
                x_abs, col_axis_const, keep_dims=keepdims
            ).output(0)
            norm_result = ov_opset.reduce_min(
                row_sum, row_axis_const, keep_dims=keepdims
            ).output(0)
        elif ord in ("nuc", 2, -2):
            # Nuclear norm, spectral norm, and minimum singular value
            # These require SVD which is not supported in OpenVINO backend
            raise NotImplementedError(
                f"`norm` with ord={ord} for matrix norms requires SVD "
                "which is not supported with openvino backend"
            )
        else:
            raise ValueError(
                f"Invalid `ord` argument for matrix norm. Received: ord={ord}"
            )
    else:
        raise ValueError(f"Invalid axis values. Received: axis={axis}")

    return OpenVINOKerasTensor(norm_result)


def qr(x, mode="reduced"):
    raise NotImplementedError("`qr` is not supported with openvino backend")


def solve(a, b):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    a_ov = get_ov_output(a)
    b_ov = get_ov_output(b)
    squeeze = b.ndim == a.ndim - 1
    if squeeze:
        minus_one = ov_opset.constant([-1], Type.i32).output(0)
        b_ov = ov_opset.unsqueeze(b_ov, minus_one).output(0)
    a_inv = ov_opset.inverse(a_ov, adjoint=False).output(0)
    result = ov_opset.matmul(a_inv, b_ov, False, False).output(0)
    if squeeze:
        minus_one = ov_opset.constant([-1], Type.i32).output(0)
        result = ov_opset.squeeze(result, minus_one).output(0)
    return OpenVINOKerasTensor(result)


def solve_triangular(a, b, lower=False):
    raise NotImplementedError(
        "`solve_triangular` is not supported with openvino backend"
    )


def svd(x, full_matrices=True, compute_uv=True):
    raise NotImplementedError("`svd` is not supported with openvino backend")


def lstsq(a, b, rcond=None):
    raise NotImplementedError("`lstsq` is not supported with openvino backend")


def jvp(fun, primals, tangents, has_aux=False):
    raise NotImplementedError("`jvp` is not supported with openvino backend")
