import numpy as np
import tensorflow as tf
from scipy.optimize import root
from functools import partial

##############################################################################
#  A) The MODEL EQUATIONS in LOG-SPACE (unchanged from your earlier approach)
##############################################################################
def ai_model_equations_exp(vars_, K_g, K_a, D, Z, params):
    """
    'vars_' = [x, y, z] => L_g=exp(x), L_a=exp(y), w=exp(z)
    Returns: residuals of your 3 conditions:

    (1) FOC wrt L_g: MP_Lg - w = 0
    (2) FOC wrt L_a: MP_La - w = 0
    (3) Labor supply: 1 - L_g - L_a - ψ*(Y/w) = 0
    """
    x, y, z = vars_
    L_g = np.exp(x)
    L_a = np.exp(y)
    w   = np.exp(z)

    β  = params['β']
    ι  = params['ι']
    γ  = params['γ']
    ρ  = params['ρ']
    ψ  = params['ψ']
    α  = params['α']
    θ  = params['θ']
    A_g= params['A_g']

    # AI output
    X = Z * (K_a**α) * (L_a**(1 - α))
    # Effective AI labor
    L_AI = (D**ρ) * (X**(1 - ρ))
    # bracket
    bracket_val = ι*(L_g**γ) + (1 - ι)*(L_AI**γ)
    # Y
    Y = A_g * (K_g**β) * (bracket_val**((1 - β)/γ))
    # p
    p = (A_g*(K_g**β)
         * bracket_val**(((1 - β)/γ - 1))
         * (1 - β)*(1 - ι)*(L_AI**(γ - 1))
         * (1 - θ)*(D**θ)*(X**(-θ))
    )

    # FOC wrt L_g
    MP_Lg = (A_g*(K_g**β)
             * ((1 - β)/γ)* ι*(1 - β)
             * bracket_val**(((1 - β)/γ - 1))
             * (L_g**(γ - 1))
    )
    res1 = MP_Lg - w

    # FOC wrt L_a
    MP_La = p * (Z*(1 - α)*(K_a**α)*(L_a**(-α)))
    res2 = MP_La - w

    # Labor supply
    res3 = 1 - L_g - L_a - ψ*(Y / w)

    return np.array([res1, res2, res3])


def solve_one_row(K_g_val, K_a_val, D_val, Z_val, params,
                  guess_log=(-1.0, -1.0, 0.0)):
    """
    Solve the system for a single set (K_g, K_a, D, Z).
    Return (L_g, L_a, w, p).
    If solver fails or yields non-positive solution => (nan, nan, nan, nan).
    """
    sol = root(
        ai_model_equations_exp,
        guess_log,
        args=(K_g_val, K_a_val, D_val, Z_val, params),
        method='hybr'
    )
    if not sol.success:
        return (np.nan, np.nan, np.nan, np.nan)

    x_sol, y_sol, z_sol = sol.x
    L_g_sol = np.exp(x_sol)
    L_a_sol = np.exp(y_sol)
    w_sol   = np.exp(z_sol)

    if (L_g_sol <= 0) or (L_a_sol <= 0) or (w_sol <= 0):
        return (np.nan, np.nan, np.nan, np.nan)

    # Once we have L_g, L_a, w, we can compute p
    β  = params['β']
    ι  = params['ι']
    γ  = params['γ']
    ρ  = params['ρ']
    ψ  = params['ψ']
    α  = params['α']
    θ  = params['θ']
    A_g= params['A_g']

    X = Z_val * (K_a_val**α) * (L_a_sol**(1 - α))
    L_AI = (D_val**ρ)*(X**(1 - ρ))
    bracket_val = ι*(L_g_sol**γ) + (1 - ι)*(L_AI**γ)
    p_val = (A_g*(K_g_val**β)
             * bracket_val**(((1 - β)/γ - 1))
             * (1 - β)*(1 - ι)*(L_AI**(γ - 1))
             * (1 - θ)*(D_val**θ)*(X**(-θ))
    )
    return (L_g_sol, L_a_sol, w_sol, p_val)


##############################################################################
# B) The pure-Python function that handles the entire BATCH in NumPy
##############################################################################
def solve_IntraTemporal_batch_py(logK_g_np, logK_a_np, logZ_np, logD_np, params):
    """
    Inputs: 4 numpy arrays of shape (N,)
    Return: 4 numpy arrays of shape (N,)
       => L_g, L_a, w, p
    """
    N = logK_g_np.shape[0]

    K_g_arr = np.exp(logK_g_np)
    K_a_arr = np.exp(logK_a_np)
    Z_arr   = np.exp(logZ_np)
    D_arr   = np.exp(logD_np)

    L_g_list = []
    L_a_list = []
    w_list   = []
    p_list   = []

    # Solve row-by-row
    for i in range(N):
        L_g_val, L_a_val, w_val, p_val = solve_one_row(
            K_g_arr[i], K_a_arr[i], D_arr[i], Z_arr[i], params
        )
        L_g_list.append(L_g_val)
        L_a_list.append(L_a_val)
        w_list.append(w_val)
        p_list.append(p_val)

    # Convert to float64 arrays
    L_g_out = np.array(L_g_list, dtype=np.float32)
    L_a_out = np.array(L_a_list, dtype=np.float32)
    w_out   = np.array(w_list,   dtype=np.float32)
    p_out   = np.array(p_list,   dtype=np.float32)
    return (L_g_out, L_a_out, w_out, p_out)


##############################################################################
# C) The TF "wrapper" that calls our Python function via tf.py_function
##############################################################################
def solve_IntraTemporal_batch_tf(logK_g, logK_a, logZ, logD, params):
    """
    Accepts Tensors logK_g, logK_a, logZ, logD of shape (N,1) or (N,).
    Returns L_g_tf, L_a_tf, w_tf, p_tf as Tensors of the same shape & dtype.

    Because we rely on Python code (scipy.optimize) we must use tf.py_function.
    """
    # 1) Ensure shape is (N,) rather than (N,1), for convenience
    #    (tf.py_function will pass them as rank-1 arrays if we do .numpy().flatten())
    logK_g = tf.reshape(logK_g, [-1])
    logK_a = tf.reshape(logK_a, [-1])
    logZ   = tf.reshape(logZ,   [-1])
    logD   = tf.reshape(logD,   [-1])

    # 2) Define a small lambda that calls our Python solver
    def _py_solve_func(logK_g_np, logK_a_np, logZ_np, logD_np):
        # logK_g_np, etc. are np arrays
        L_g_np, L_a_np, w_np, p_np = solve_IntraTemporal_batch_py(
            logK_g_np, logK_a_np, logZ_np, logD_np, params
        )
        return (L_g_np, L_a_np, w_np, p_np)

    # 3) Use tf.py_function
    outputs = tf.py_function(
        func=_py_solve_func,
        inp=[logK_g, logK_a, logZ, logD],
        Tout=[tf.float32, tf.float32, tf.float32, tf.float32]
    )
    # outputs is a list of 4 rank-1 Tensors of dtype float64

    L_g_tf, L_a_tf, w_tf, p_tf = outputs

    # 4) Set shapes back to (N,1) to match your original usage
    
    L_g_tf = tf.reshape(L_g_tf, [params['batch_size'], 1])
    L_a_tf = tf.reshape(L_a_tf, [params['batch_size'], 1])
    w_tf   = tf.reshape(w_tf,   [params['batch_size'], 1])
    p_tf   = tf.reshape(p_tf,   [params['batch_size'], 1])

    # Optionally cast them to the same float type as input
    # (if your net uses float32, you might want to do .cast)
    # For example:
    # L_g_tf = tf.cast(L_g_tf, logK_g.dtype)

    return L_g_tf, L_a_tf, w_tf, p_tf

