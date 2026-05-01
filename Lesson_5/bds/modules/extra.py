import sympy as sp
import scipy as sc
import numpy as np

import modules.polys as pl


def poly_fit(data: np.ndarray, rank: int, length: int = 0):
    # Determine bounds
    if length == 0:
        length = data.size
    if length < 0:
        raise ValueError("Bounds specified are invalid")
    # Deal with the rank
    if rank >= 1 and rank <= 10:
        # Pick poly model
        poly = pl.polys[rank - 1]
        coeffs_p = sp.poly(poly, pl.t).coeffs()
        # Fit it
        coeffs_v = np.polyfit(np.arange(data.size), data, rank - 1)
        for i, c in enumerate(coeffs_p):
            poly = poly.subs(c, coeffs_v[i])
        poly = sp.lambdify(pl.t, poly)
        # Calculate values
        poly_y = np.ndarray(length)
        with np.nditer(poly_y, op_flags=["readwrite"], flags=["f_index"]) as it:
            for value in it:
                value[...] = poly(it.index)
        return poly_y
    else:
        raise ValueError("Number of coeffs must be withing [2 - 10] range.")


def dsb_fit(
    expression: str,
    main_var: str,
    data: np.ndarray,
    rank: int = 0,
    numeric: bool = True,
    maxfev: int = 100000,
    length: int = 0,
) -> np.ndarray:
    # Parse given strings
    model = sp.parse_expr(expression)
    model_a = sorted(model.free_symbols, key=lambda s: s.name)

    var = sp.var(main_var)
    model_a.remove(var)

    # Determine bounds
    if length == 0:
        length = data.size
    if length < 0:
        raise ValueError("Bounds specified are invalid")

    # Do the LSM thing (numpy territory)

    # Determine the poly rank here
    poly = pl.poly0
    if rank == 0:
        rank = len(model_a)
    if rank >= 1 and rank <= 10:
        poly = pl.polys[rank - 1]
    else:
        raise ValueError("Number of coeffs must be withing [2 - 10] range.")
    # Calculate poly weights
    poly_vals = np.polyfit(np.arange(data.size), data, rank - 1)

    # Do the DSB thing (sympy territory)

    # Extract serries expansion's coeffs
    series_model = sp.series(model, var, n=rank).removeO()
    coeffs_m = sp.poly(series_model, var).coeffs()
    coeffs_p = sp.poly(poly, pl.t).coeffs()
    # To balance discretes we need coeffs of a plain series
    # (the same curvature but without free coeffs)
    model_plain = model.copy()
    for a in model_a:
        model_plain = model_plain.subs(a, 1)
    if model_plain.has(1):
        model_plain = model_plain - 1
    series_plain = sp.series(model_plain, var, n=rank).removeO()
    coeffs_mp = sp.poly(series_plain, var).coeffs()
    # Get the spectrum and balance it against poly coeffs
    spectrum = []
    for i, a in enumerate(coeffs_m):
        spectrum.append(a / coeffs_mp[i])
    balance = []
    for i, d in enumerate(spectrum):
        balance.append(sp.Eq(d, coeffs_p[i]))
    # Solve the resulting system against model's coeffs
    solution = sp.nonlinsolve(balance, model_a)
    solution = list(solution)[0]

    # End of the road if no numeric(
    if numeric == False:
        # Substitute model with calculated poly weights
        for i, a in enumerate(model_a):
            model = model.subs(a, solution[i])
        for i, c in enumerate(coeffs_p):
            model = model.subs(c, poly_vals[i])
        # Calculate fitted values
        model = sp.lambdify(var, model)
        model_y = np.ndarray(length)
        with np.nditer(model_y, op_flags=["readwrite"], flags=["f_index"]) as it:
            for value in it:
                value[...] = model(it.index)
        return model_y

    else:
        # Do the numeric thing (numpy and scipy territory)

        # Substitute solution with poly weights
        for i, c in enumerate(coeffs_p):
            solution = solution.subs(c, poly_vals[i])
        # Add main var to the front of the symbol list
        model_a.insert(0, var)
        # We need callable function to pass it into scipy
        model_l = sp.lambdify(model_a, model)
        # Scipy cannot work with sympy types(
        solution = np.array(solution).astype(np.float64)
        # Curve fit this thing!
        coeffs_n = sc.optimize.curve_fit(
            model_l, np.arange(data.size), data, p0=solution, maxfev=maxfev
        )
        # Calculate new fitted values
        model_y = np.ndarray(length)
        with np.nditer(model_y, op_flags=["readwrite"], flags=["f_index"]) as it:
            for value in it:
                value[...] = model_l(it.index, *coeffs_n[0])
        return model_y
