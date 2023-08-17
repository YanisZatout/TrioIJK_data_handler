import numpy as np
import pandas as pd


def polynomial_string(df: pd.Series, *, degree: float = 8):
    """
    Generate a string representation of a polynomial function with given a
    pandas series

    Args:
        df (pd.Series): Quantity of interest to be interpolated
        degree (float): Degree of the interpolated polynomial

    Returns:
        str: String representation of the polynomial function.
    """
    x = df.index.values
    y = df.values
    coefficients = np.polyfit(x, y, degree)
    terms = []

    for i, coeff in enumerate(coefficients):
        degree = len(coefficients) - i - 1
        if coeff != 0 and coeff != 1 and degree != 0:
            sign = "+" if coeff >= 0 else "-"
            term = f"{sign}{abs(coeff)}*z^{degree}"
        elif coeff != 0 and coeff != 1:
            sign = "+" if coeff >= 0 else "-"
            term = f"{sign}{abs(coeff)}"
        elif coeff == 1:
            sign = "+" if coeff >= 0 else "-"
            term = f"{sign}{abs(coeff)}*z"
        terms.append(term)

    polynomial_str = "".join(terms)
    if polynomial_str.startswith("+"):
        polynomial_str = polynomial_str[1:]
    return polynomial_str, coefficients


def polynomial_string_deguelasse(df: pd.Series, *, degree: float = 8):
    """
    Generate a string representation of a polynomial function with given a
    pandas series

    Args:
        df (pd.Series): Quantity of interest to be interpolated
        degree (float): Degree of the interpolated polynomial

    Returns:
        str: String representation of the polynomial function.
    """
    x = df.index.values
    y = df.values
    coefficients = np.polyfit(x, y, degree)

    coefficients = "".join("".join(np.array_repr(coefficients).split("array([")[1].split("])")[0].split("\n"))).split()
    terms = []

    for i, coeff in enumerate(coefficients):
        degree = len(coefficients) - i - 1
        coeff = coeff.split(",")[0]
        if coeff != 0 and coeff != 1 and degree != 0:
            try:
                if isinstance(int(coeff[0]), int):
                    sign = "+"
            except ValueError:
                sign = ""
            term = f"{sign}{coeff}*z^{degree}"
        elif coeff != 0 and coeff != 1:
            try:
                if isinstance(int(coeff[0]), int):
                    sign = "+"
            except ValueError:
                sign = ""
            term = f"{sign}{coeff}"
        elif coeff == 1:
            try:
                if isinstance(int(coeff[0]), int):
                    sign = "+"
            except ValueError:
                sign = ""
            term = f"{sign}{coeff}*z"
        terms.append(term)

    polynomial_str = "".join(terms)
    if polynomial_str.startswith("+"):
        polynomial_str = polynomial_str[1:]
    return polynomial_str, coefficients
