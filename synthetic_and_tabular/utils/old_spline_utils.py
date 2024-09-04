import numpy as np
from scipy.interpolate import BSpline


class BasisIndicesSearcher:
    def __init__(self, knots, order=4):
        self.knots = knots
        self.order = order

    def __call__(self, x):
        knots = self.knots
        order = self.order

        idx = np.searchsorted(knots, x, side='right')
        if idx == len(knots):
            idx -= order

        for i in range(order, 0, -1):
            j = idx - i
            if j >= 0:
                yield j


def get_spline_basis(n, order=4):
    knots = get_knots(n, order)

    cubic_basis = [
        BSpline.basis_element(knots[j:(j + order + 1)], extrapolate=False)
        for j in range(len(knots) - order)
    ]

    return cubic_basis, BasisIndicesSearcher(knots, order)


def get_knots(n, order):
    inner_knots = list(np.linspace(0., 1., n))
    knots = np.array(
        [inner_knots[0]] * (order - 1) +
        inner_knots +
        [inner_knots[-1]] * (order - 1)
    )
    return knots
