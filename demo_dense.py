#!/usr/bin/env python3

import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.typing import NDArray

num_var = 3
num_eq = 1

np.random.seed(0)
c = np.random.rand(num_var)
a = np.random.rand(num_eq, num_var)
x0 = np.random.rand(num_var)
b = a @ x0
lb_rhs = x0 - 1.0
ub_rhs = x0 + 1.0

model = gp.Model()
variables = model.addMVar(num_var, lb=lb_rhs, ub=ub_rhs)
equalities = model.addMConstr(a, variables, "=", b)
model.setObjective(c @ variables, sense=gp.GRB.MINIMIZE)

model.optimize()
assert model.Status == gp.GRB.OPTIMAL

x = np.fromiter((v.x for v in variables), dtype=float)
lb_ = np.fromiter((v.lb for v in variables), dtype=float)
ub_ = np.fromiter((v.ub for v in variables), dtype=float)
rc = np.fromiter((v.RC for v in variables), dtype=float)
basis = np.fromiter((v.VBasis for v in variables), dtype=int)
pi = np.fromiter((e.Pi for e in equalities), dtype=float)
lb_pi = np.where(basis == GRB.NONBASIC_LOWER, rc, 0.0)
ub_pi = np.where(basis == GRB.NONBASIC_UPPER, rc, 0.0)

assert model.ObjCon == 0.0
assert (lb_rhs == lb_).all()
assert (ub_rhs == ub_).all()
assert model.ObjVal <= c @ x0

print("   pi:", pi)
print("pi_lb:", lb_pi)
print("pi_ub:", ub_pi)


def check_close(
    label: str,
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    tol: float = 1e-10,
) -> None:

    scale = np.maximum(np.abs(left), np.abs(right)) + 1.0
    mismatch = np.abs(left - right)
    print(label, np.max(mismatch))
    assert np.all(mismatch < tol * scale), mismatch


# Consistency of reported objective
check_close("objective", c @ x, model.ObjVal)
# KKT stationarity
check_close("KKT stationarity", c - (a.T @ pi + lb_pi + ub_pi), np.zeros(c.size))
# KKT primal feasibility
check_close("equalities", a @ x, b)
assert (lb_rhs <= x).all()
assert (x <= ub_rhs).all()
# KKT dual feasibility
assert (0 <= lb_pi).all()
assert (ub_pi <= 0).all()
# KKT complementarity
assert np.logical_xor(0 == pi, a @ x == b).all()
assert np.logical_xor(0 == lb_pi, x - lb_rhs == 0).all()
assert np.logical_xor(0 == ub_pi, ub_rhs - x == 0).all()
