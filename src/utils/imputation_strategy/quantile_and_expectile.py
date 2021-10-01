# -*- coding:utf-8 -*-
import numpy as np
from scipy.optimize import minimize, root

y1 = [1, 1, 2, 2, 4, 4, 6, 8, 8, 8, 8, 10, 10]
y2 = [1, 2, 6, 8, 10]
y3 = [1, 1, 1.5, 2, 6, 6, 6, 8, 8, 8, 9, 10, 10]

# quantiles
q = [0.25, 0.5, 0.75]
q1 = np.quantile(y1, q)
q2 = np.quantile(y2, q)
q3 = np.quantile(y3, q)

cum_density = [0.25, 0.5, 0.75]

print('-' * 10 + ' size ' + '-' * 10)
print("y1: " + str(len(y1)))
print("y2: " + str(len(y2)))
print("y3: " + str(len(y3)))

print('-' * 10 + ' mean ' + '-' * 10)
print("y1: " + str(np.mean(y1)))
print("y2: " + str(np.mean(y2)))
print("y3: " + str(np.mean(y3)))

print('-' * 10 + ' std ' + '-' * 10)
print("y1: " + str(np.std(y1)))
print("y2: " + str(np.std(y2)))
print("y3: " + str(np.std(y3)))

print('-' * 10 + ' Quantiles ' + '-' * 10)
print("y1: " + str(q1))
print("y2: " + str(q2))
print("y3: " + str(q3))


# =============================================
# minimization
# =============================================
def min_objective_fc(x, tau, samples):
    diff = samples - x
    expected_loss = np.mean(np.where(diff > 0, tau, 1 - tau) * np.square(diff))

    return expected_loss


def expectiles_min_method(samples, taus):
    optimization_results = []
    for each_tau in taus:
        results = minimize(min_objective_fc, args=(each_tau, samples), x0=np.mean(samples), method="SLSQP")
        optimization_results.append(results.x[0])

    return optimization_results


print('-' * 10 + ' Expectile Minimization Method ' + '-' * 10)
print("y1: " + str(expectiles_min_method(y1, q)))
print("y2: " + str(expectiles_min_method(y2, q)))
print("y3: " + str(expectiles_min_method(y3, q)))

# =============================================
# imputation strategy
# =============================================
print('=' * 10 + ' imputation strategy ' + '=' * 10)

# expectiles = [3.67, 5.40, 7.00]
e = [4, 6, 7]
cum_density = [0.25, 0.5, 0.75]
assert len(e) == len(cum_density)


def min_objective_fc(x, expectiles):
    vals = 0
    for idx, each_expecile in enumerate(expectiles):
        diff = x - each_expecile
        diff = np.where(diff > 0, - cum_density[idx] * diff, (cum_density[idx] - 1) * diff)

        vals += np.square(np.mean(diff))

    return vals


def imputation_minimization():
    sample_size = 5
    bnds = tuple((1, 10) for _ in range(sample_size))  # for each x_i, set a boundary
    search_start = np.linspace(start=1, stop=10, num=sample_size)  # also defines the number of samples to be imputed
    results = minimize(min_objective_fc, args=e, method="SLSQP", bounds=bnds, x0=search_start)
    print(results)


def root_objective_fc(x, expectiles):
    vals = []
    for idx, each_expectile in enumerate(expectiles):
        diff = x - each_expectile
        diff = np.where(diff > 0, - cum_density[idx] * diff, (cum_density[idx] - 1) * diff)
        vals.append(np.mean(diff))
    return vals


def imputation_root():
    # the default root method is "hybr", it requires the input shape of x to be the same as
    # the output shape of the root results
    sample_size = 5
    search_start = np.linspace(start=1, stop=10, num=sample_size)  # also defines the number of samples to be imputed
    results = root(root_objective_fc, args=e, x0=search_start, method="hybr")
    print(results)


print('-' * 10 + ' Minimization Method ' + '-' * 10)
imputation_minimization()
print('-' * 10 + ' Root Method ' + '-' * 10)
imputation_root()
