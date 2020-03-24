from functions.indirect_calibration import *
import time
from multiprocessing import Pool
import json
import numpy as np

np.seterr(all='ignore')

start_time = time.time()

# INPUT PARAMETERS
LATIN_NUMBER = 0
NRUNS = 4
BURN_IN = 0
CORES = NRUNS # set the amount of cores equal to the amount of runs

problem = {'num_vars': 1,
 'names': ['white_noise'],
 'bounds': [[0.0007913977712213694, 0.07913977712213693]]}

with open('hypercube.txt', 'r') as f:
    latin_hyper_cube = json.loads(f.read())

# Bounds
LB = [x[0] for x in problem['bounds']]
UB = [x[1] for x in problem['bounds']]

init_parameters = latin_hyper_cube[LATIN_NUMBER]

params = {'ticks': 251,
 'n_traders': 20,
 'init_stocks': 81,
 'init_price': 287.686015936255,
 'white_noise': 0.007913977712213693}


def simulate_a_seed(seed_params):
    """Simulates the model for a single seed and outputs the associated cost"""
    seed = seed_params[0]
    params = seed_params[1]

    obs = []
    # run model with parameters
    traders, orderbook = init_objects_model(params, seed)
    traders, orderbook = volatility_inequality_model(traders, orderbook, params, seed)
    obs.append(orderbook)

    # store simulated stylized facts
    mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume = organise_data(
        obs, burn_in_period=BURN_IN)

    stdevs = []
    first_order_autocors = []
    for col in mc_returns:
        stdevs.append(mc_returns[col][1:].std())
        first_order_autocors.append(autocorrelation_returns(mc_returns[col][1:], 25))

    stylized_facts_sim = np.array([
        np.mean(stdevs),
        np.mean(first_order_autocors),
    ])

    W = np.load('distr_weighting_matrix.npy')  # if this doesn't work, use: np.identity(len(stylized_facts_sim))

    empirical_moments = np.array([0.00791398, -0.01593133])

    # calculate the cost
    cost = quadratic_loss_function(stylized_facts_sim, empirical_moments, W)
    return cost


def pool_handler():
    p = Pool(CORES) # argument is how many process happening in parallel
    list_of_seeds = [x for x in range(NRUNS)]

    def model_performance(input_parameters):
        """
        Simple function calibrate uncertain model parameters
        :param input_parameters: list of input parameters
        :return: average cost
        """
        # convert relevant parameters to integers
        new_input_params = []
        for idx, par in enumerate(input_parameters):
            new_input_params.append(par)

        # update params
        uncertain_parameters = dict(zip(problem['names'], new_input_params))
        params = {'ticks': 251,
                  'n_traders': 20,
                  'init_stocks': 81,
                  'init_price': 287.686015936255,
                  'white_noise': 0.007913977712213693}
        params.update(uncertain_parameters)

        list_of_seeds_params = [[seed, params] for seed in list_of_seeds]

        costs = p.map(simulate_a_seed, list_of_seeds_params) # first argument is function to execute, second argument is tuple of all inputs TODO uncomment this

        return np.mean(costs)

    output = constrNM(model_performance, init_parameters, LB, UB, maxiter=6, full_output=True)

    with open('estimated_params.json', 'w') as f:
        json.dump(list(output['xopt']), f)

    print('All outputs are: ', output)


if __name__ == '__main__':
    pool_handler()
    print("The simulations took", time.time() - start_time, "to run")
