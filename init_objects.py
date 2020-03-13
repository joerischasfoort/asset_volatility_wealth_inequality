from objects.trader import *
from objects.orderbook import *
import random
import numpy as np
from functions.helpers import calculate_covariance_matrix, div0


def init_objects_model(parameters, seed):
    """
    Init object for the distribution version of the model
    :param parameters:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    traders = []
    n_traders = parameters["n_traders"]

    weight_f = 0 #(1 - parameters['strat_share_chartists']) * (1 - parameters['w_random'])
    weight_c = 0 #parameters['strat_share_chartists'] * (1 - parameters['w_random'])

    f_points = int(weight_f * 100 * n_traders)
    c_points = int(weight_c * 100 * n_traders)
    r_points = int(1 * 100 * n_traders) # TODO 1 used to be parameters['w_random']

    # create list of strategy points, shuffle it and divide in equal parts
    strat_points = ['f' for f in range(f_points)] + ['c' for c in range(c_points)] + ['r' for r in range(r_points)]
    random.shuffle(strat_points)
    agent_points = np.array_split(strat_points, n_traders)

    #max_horizon = parameters['horizon'] * 2  # this is the max horizon of an agent if 100% fundamentalist
    max_horizon = parameters['ticks']
    #historical_stock_returns = np.random.normal(0, parameters["std_fundamental"], max_horizon) TODO add back?
    historical_stock_returns = np.random.normal(0, parameters["std_noise"], max_horizon)

    for idx in range(n_traders):
        weights = []
        for typ in ['f', 'c', 'r']:
            weights.append(list(agent_points[idx]).count(typ) / float(len(agent_points[idx])))

        init_stocks = int(np.random.uniform(0, parameters["init_stocks"]))
        init_money = np.random.uniform(0, (parameters["init_stocks"] * parameters['init_price']))

        # If there are chartists (c) & fundamentalists (f) in the model, keep track of the fraction between c & f.
        if weights[2] < 1.0:
            c_share_strat = div0(weights[1], (weights[0] + weights[1]))
        else:
            c_share_strat = 0.0

        # initialize co_variance_matrix
        #init_covariance_matrix = calculate_covariance_matrix(historical_stock_returns, parameters["std_fundamental"])
        init_covariance_matrix = calculate_covariance_matrix(historical_stock_returns, parameters["std_noise"])

        lft_vars = TraderVariablesDistribution(weights[0], weights[1], weights[2], c_share_strat,
                                               init_money, init_stocks, init_covariance_matrix,
                                               parameters['init_price'])

        # determine heterogeneous horizon and risk aversion based on
        #individual_horizon = np.random.randint(10, parameters['horizon'])
        individual_horizon = parameters['ticks']

        #individual_risk_aversion = abs(np.random.normal(parameters["base_risk_aversion"], parameters["base_risk_aversion"] / 5.0))#parameters["base_risk_aversion"] * relative_fundamentalism
        individual_risk_aversion = 1.0
        individual_learning_ability = 0.0#min(abs(np.random.normal(parameters['average_learning_ability'], 0.1)), 1.0) #TODO what to do with std_dev

        lft_params = TraderParametersDistribution(individual_horizon, individual_risk_aversion,
                                                  individual_learning_ability, parameters['spread_max'])
        lft_expectations = TraderExpectations(parameters['init_price'])
        traders.append(Trader(idx, lft_vars, lft_params, lft_expectations))

    orderbook = LimitOrderBook(parameters['init_price'], parameters["std_noise"], #std_fundamental TODO used to be this
                               max_horizon,
                               parameters['ticks'])

    # initialize order-book returns for initial variance calculations
    orderbook.returns = list(historical_stock_returns)

    return traders, orderbook
