from objects.trader import *
from objects.orderbook import *
import numpy as np
from functions.helpers import calculate_covariance_matrix


def init_objects_model(parameters, seed):
    """
    Init object for the distribution version of the model
    :param parameters:
    :param seed:
    :return:
    """
    np.random.seed(seed)

    traders = []
    n_traders = parameters["n_traders"]

    historical_stock_returns = np.random.normal(0, parameters["white_noise"], parameters['ticks'])

    for idx in range(n_traders):
        init_stocks = int(np.random.uniform(0, parameters["init_stocks"]))
        init_money = np.random.uniform(0, (parameters["init_stocks"] * parameters['init_price']))
        init_covariance_matrix = calculate_covariance_matrix(historical_stock_returns, parameters["white_noise"])

        lft_vars = TraderVariables(init_money, init_stocks, init_covariance_matrix,
                                   parameters['init_price'])

        lft_params = TraderParameters(parameters['ticks'])
        lft_expectations = TraderExpectations(parameters['init_price'])
        traders.append(Trader(idx, lft_vars, lft_params, lft_expectations))

    orderbook = LimitOrderBook(parameters['init_price'], parameters["white_noise"],
                               parameters['ticks'], parameters['ticks'])

    # initialize order-book returns for initial variance calculations
    orderbook.returns = list(historical_stock_returns)

    return traders, orderbook
