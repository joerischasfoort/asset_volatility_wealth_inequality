from init_objects import *
from model import *
import time

start_time = time.time()

# # 1 setup parameters
parameters = {"fundamental_value": 166,
          "trader_sample_size": 10, "n_traders": 1000,
          "ticks": 500, "std_fundamental": 180.0,#0.530163128919286,
          "std_noise": 0.10696588473846724,
          "w_random": 0.1,
          "init_stocks": 50, "base_risk_aversion": 1.0,
          'spread_max': 0.004087, "horizon": 200,
          "fundamentalist_horizon_multiplier": 0.2,
          "mean_reversion": 1.2,
          # fixed / not modelled parameters
          "strat_share_chartists": 0.0,
          "mutation_intensity": 0.0,
          "average_learning_ability": 0.0,
          "trades_per_tick": 1
}

# 2 initialise model objects
traders, orderbook = init_objects_distr(parameters, seed=0)

# 3 simulate model
traders, orderbook = volatility_inequality_model2(traders, orderbook, parameters, seed=0)


print("The simulations took", time.time() - start_time, "to run")