from init_objects import *
from model import *
import time

start_time = time.time()

# # 1 setup parameters
parameters = {
    'ticks': 601,
    'n_traders': 50,
    'init_stocks': 81,
    'init_price': 1112.2356754564078,
    'white_noise': 0.05149715506250338,
}

# 2 initialise model objects
traders, orderbook = init_objects_model(parameters, seed=0)

# 3 simulate model
traders, orderbook = volatility_inequality_model(traders, orderbook, parameters, seed=0)


print("The simulations took", time.time() - start_time, "to run")