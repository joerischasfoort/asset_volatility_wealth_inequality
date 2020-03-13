""""This file contains functions which can be used for an evolutionary algorithm"""
import random
import bisect
import numpy as np
#import simfinmodel
import init_objects
from functions.stylizedfacts import *
from multiprocessing import Pool
from functions.helpers import *


def average_fitness(population):
    total_cost = 0
    for individual in population:
        total_cost += individual.cost
    return total_cost / (float(len(population)))


def cost_function(observed_values, average_simulated_values):
    """
    Simple cost function to calculate average squared deviation of simulated values from observed values
    :param observed_values: dictionary of observed stylized facts
    :param average_simulated_values: dictionary of corresponding simulated stylized facts
    :return:
    """
    score = 0
    for key in observed_values:
        score += np.true_divide((observed_values[key] - average_simulated_values[key]), observed_values[key])**2

    if np.isnan(score):
        return np.inf
    else:
        return score


def quadratic_loss_function(m_sim, m_emp, weights):
    """
    Quadratic loss function to calculate deviation of simulated values from observed values
    :param m_sim:
    :param m_emp:
    :param weights:
    :return:
    """
    score = np.dot(np.dot(np.array([m_sim - m_emp]), weights), np.array([m_sim - m_emp]).transpose())[0][0]

    if np.isnan(score):
        return np.inf
    else:
        return score


def m_fitness(mc_rets, mc_p, mc_f, emp_m, weights):
    """
    Calculate the model fitness based on a quadtratic loss function
    :param mc_rets: pd.DataFrame of simulation returns
    :param mc_p: pd.DataFrame of simulation prices
    :param mc_f: pd.DataFrame of simulation fundamentals
    :param emp_m: np.Array of empirical moments
    :param weights: np.Matrix of weights
    :return:
    """
    first_order_autocors = []
    autocors1 = []
    autocors5 = []
    mean_abs_autocor = []
    kurtoses = []
    spy_abs_auto10 = []
    spy_abs_auto25 = []
    spy_abs_auto50 = []
    spy_abs_auto100 = []
    cointegrations = []
    for col in mc_rets:
        first_order_autocors.append(autocorrelation_returns(mc_rets[col][1:], 25))
        autocors1.append(mc_rets[col][1:].autocorr(lag=1))
        autocors5.append(mc_rets[col][1:].autocorr(lag=5))
        mean_abs_autocor.append(autocorrelation_abs_returns(mc_rets[col][1:], 25))
        kurtoses.append(mc_rets[col][2:].kurtosis())
        spy_abs_auto10.append(mc_rets[col][1:].abs().autocorr(lag=10))
        spy_abs_auto25.append(mc_rets[col][1:].abs().autocorr(lag=25))
        spy_abs_auto50.append(mc_rets[col][1:].abs().autocorr(lag=50))
        spy_abs_auto100.append(mc_rets[col][1:].abs().autocorr(lag=100))
        cointegrations.append(cointegr(mc_p[col][1:], mc_f[col][1:])[0])

    stylized_facts_sim = np.array([
        np.mean(first_order_autocors),
        np.mean(autocors1),
        np.mean(autocors5),
        np.mean(mean_abs_autocor),
        np.mean(kurtoses),
        np.mean(spy_abs_auto10),
        np.mean(spy_abs_auto25),
        np.mean(spy_abs_auto50),
        np.mean(spy_abs_auto100),
        np.mean(cointegrations)
    ])

    # return cost
    return quadratic_loss_function(stylized_facts_sim, emp_m, weights)


class Individual:
    """The order class can represent both bid or ask type orders"""
    def __init__(self, parameters, stylized_facts, cost):
        self.parameters = parameters
        self.stylized_facts = stylized_facts
        self.cost = cost

    def __lt__(self, other):
        """Allows comparison to other individuals based on its cost (negative fitness)"""
        return self.cost < other.cost