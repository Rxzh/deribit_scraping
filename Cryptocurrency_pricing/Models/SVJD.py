from Cryptocurrency_pricing.Models.common_all import Model
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize, least_squares
from random import random
import cmath
import math


class SVJD(Model):
    def __init__(self, raw_dataframe, option_type = 'B'):
        super().__init__(raw_dataframe,option_type)
        self.model_name = 'svjd' #stochastic volatility jump diffusion