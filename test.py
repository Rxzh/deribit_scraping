
"""
class Model:
    def __init__(self, option_type = 'B'):
        self.model = None
        self.initialize(reset = True) # initialize les params


    def initialize(self, theta=None, reset = False):
        self.var = 0


class BlackScholes(Model):
    
    def __init__(self, option_type = 'B'):
        super().__init__(option_type)
        self.model = 'black-scholes'


    def initialize(self,reset = False):
        self.var = 2



model = BlackScholes()
print(model.var)
"""

"""
from random import random
import numpy as np 

bounds = ((0.01, 2), (1e-5, np.inf) , (0, 5))
x0 = [  bound[0]+random()*(bound[1]-bound[0]) 
        if np.inf not in bound 
        else random() 
        for bound in bounds
    ]

print(x0)

"""

