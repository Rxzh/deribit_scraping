from Cryptocurrency_pricing.Models.common_all import Model
import numpy as np
from scipy.optimize import minimize, least_squares
from random import random


class Merton(Model):
    def __init__(self, raw_dataframe, option_type = 'B'):
        super().__init__(raw_dataframe,option_type)
        self.model_name = 'merton'

    ############# MERTON #################

    def initialize(self,theta = None, reset = False):
        """
        Initialise les paramètres du Modèle de Merton.
        =============================
        Input:  theta : List, [m,v,lam]
                    m     : Float, Jump Mean
                    lam   : Float, Lambda
                    v     : Float, Jump Standard Deviation
                reset : Bool, Pour choisir de reset vers les meilleurs paramètres calibrés
        =============================
        Output: None
        """

        if reset:

            #self.m = 1.68542729380368
            #self.v = 0.00036428123601998366
            #self.lam = 4.682520794818356e-05

            self.m, self.v, self.lam = 9.17579001e-01, 3.50404358e-02, 2.89575783e-04

        elif theta is None:
            raise Exception('theta and reset can\'t both be None')
        else:
            if len(theta) != 3:
                raise ValueError('theta must be 3-dimensional array for the Merton model')
            else:
                self.theta = theta

            self.m = theta[0]
            self.v = theta[1]
            self.lam = theta[2]
            







    def Price(self, S,K,T,sigma,r=0.01, q = 0, CallPutFlag = 'C'):
        """
        Retourne le prix Merton d'une Option Européene
        =============================
        Input:  CallPutFlag : {'C','P'} / caractérise le type d'option
                K : Strike 
                T : Maturité
                S : Prix du sous-jacent
                v : Volatilité
                r : Taux d'intérêt 
                #TODO q
        =============================
        Output: Prix BS (float)
        """
        p = 0
        for k in range(40):
            r_k = r - self.lam*(self.m-1) + (k*np.log(self.m) ) / T
            sigma_k = np.sqrt( sigma**2 + (k* self.v** 2) / T)
            k_fact = np.math.factorial(k)
            p += (np.exp(-self.m*self.lam*T) * (self.m*self.lam*T)**k / (k_fact)) * self.BlackScholesPrice(sigma = sigma_k, CallPutFlag=CallPutFlag, S=S, K=K, T=T, r=r_k)
        return p 




