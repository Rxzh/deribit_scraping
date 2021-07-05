from Cryptocurrency_pricing.Models.common_all import Model
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize, least_squares
from random import random
import cmath
import math


class Heston(Model):
    def __init__(self, raw_dataframe, option_type = 'B'):
        super().__init__(raw_dataframe,option_type)
        self.model_name = 'heston'
    
    def initialize(self, theta = None, reset = False):

        if reset:
            #self.theta = np.array([0.41055433, 2.87649559, 1.0035074 , 1.14520439, 2.15878211])
            self.theta = np.array([0.90260584, 0.42127592, 0.13607334, 0.37337941, 0.20804514])
        
        elif theta is None:
            raise Exception('theta and reset can\'t both be None')
        else:
            if len(theta) != 5:
                raise ValueError('theta must be 5-dimensional array for the Merton model')
            else:
                self.theta = theta

        self.M = 100  #why ?
        self.deg = 32 #why ?

    ##### HESTON #################################
    
    # SECTION 1 - Functions, which are necessary to compute price of the european option for given parameters.
    def ksi(self, u):
        return self.theta[3] - self.theta[4]*self.theta[2]*u*1j
    def d(self, u):
        return cmath.sqrt(self.ksi(u)*self.ksi(u) + math.pow(self.theta[4], 2)*(u*u + u*1j))
    def g1(self, u):
        return (self.ksi(u) + self.d(u))/(self.ksi(u) - self.d(u))
    def A1(self, u, t):
        return (u*u + 1j*u)*cmath.sinh(self.d(u)*t/2)
    def A2(self, u, t):
        return self.d(u)*cmath.cosh(self.d(u)*t/2)/self.theta[0] + self.ksi(u)*cmath.sinh(self.d(u)*t/2)/self.theta[0]
    def A(self, u, t):
        return self.A1(u, t)/self.A2(u, t)
    def B(self, u, t):
        return self.d(u)*cmath.exp(self.theta[3]*t/2)/(self.A2(self.theta, u, t)*self.theta[0])
    def D(self, u, t):
        return cmath.log(self.d(u)/self.theta[0]) + self.theta[3]*t/2 - cmath.log(self.A2(u, t))
    # Equation (18) p. 9 - characteristic function, which we are going to use in out project
    def char_function(self, u, t, S_0, r):
        return cmath.exp(1j*u*(np.log(S_0*np.exp(r*t)/S_0)) - t*self.theta[3]*self.theta[1]*self.theta[2]*1j*u/self.theta[4] - self.A(u, t) + \
                        2*self.theta[3]*self.theta[1]*self.D(u, t)/math.pow(self.theta[4], 2))

    # integrate_char_function - integrals computed by means of Gauss-Legendre Quadrature
    def integrate_char_function(self, K, t, S_0, r, i):
        x, w = np.polynomial.legendre.leggauss(self.deg)
        u = (x[0]+1)*0.5*self.M
        value = w[0]*cmath.exp(-1j*u*np.log(K/S_0))/(1j*u)*self.char_function(u - i, t, S_0, r)
        for j in range(1, self.deg): # deg - number of nodes
            u = (x[j] + 1)*0.5*self.M
            value = value + w[j]*cmath.exp(-1j*u*np.log(K/S_0))/(1j*u)*self.char_function(u - i, t, S_0, r)
        value = value*0.5*self.M
        return value.real
    # HestonPrice - Equation (9)



    def Price(self,S,K, T, r=0.0, CallPutFlag = 'C', sigma = None): #sigma is None for Heston Price
        if CallPutFlag == 'C':
            return 1/S *( 
                        (S - np.exp(-r*T)*K)/2 + np.exp(-r*T)/np.pi*(   S*self.integrate_char_function(K, T, S, r, 1j) - \
                                                                        K*self.integrate_char_function(K, T, S, r, 0))
                        )
        elif CallPutFlag == 'P': #TODO
            return 1/S *( 
                        (- S + np.exp(-r*T)*K)/2 + np.exp(-r*T)/np.pi*(   S*self.integrate_char_function(K, T, S, r, 1j) - \
                                                                        K*self.integrate_char_function(K, T, S, r, 0)))


    # SECTION 2 - functions, which are necessary to compute gradient of characteristic function
    def h_1(self, u, t):
        return -self.A(u, t)/self.theta[0]
    def h_2(self, u ,t):
        return 2*self.theta[3]*self.D(u, t)/math.pow(self.theta[4], 2) - t*self.theta[3]*self.theta[2]*1j*u/self.theta[4]
    def h_3(self, u, t):
        return - self.A_rho(u, t) + 2*self.theta[3]*self.theta[1]*(self.d_rho(u) - self.d(u)*self.A2_rho(u, t)/self.A2(u,t))/ \
                                    (self.theta[4]*self.theta[4]*self.d(u)) - t*self.theta[3]*self.theta[1]*1j*u/self.theta[4]
    def h_4(self, u, t):
        return self.A_rho(u, t)/(self.theta[4]*1j*u) + 2*self.theta[1]*self.D(u, t)/(self.theta[4]*self.theta[4]) + \
            2*self.theta[3]*self.theta[1]*self.B_kappa(u, t)/(self.theta[4]*self.theta[4]*self.B(u, t)) - \
            t*self.theta[1]*self.theta[2]*1j*u/self.theta[4]
    def h_5(self, u, t):
        return - self.A_sigma(u, t) - 4*self.theta[3]*self.theta[1]*self.D(u, t)/(math.pow(self.theta[4], 3)) + \
            2*self.theta[3]*self.theta[1]*(self.d_rho(u) - self.d(u)*self.A2_sigma(u, t)/self.A2(u, t))/ \
            (self.theta[4]*self.theta[4]*self.d(u)) + t*self.theta[3]*self.theta[1]*self.theta[2]*1j*u/(self.theta[4]*self.theta[4])
    def d_rho(self, u):
        return - self.ksi(u)*self.theta[4]*1j*u/self.d(u)
    def A2_rho(self, u, t):
        return - self.theta[4]*1j*u*(2 + t*self.ksi(u))*(self.ksi(u)*cmath.cosh(self.d(u)*t/2) + \
                                                    self.d(u)*cmath.sinh(self.d(u)*t/2))/(2*self.d(u)*self.theta[0])
    def B_rho(self, u, t):
        return cmath.exp(self.theta[3]*t/2)*(self.d_rho(u)/self.A2(u, t) - \
                                        self.A2_rho(u, t)/(self.A2(u,t)*self.A2(u,t)))/self.theta[0]
    def A1_rho(self, u, t):
        return - 1j*u*(u*u + 1j*u)*t*self.ksi(u)*self.theta[4]*cmath.cosh(self.d(u)*t/2)/(2*self.d(u))
    def A_rho(self, u, t):
        return self.A1_rho(u, t)/self.A2(u, t) - self.A2_rho(u, t)*self.A(u, t)/self.A2(u, t)
    def A_kappa(self, u, t):
        return 1j*self.A_rho(u, t)/(u*self.theta[4])
    def B_kappa(self, u, t):
        return self.B_rho(u, t)*1j/(self.theta[4]*u) + t*self.B(u, t)/2
    def d_sigma(self, u):
        return (self.theta[2]/self.theta[4] - 1/self.ksi(u))*self.d_rho(u) + self.theta[4]*u*u/self.d(u)
    def A1_sigma(self, u, t):
        return (u*u + 1j*u)*t*self.d_sigma(u)*cmath.cosh(self.d(u)*t/2)/2
    def A2_sigma(self, u, t):
        return self.theta[2]*self.A2_rho(u, t)/self.theta[4] - (2 + t*self.ksi(u))*self.A1_rho(u, t)/ \
                                                    (1j*u*t*self.ksi(u)*self.theta[0]) + self.theta[4]*t*self.A1(u, t)/(2*self.theta[0])
    def A_sigma(self, u, t):
        return self.A1_sigma(u, t)/self.A2(u, t) - self.A(u, t)*self.A2_sigma(u, t)/self.A2(u, t)
    def h(self, u, t, which):
        if which == 1:
            return self.h_1(u, t)
        if which == 2:
            return self.h_2(u, t)
        if which == 3:
            return self.h_3(u, t)
        if which == 4:
            return self.h_4(u, t)
        if which == 5:
            return self.h_5(u, t)
    # integrate_grad_function - integrals computed by means of Gauss-Legendre Quadrature
    def integrate_grad_function(self, K, t, S_0, r, i, which):
        x, w = np.polynomial.legendre.leggauss(self.deg)
        u = (x[0] + 1)*0.5*self.M
        value = w[0]*cmath.exp(-1j*u*np.log(K/S_0))/(1j*u)*self.char_function(u - i, t, S_0, r)*self.h(u - i, t, which)
        for j in range(1, self.deg):
                u = (x[j] + 1)*0.5*self.M
                value = value + w[j]*cmath.exp(-1j*u*np.log(K/S_0))/(1j*u)*self.char_function(u - i, t, S_0, r)* \
                                self.h(u - i, t, which)
        return value.real
    # grad_heston_price - Equation (22)
    def grad_heston_price(self, t, K, S_0, r):
        first_int = np.array(self.integrate_grad_function(K, t, S_0, r, 1j, 1))
        second_int = np.array(self.integrate_grad_function(K, t, S_0, r, 0, 1))
        for i in range(2, 6):
            first_int  = np.append(first_int,  self.integrate_grad_function(K, t, S_0, r, 1j, i))
            second_int = np.append(second_int, self.integrate_grad_function(K, t, S_0, r, 0, i))
        return (np.exp(-r*t)/np.pi)*(first_int - K*second_int)

    def r_function(self, x):
        raise NotImplementedError("Not implemented")
        if False:
            self.initialize(theta = x)
            r_vector = np.vectorize(self.HestonPrice)(self.df['K'], self.df['_T'], self.df['S']) - self.df['mid']
            #for i in range(1, len(mkt_data)):
            #    r_vector = np.append(r_vector, self.HestonPrice(mkt_data[i, 1], mkt_data[i, 2], S_0, r) - \
            #                        mkt_data[i, 0])
            return r_vector
        
