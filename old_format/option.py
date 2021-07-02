from utils import *
#from scipy import optimize
from scipy.stats import norm
from scipy.optimize import minimize, least_squares
import numpy as np
from random import random
import cmath
import math

class Option:
    def __init__(self, raw_dataframe, option_type = 'B'):
        """
        Initialise la classe Option,
    
        ======================
        Input:  raw_dataframe : DataFrame brut scrapé depuis deribit via la classe Scaper
                option_type   : {'B':,'C','P'} / caractérise le type d'option qu'on veut garder depuis le DataFrame brut. 
        =======================
        Output: None
        """

        self.option_type = option_type
        self.df = pipeline(raw_dataframe, option_type=self.option_type)
    
        self.init_merton(reset = True)
        """
        self.m = 1.68542729380368
        self.v = 0.00036428123601998366
        self.lam = 4.682520794818356e-05
        """


    def CND(self,X):
        """
        Retourne la fonction de répartition de la loi Normale centrée réduite évaluée en x
        """
        return float(norm.cdf(X))

    def BlackScholesPrice(self,v,CallPutFlag = 'C',S = 100.,K = 100.,T = 1.,r = 0.01):
        """
        Retourne le prix Black-Scholes d'une Option Européene
        =============================
        Input:  CallPutFlag : {'C','P'} / caractérise le type d'option
                K : Strike 
                T : Maturité
                S : Prix du sous-jacent
                v : Volatilité
                r : Taux d'intérêt 
        =============================
        Output: Prix BS (float)
        """
        try:
            d1 = (np.log(S/K)+(r+v*v/2.)*T)/(v*np.sqrt(T))
            d2 = d1-v*np.sqrt(T)

            if CallPutFlag == 'C':
                return S*self.CND(d1)-K*np.exp(-r*T)*self.CND(d2)
            elif CallPutFlag == 'P':
                return K*np.exp(-r*T)*self.CND(-d2)-S*self.CND(-d1)
            else:
                print("Mauvais type d'option : {'P','C'}")
        except: 
            return 0
        
    def dP_dK(self,S,K,T,sigma,right='C',r=0.01):
        """
        Retourne la dérivée partielle du prix par rapport au strike        
        =============================
        Input:  right : {'C','P'} / caractérise le type d'option
                K : Strike 
                T : Maturité
                S : Prix du sous-jacent
                v : Volatilité
                r : Taux d'intérêt 
        =============================
        Output: Float
        """
        try:
            d1 = (np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))
            d2 = d1-sigma*np.sqrt(T)
            if right == 'C':
                return -S/K*1/np.sqrt(2*np.pi)*np.exp(-.5*d1**2) + np.exp(r*T)*(-self.CND(d2)+1/np.sqrt(2*np.pi)*np.exp(-.5*d2**2))
            else:
                return +S/K*1/np.sqrt(2*np.pi)*np.exp(-.5*d1**2) + np.exp(r*T)*(self.CND(-d2)+1/np.sqrt(2*np.pi)*np.exp(-.5*d2**2))
        except:
            return 0.0
    
    def d2P_dK2(self,S,K,T,sigma,right='C',r=0.01):
        """
        Retourne la dérivée partielle seconde du prix par rapport au strike
        =============================
        Input:  right : {'C','P'} / caractérise le type d'option
                K : Strike 
                T : Maturité
                S : Prix du sous-jacent
                v : Volatilité
                r : Taux d'intérêt 
        =============================
        Output: Float
        """
        try:
            d1 = (np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))
            d2 = d1-sigma*np.sqrt(T)
            if right == 'C':
                return  (S/K**2 * 1/np.sqrt(2*np.pi) *np.exp(-.5*d1**2))*(1-d1)   \
                        + 1/K*1/np.sqrt(2*np.pi)*np.exp(-r*T-.5*d2**2)            \
                        - d2/K**2 /np.sqrt(2*np.pi)*np.exp(-r*T-.5*d2**2)
            else:
                return  -(S/K**2 * 1/np.sqrt(2*np.pi) *np.exp(-.5*d1**2))*(1-d1)  \
                        - 1/K*1/np.sqrt(2*np.pi)*np.exp(-r*T-.5*d2**2)            \
                        - d2/K**2 /np.sqrt(2*np.pi)*np.exp(-r*T-.5*d2**2)
        except:
            return 0.0

    def dP_dT(self,S,K,T,sigma,right='C',r=0.01):
        """
        Retourne la dérivée partielle du prix par rapport à la date d'expiration
        Il s'agit de l'opposé du theta.
        =============================
        Input:  right : {'C','P'} / caractérise le type d'option
                K : Strike 
                T : Maturité
                S : Prix du sous-jacent
                v : Volatilité
                r : Taux d'intérêt 
        =============================
        Output: Float
        """
        try:
            d1 = (np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))
            d2 = d1-sigma*np.sqrt(T)
            if right == 'C':
                return  (S*sigma/2 * 1/np.sqrt(2*np.pi*T) *np.exp(-.5*d1**2)) + r*K*np.exp(-r*T)*self.CND(d2)

            else:
                return  (S*sigma/2 * 1/np.sqrt(2*np.pi*T) *np.exp(-.5*d1**2)) - r*K*np.exp(-r*T)*self.CND(-d2)
        except:
            return 0.0
        
    """
    def implied_v(self, P, S , K, T, r=0.0, right = 'C', inc = 0.0001):


        f = lambda x: self.BlackScholesPrice(x,CallPutFlag=right,S=S,K=K,T=T,r=r)-P
        return optimize.brentq(f,0.,5.,maxiter=500)
    """

    
    def implied_v_merton(self, P, S , K, T, r=0.0, right = 'C'):
        """
        Retourne la volatilité implicite pour le modèle de Merton
        =============================
        Input:  right : {'C','P'} / caractérise le type d'option
                K : Strike 
                T : Maturité
                S : Prix du sous-jacent
                P : Prix de l'option
                r : Taux d'intérêt 
        =============================
        Output: Float, volatilité
        """
        x,y,n = 0,1,20
        for _ in range(n):
            z = (x+y)/2
            if P > self.MertonPrice(S=S,K=K,T=T,sigma=z/(1-z),r=r, CallPutFlag = right):
                x = z
            else:
                y = z
        z = (x+y)/2
        sigma = z/(1-z)
        
        return sigma

    
    def implied_v_bs(self, P, S , K, T, r=0.0, right = 'C'):
        """
        Retourne une volatilité implicite
        ============================
        Input:
        ============================
        Output:
        """
        x,y,n = 0,1,20
        for _ in range(n):
            z = (x+y)/2
            if P > self.BlackScholesPrice(S=S,K=K,T=T,v=z/(1-z),r=r, CallPutFlag = right):
                x = z
            else:
                y = z
        z = (x+y)/2
        sigma = z/(1-z)
        
        return sigma
    
    """
    def implied_v(self, P, S, K, T, r = 0.01, right = 'C', inc = 0.001):
        f = lambda x: self.BlackScholesPrice(x,CallPutFlag=right,S=S,K=K,T=T,r=r)-P
        return optimize.brentq(f,0.,120.,maxiter=1000)
    """
    
    def local_v(self,S,K,T,sigma,right='C',r=0.01): 
        """
        Retourne une volatilité locale
        ============================
        Input:
        ============================
        Output:
        """
        try:
            num = 2*(self.dP_dT(S,K,T,sigma,right,r) + r*K* self.dP_dK(S,K,T,sigma,right,r))
            denom = K**2 * self.d2P_dK2(S,K,T,sigma,right,r)
            return np.sqrt(num/denom)
        except:
            return 0.0
        
        
    def append_imp_vol_to_df(self, r=0.01):
        """
        Rajoute la colonne 'I_VOL' des volatilités implicite au dataframe
        ==================
        Input:  None
        ==================
        Output: None
        """
        
        self.df['I_VOL_MERTON'] = np.vectorize(self.implied_v_merton)(P = self.df['mid'].astype(float),
                                            right = self.df['option_type'],
                                            S = self.df['S'].astype(float),
                                            K = self.df['K'].astype(float),
                                            T = self.df['_T'].astype(float)
                                            )
    
        self.df['I_VOL_BS'] = np.vectorize(self.implied_v_bs)(P = self.df['mid'].astype(float),
                                            right = self.df['option_type'],
                                            S = self.df['S'].astype(float),
                                            K = self.df['K'].astype(float),
                                            T = self.df['_T'].astype(float)
                                            )
        
        self.df['IV_moneyness'] = self.df['S']/self.df['I_VOL_BS']
        
    def append_loc_vol_to_df(self, r=0.01):
        """
        Rajoute la colonne 'L_VOL' des volatilités implicite au dataframe
        Si la colonne 'I_V' n'a pas déjà été créée, elle aussi est créée.
        ==================
        Input:  None
        ==================
        Output: None
        """
        
        if 'I_VOL_BS' not in self.df.columns:
            self.append_imp_vol_to_df(r)
            
        self.df['L_VOL'] = np.vectorize(self.local_v)(  right = self.df['option_type'],
                                                        S = self.df['S'].astype(float),
                                                        K = self.df['K'].astype(float),
                                                        T = self.df['_T'].astype(float),
                                                        sigma = self.df['I_VOL_BS'].astype(float)
                                                        )

    def append_BS_price(self):
        """
        Ajoute les prix de Black-Scholes au DataFrame principal
        ==================
        Input:  None
        ==================
        Output: None
        """
        self.df['BS_PRICE'] = np.vectorize(self.BlackScholesPrice)(CallPutFlag = self.df['option_type'],
                                                                S = self.df['S'].astype(float),
                                                                K = self.df['K'].astype(float),
                                                                T = self.df['_T'].astype(float),
                                                                v = self.df['I_VOL_BS'].astype(float)
                                                                )



    ############# MERTON #################

    def init_merton(self,m = None,lam = None,v = None, reset = False):
        """
        Initialise les paramètres du Modèle de Merton.
        =============================
        Input:  m     : Float, Jump Mean
                lam   : Float, Lambda
                v     : Float, Jump Standard Deviation
                reset : Bool, Pour choisir de reset vers les meilleurs paramètres calibrés
        =============================
        Output: None
        """
        if reset:
            ## after calibration (RMSE = 0.406%)
            self.m = 1.68542729380368
            self.v = 0.00036428123601998366
            self.lam = 4.682520794818356e-05
        else:
            self.m = m
            self.lam = lam
            self.v = v

    def MertonPrice(self, S,K,T,sigma,r=0.01, q = 0, CallPutFlag = 'C'):
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
            p += (np.exp(-self.m*self.lam*T) * (self.m*self.lam*T)**k / (k_fact)) * self.BlackScholesPrice(v = sigma_k, CallPutFlag=CallPutFlag, S=S, K=K, T=T, r=r_k)
        return p 

    def append_Merton_price(self):
        """
        Ajoute les prix de Merton au DataFrame principal
        ==================
        Input:  None
        ==================
        Output: None
        """
        if 'L_VOL_BS' not in self.df.columns:
            self.append_loc_vol_to_df() #??

        self.df['MERTON_PRICE'] = np.vectorize(self.MertonPrice)(CallPutFlag = self.df['option_type'],
                                                                S = self.df['S'].astype(float),
                                                                K = self.df['K'].astype(float),
                                                                T = self.df['_T'].astype(float),
                                                                sigma = self.df['I_VOL_MERTON'].astype(float) #changer pour Local VOL ?
                                                                )


    def target_f_merton(self, x):
        """
        Fonction à minimiser lors de la recherche des paramètres optimaux pour le modèle de Merton.
        ==================
        Input:  x  : liste [m, v, lam] des 3 paramètres à optimiser.
        ==================
        Output: Vecteur normalisé de la différence des prix sous le modèle de Merton calibré avec x et les prix du marché.
        """
        self.init_merton(m = x[0]  , lam = x[2], v = x[1])
        candidate_prices = np.vectorize(self.MertonPrice)(  CallPutFlag = self.df['option_type'],
                                                            S = self.df['S'].astype(float),
                                                            K = self.df['K'].astype(float),
                                                            T = self.df['_T'].astype(float),
                                                            sigma = self.df['I_VOL_MERTON'].astype(float) #changer pour Local
                                                            )

        return np.linalg.norm(self.df['mid'] - candidate_prices, 2)


    def optimize_merton(self, tol = 1e-10, max_iter = 102, update_when_done = True):
        """
        Fonction à appeler pour optimiser les paramètres des Merton
        ==================
        Input:  tol              : Float, Tolérence pour arrêter l'optimisation.
                max_iter         : Int, Nombre maximum d'itération pour l'optimizer.
                update_when_done : Bool, Update ou non le modèle de Merton avec les paramètres issus de l'optimisation.
        ==================
        Output: x = [m,v,lam]
        """
        #x0 = [1, 0.1, 1] # initial guess for algorithm
        x0 = [  random()*(2-0.01)+0.01,
                0.1,
                random()*5]
        #x0 = [0.7910348976571686, 0.3451336374548454, 0.0012410304674673033]
        bounds = ((0.01, 2), (1e-5, np.inf) , (0, 5)) #bounds as described above

        res = minimize(self.target_f_merton, 
                        method='SLSQP',
                        #method = 'Nelder-Mead',
                        x0=x0,
                        bounds = bounds, 
                        tol=tol, 
                        options={"maxiter":max_iter, "ftol":tol})

        mt = res.x[0]
        vt = res.x[1]
        lamt = res.x[2]
        if update_when_done:
            self.init_merton(m = mt , 
                            lam = lamt,
                            v = vt)
        print('Calibrated Jump Mean = ', mt)
        print('Calibrated Jump Std = ', vt)
        print('Calibrated intensity = ', lamt)
        return res.x
    
    ##### HESTON #################################
    def init_heston(self, theta = None, reset = False):
        if reset:
            self.theta = np.array([0.41055433, 2.87649559, 1.0035074 , 1.14520439, 2.15878211])
        else:
            self.theta = theta
        self.M = 100
        self.deg = 32

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



    def HestonPrice(self,S,K, T, r=0.0, CallPutFlag = 'C'):
        if CallPutFlag == 'C':
            return 1/S *( 
                        (S - np.exp(-r*T)*K)/2 + np.exp(-r*T)/np.pi*(   S*self.integrate_char_function(K, T, S, r, 1j) - \
                                                                        K*self.integrate_char_function(K, T, S, r, 0))
                        )
        elif CallPutFlag == 'P':
            return 0.0 # Not implemented yet.


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
        #raise NotImplementedError("This function is not working")
        self.init_heston(theta = x)
        r_vector = np.vectorize(self.HestonPrice)(self.df['K'], self.df['_T'], self.df['S']) - self.df['mid']
        #for i in range(1, len(mkt_data)):
        #    r_vector = np.append(r_vector, self.HestonPrice(mkt_data[i, 1], mkt_data[i, 2], S_0, r) - \
        #                        mkt_data[i, 0])
        return r_vector
        


    def target_f_heston(self, x):
        """
        Fonction à minimiser lors de la recherche des paramètres optimaux pour le modèle de Heston.
        ==================
        Input:  x  : liste [m, v, lam] des 3 paramètres à optimiser.
        ==================
        Output: Vecteur normalisé de la différence des prix sous le modèle de Merton calibré avec x et les prix du marché.
        """
        self.init_heston(theta = x)
        candidate_prices = np.vectorize(self.HestonPrice)(  CallPutFlag = self.df['option_type'],
                                                            S = self.df['S'].astype(float),
                                                            K = self.df['K'].astype(float),
                                                            T = self.df['_T'].astype(float),
                                                            )

        return np.linalg.norm(self.df['mid'] - candidate_prices, 2)

    def optimize_heston(self, x):
        #return  least_squares(self.target_f_heston, x0, method='lm').x
        if False:
            return  least_squares(self.r_function, x, method='lm').x

        else:
            tol = 1e-10
            max_iter = 102

            x0 = x

            #bounds = ((0.01, 2), (1e-5, np.inf) , (0, 5)) #bounds as described above
            bounds = None

            res = minimize(self.target_f_heston, 
                            method='SLSQP',
                            #method = 'Nelder-Mead',
                            x0=x0,
                            bounds = bounds, 
                            tol=tol, 
                            options={"maxiter":max_iter, "ftol":tol})

            print(res)
            return res.x

    def append_Heston_prices(self):
        self.df['HESTON_PRICE'] = np.vectorize(self.HestonPrice)(   CallPutFlag = self.df['option_type'],
                                                                    S = self.df['S'].astype(float),
                                                                    K = self.df['K'].astype(float),
                                                                    T = self.df['_T'].astype(float),
                                                                    #v = self.df['I_VOL_BS'].astype(float)
                                                                    )
        