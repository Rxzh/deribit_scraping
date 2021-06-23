from utils import *
from scipy import optimize
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
from random import random

class Option:
    def __init__(self, raw_dataframe, option_type = 'B'):
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


    def optimal_params(self, x):
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

        return np.linalg.norm(self.df['mark_price'] - candidate_prices, 2)


    def optimize_merton(self, tol = 1e-10, max_iter = 102, update_when_done = True):
        """
        Fonction à appeler pour optimiser les paramètres des Merton
        ==================
        Input:  tol              : Float, Tolérence pour arrêter l'optimisation.
                max_iter         : Int, Nombre maximum d'itération pour l'optimizer.
                update_when_done : Bool, Update ou non le modèle de Merton avec les paramètres issus de l'optimisation.
        ==================
        Output: None
        """
        #x0 = [1, 0.1, 1] # initial guess for algorithm
        x0 = [  random()*(2-0.01)+0.01,
                0.1,
                random()*5]
        #x0 = [0.7910348976571686, 0.3451336374548454, 0.0012410304674673033]
        bounds = ((0.01, 2), (1e-5, np.inf) , (0, 5)) #bounds as described above

        res = minimize(self.optimal_params, 
                        
                        method='SLSQP',
                        #method = 'Nelder-Mead',
                        x0=x0,
                        bounds = bounds, 
                        tol=tol, 
                        options={"maxiter":max_iter})

        mt = res.x[0]
        vt = res.x[1]
        lamt = res.x[2]
        if update_when_done:
            self.init_merton(m = mt  , lam = lamt, v = vt)
        print('Calibrated Jump Mean = ', mt)
        print('Calibrated Jump Std = ', vt)
        print('Calibrated intensity = ', lamt)