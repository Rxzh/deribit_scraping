from Cryptocurrency_pricing.Models.utils import *
#from scipy import optimize
from scipy.optimize import minimize, least_squares
import numpy as np
from random import random
from scipy.stats import norm


class Model:
    def __init__(self, raw_dataframe, option_type = 'B'):
        """
        Initialise la classe Model, anciennement Option,
    
        ======================
        Input:  raw_dataframe : DataFrame brut scrapé depuis deribit via la classe Scaper
                option_type   : {'B':,'C','P'} / caractérise le type d'option qu'on veut garder depuis le DataFrame brut. 
        =======================
        Output: None
        """
        self.model_name = None
        
        self.option_type = option_type

        self.df = pipeline(raw_dataframe, option_type=self.option_type)

        self.initialize(reset = True) # initialize les params


        """
        self.m = 1.68542729380368
        self.v = 0.00036428123601998366
        self.lam = 4.682520794818356e-05
        """

    def initialize(self, theta = None, reset = False):
        pass

    def CND(self,X):
        """
        Retourne la fonction de répartition de la loi Normale centrée réduite évaluée en x
        """
        return float(norm.cdf(X))

    def BlackScholesPrice(self,sigma,CallPutFlag = 'C',S = 100.,K = 100.,T = 1.,r = 0.01):
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
        if CallPutFlag != 'C' and CallPutFlag != 'P':
            raise ValueError("Mauvais type d'option : {'P','C'}")
            
        try:
            d1 = (np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))
            d2 = d1-sigma*np.sqrt(T)

            if CallPutFlag == 'C':
                return S*self.CND(d1)-K*np.exp(-r*T)*self.CND(d2)
            elif CallPutFlag == 'P':
                return K*np.exp(-r*T)*self.CND(-d2)-S*self.CND(-d1)
    
        except: 
            return 0.0

    def implied_v(self, P, S , K, T, r=0.0, right = 'C'):
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
            if P > self.BlackScholesPrice(S=S,K=K,T=T,sigma=z/(1-z),r=r, CallPutFlag = right):
                x = z
            else:
                y = z
        z = (x+y)/2
        sigma = z/(1-z)
        
        return sigma


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
            if self.model_name == 'black-scholes':
                d1 = (np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))
                d2 = d1-sigma*np.sqrt(T)
                if right == 'C':
                    return -S/K*1/np.sqrt(2*np.pi)*np.exp(-.5*d1**2) + np.exp(r*T)*(-self.CND(d2)+1/np.sqrt(2*np.pi)*np.exp(-.5*d2**2))
                else:
                    return +S/K*1/np.sqrt(2*np.pi)*np.exp(-.5*d1**2) + np.exp(r*T)*(self.CND(-d2)+1/np.sqrt(2*np.pi)*np.exp(-.5*d2**2))
            else:
                h = 0.01
                return (self.Price(sigma,right,S,K+h,T,r)-self.Price(sigma,right,S,K,T,r))/h
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
            if self.model_name == 'black-scholes':
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
            else:
                h = 0.001
                return (self.dP_dK(self,S,K+h,T,sigma,right,r)-self.dP_dK(self,S,K,T,sigma,right,r))/h
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
            if self.model_name == 'black-scholes':
                d1 = (np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))
                d2 = d1-sigma*np.sqrt(T)
                if right == 'C':
                    return  (S*sigma/2 * 1/np.sqrt(2*np.pi*T) *np.exp(-.5*d1**2)) + r*K*np.exp(-r*T)*self.CND(d2)

                else:
                    return  (S*sigma/2 * 1/np.sqrt(2*np.pi*T) *np.exp(-.5*d1**2)) - r*K*np.exp(-r*T)*self.CND(-d2)
            else:
                h = 0.01
                return (self.Price(sigma,right,S,K,T+h,r)-self.Price(sigma,right,S,K,T,r))/h
        except:
            return 0.0
    
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
    """
    def implied_v(self, P, S, K, T, r = 0.01, right = 'C', inc = 0.001):
        f = lambda x: self.BlackScholesPrice(x,CallPutFlag=right,S=S,K=K,T=T,r=r)-P
        return optimize.brentq(f,0.,120.,maxiter=1000)
    """

    def append_imp_vol_to_df(self, r=0.01):
            """
            Rajoute la colonne 'I_VOL' des volatilités implicite au dataframe
            ==================
            Input:  None
            ==================
            Output: None
            """
            
            self.df['I_VOL'] = np.vectorize(self.implied_v)(P = self.df['mark_price'].astype(float),
                                                right = self.df['option_type'],
                                                S = self.df['S'].astype(float),
                                                K = self.df['K'].astype(float),
                                                T = self.df['_T'].astype(float)
                                                )
            
            self.df['IV_moneyness'] = self.df['S']/self.df['I_VOL']


    def append_loc_vol_to_df(self, r=0.01):
        """
        Rajoute la colonne 'L_VOL' des volatilités implicite au dataframe
        Si la colonne 'I_V' n'a pas déjà été créée, elle aussi est créée.
        ==================
        Input:  None
        ==================
        Output: None
        """
        
        if 'I_VOL' not in self.df.columns:
            self.append_imp_vol_to_df(r)
            
        self.df['L_VOL'] = np.vectorize(self.local_v)(  right = self.df['option_type'],
                                                        S = self.df['S'].astype(float),
                                                        K = self.df['K'].astype(float),
                                                        T = self.df['_T'].astype(float),
                                                        sigma = self.df['I_VOL'].astype(float)
                                                        )


    def append_price(self): #TODO ici append_price
        """
        Ajoute les prix de Black-Scholes au DataFrame principal
        ==================
        Input:  None
        ==================
        Output: None
        """
        if self.model_name == 'black-scholes':
            col_name = 'BS_PRICE'
        elif self.model_name == 'merton':
            col_name = 'MERTON_PRICE'
        elif self.model_name == 'heston':
            col_name = 'HESTON_PRICE'
        else:
            raise Exception('A correct model must be chosen')
        

        self.df[col_name] = np.vectorize(self.Price)(CallPutFlag = self.df['option_type'],
                                                                S = self.df['S'].astype(float),
                                                                K = self.df['K'].astype(float),
                                                                T = self.df['_T'].astype(float),
                                                                sigma = self.df['I_VOL'].astype(float)
                                                                )



    
    ### OPTIMISATION DE PARAMETRES ###

    def target_func(self, x):
        """
        Fonction à minimiser lors de la recherche des paramètres optimaux pour le modèle de Merton.
        ==================
        Input:  x  : liste [m, v, lam] des 3 paramètres à optimiser.
        ==================
        Output: Vecteur normalisé de la différence des prix sous le modèle de Merton calibré avec x et les prix du marché.
        """
        self.initialize(theta = x)
        candidate_prices = np.vectorize(self.Price)(CallPutFlag = self.df['option_type'],
                                                    S = self.df['S'].astype(float),
                                                    K = self.df['K'].astype(float),
                                                    T = self.df['_T'].astype(float),
                                                    sigma = self.df['I_VOL'].astype(float) #changer pour Local
                                                    )

        return np.linalg.norm(self.df['mark_price'] - candidate_prices, 2)



    def optimize(self,x0 = None, tol = 1e-10, max_iter = 102, update_when_done = False, bounds = None, verbose = False):
        """
        Fonction à appeler pour optimiser les paramètres des Merton
        ==================
        Input:  tol              : Float, Tolérence pour arrêter l'optimisation.
                max_iter         : Int, Nombre maximum d'itération pour l'optimizer.
                update_when_done : Bool, Update ou non le modèle de Merton avec les paramètres issus de l'optimisation.
        ==================
        Output: x = [m,v,lam]
        """
        if self.model_name == 'merton' and bounds is None: #TODO retirer ça ?
            bounds = ((0.01, 2), (1e-5, np.inf) , (0, 5)) #bounds for Merton

        if x0 == None:
            if bounds is not None:
                #crée un vecteur aléatoire respectant les bornes
                x0 = [ 
                        bound[0]+random()*(bound[1]-bound[0]) 
                        if np.inf not in bound 
                        else random() 
                        for bound in bounds
                    ] 
            else:
                x0 = np.random.random(size = np.size(self.theta))

        if verbose:
            print("Starting parameters optimisation for {} model".format(self.model_name))
            print("x0 = {}".format(x0))
            print("bounds = {}".format(bounds))
            print("Method = {}".format("SLSQP")) #TODO

        res = minimize(self.target_func, 
                        method='SLSQP',
                        #method = 'Nelder-Mead',
                        x0=x0,
                        bounds = bounds, 
                        tol=tol, 
                        options={"maxiter":max_iter, "ftol":tol})


        if update_when_done:
            self.initialize(theta = res.x)
        if verbose:
            print('Calibrated Theta = {}'.format(res.x))

        return res.x