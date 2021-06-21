from utils import *
from scipy import optimize
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np


class Option:
    def __init__(self, raw_dataframe, option_type = 'B'):
        self.option_type = option_type
        self.df = pipeline(raw_dataframe, option_type=self.option_type)
    
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

    
    def implied_v(self, P, S , K, T, r=0.0, right = 'C'):
        """
        Retourne une volatilité implicite
        ============================
        Input:
        ============================
        Output:
        """
        x,y,n = 0,1,20
        for _ in range(1,n+1):
            z = (x+y)/2

            if P > self.BlackScholesPrice(z/(1-z),CallPutFlag=right,S=S,T=T,K=K,r=r):
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
        self.df['I_VOL'] = np.vectorize(self.implied_v)(P = self.df['mid'].astype(float),
                                                    right = self.df['option_type'],
                                                    S = self.df['S'].astype(float),
                                                    K = self.df['K'].astype(float),
                                                    T = self.df['_T'].astype(float)
                                                    )
        
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

    def append_BS_price(self):
        self.df['BS_PRICE'] = np.vectorize(self.BlackScholesPrice)(CallPutFlag = self.df['option_type'],
                                                                S = self.df['S'].astype(float),
                                                                K = self.df['K'].astype(float),
                                                                T = self.df['_T'].astype(float),
                                                                v = self.df['I_VOL'].astype(float)
                                                                )



    ############# MERTON #################

    def init_merton(self,m,lam,v):
        #TODO calibrations
        self.m = m
        self.lam = lam
        self.v = v

    def MertonPrice(self, S,K,T,sigma,r=0.01, q = 0, CallPutFlag = 'C'):
        p = 0
        for k in range(40):
            r_k = r - self.lam*(self.m-1) + (k*np.log(self.m) ) / T
            sigma_k = np.sqrt( sigma**2 + (k* self.v** 2) / T)
            k_fact = np.math.factorial(k)
            p += (np.exp(-self.m*self.lam*T) * (self.m*self.lam*T)**k / (k_fact)) * self.BlackScholesPrice(v = sigma_k, CallPutFlag=CallPutFlag, S=S, K=K, T=T, r=r_k)
        return p 

    def append_Merton_price(self):
        if 'L_VOL' not in self.df.columns:
            self.append_loc_vol_to_df() #??

        self.df['MERTON_PRICE'] = np.vectorize(self.MertonPrice)(CallPutFlag = self.df['option_type'],
                                                                S = self.df['S'].astype(float),
                                                                K = self.df['K'].astype(float),
                                                                T = self.df['_T'].astype(float),
                                                                sigma = self.df['L_VOL'].astype(float) #changer pour Local VOL ?
                                                                )


    def optimal_params(self, x):
        self.init_merton(m = x[0]  , lam = x[2], v = x[1])
        candidate_prices = np.vectorize(self.MertonPrice)(  CallPutFlag = self.df['option_type'],
                                                            S = self.df['S'].astype(float),
                                                            K = self.df['K'].astype(float),
                                                            T = self.df['_T'].astype(float),
                                                            sigma = self.df['L_VOL'].astype(float) #changer pour Local
                                                            )

        return np.linalg.norm(self.df['mid'] - candidate_prices, 2)


    def optimize_merton(self):
        x0 = [1, 0.1, 1] # initial guess for algorithm
        bounds = ((0.01, 2), (1e-5, np.inf) , (0, 5)) #bounds as described above

        res = minimize(self.optimal_params, 
                        method='SLSQP',  
                        x0=x0,
                        bounds = bounds, 
                        tol=1e-20, 
                        options={"maxiter":1000})

        mt = res.x[0]
        vt = res.x[1]
        lamt = res.x[2]

        self.init_merton(m = mt  , lam = lamt, v = vt)
        print('Calibrated Jump Mean = ', mt)
        print('Calibrated Jump Std = ', vt)
        print('Calibrated intensity = ', lamt)