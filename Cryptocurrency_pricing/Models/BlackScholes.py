from Cryptocurrency_pricing.Models.common_all import Model
from scipy.stats import norm
import numpy as np

class BlackScholes(Model):
    
    def __init__(self, raw_dataframe, option_type = 'B'):
        super().__init__(raw_dataframe,option_type)
        self.model_name = 'black-scholes'


    def CND(self,X):
        """
        Retourne la fonction de répartition de la loi Normale centrée réduite évaluée en x
        """
        return float(norm.cdf(X))


    def Price(self,sigma,CallPutFlag = 'C',S = 100.,K = 100.,T = 1.,r = 0.01):
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
        



