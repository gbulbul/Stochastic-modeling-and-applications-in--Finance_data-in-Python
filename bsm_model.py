# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 13:47:32 2024

@author: gbulb
"""

from math import log, sqrt, exp
from scipy import stats
from random import gauss, seed
class BSM_model:
    def __init__(self, S0, K,T,r,sigma):
        # Instance variable
        self.S0 = S0
        self.K= K
        self.r= r
        self.T= T
        self.sigma= sigma
        
    def bsm_call_value(self):
        ''' Valuation of European call option in BSM model.
        Analytical formula.
        Parameters
        ==========
        S0 : float
        initial stock/index level
        K : float
        strike price
        T : float
        maturity date (in year fractions)
        r : float
        constant risk-free short rate
        sigma : float
        volatility factor in diffusion term
        Returns
        =======
        value : float
        present value of the European call option
        '''
        S0 = float(self.S0)
        d1 = (log(S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T))
        d2 = (log(S0 / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T))
        value = (S0 * stats.norm.cdf(d1, 0.0, 1.0)
        - self.K * exp(-self.r * self.T) * stats.norm.cdf(d2, 0.0, 1.0))
        return value
    def bsm_vega(self):
        ''' Vega of European option in BSM model.
        Returns
        =======
        vega : float
        partial derivative of BSM formula with respect
        to sigma, i.e. Vega
        '''

        S0 = float(self.S0)
        d1 = (log(S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T )/ (self.sigma * sqrt(self.T))
        vega = S0 * stats.norm.cdf(d1, 0.0, 1.0) * sqrt(self.T)
        return vega
    def bsm_call_imp_vol(self, dt, it=100, M = 50,  I = 250000):
        ''' Implied volatility of European call option in BSM model.
        Returns
        =======
        simga_est : float
        numerically estimated implied volatility
        '''
        S = []
        for i in range(I):
            h = []
            for t in range(M + 1):
                if t == 0:
                   h.append(S0)
                else:
                   z = gauss(0.0, 1.0)
                   St = h[t - 1] * exp((r - 0.5 * sigma ** 2) * dt
                   + sigma * sqrt(dt) * z)
                   h.append(St)
            S.append(h)
        ##Monte Carlo estimator##
        C0=exp(-r * T) * sum([max(h[-1] - K, 0) for h in S]) / I
        sigma_est=0
        for i in range(it):
            sigma_est -= (BSM_model.bsm_call_value(self) - C0) / BSM_model.bsm_vega(self)
        return sigma_est
if __name__=="__main__":
    S0 = 100. # initial value
    K = 105. # strike price
    T = 1.0 # maturity
    r = 0.05 # riskless short rate
    it=100 #number of iterations
    M = 50 # number of time steps
    dt = T / M # length of time interval
    I = 250000 # number of paths=h
    sigma = 0.2 # volatility
    BSM_model_val=BSM_model(S0, K,T,r,sigma)   
    print('Valuation of European call option in BSM model:',BSM_model_val.bsm_call_value())
    print('Vega of European option in BSM model:',BSM_model_val.bsm_vega())
    print('Implied volatility of European call option in BSM model:',BSM_model_val.bsm_call_imp_vol(T,M))
 
    
    
    
    
    
       
