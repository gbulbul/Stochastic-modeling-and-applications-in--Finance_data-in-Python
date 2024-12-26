# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 17:09:30 2024

@author: gbulb
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import numpy.random as npr
"""
Geometric Brownian Motion can be seen as a stochastic method which is suitable to simulate BSM model in finance.

"""
class Geometric_Brownian_Motion:
        def __init__(self,S0,r,sigma,T,I):
            self.S0=S0
            self.r=r
            self.sigma=sigma
            self.T=T
            self.I=I
        def ST1_function(self):
            S0,r,sigma,T,I=self.S0,self.r,self.sigma,self.T,self.I
            ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T +
            sigma * math.sqrt(T) * npr.standard_normal(I))
            return ST1
        def ST2_function(self):
            S0,r,sigma,T,I=self.S0,self.r,self.sigma,self.T,self.I
            ST2 = S0 * npr.lognormal((r - 0.5 * sigma ** 2) * T,
            sigma * math.sqrt(T), size=I)
            return ST2
        def plotting(f,ST1):
            plt.hist(ST1, bins=50)
            plt.xlabel('index level')
            plt.title('Statically Simulated geometric Brownian motion (via npr.standard_normal())')
            plt.ylabel('frequency')
            plt.show()
        def plotting(f,ST2): 
            plt.hist(ST2, bins=50)
            plt.xlabel('index level')
            plt.title('Statically Simulated geometric Brownian motion (via npr.lognormal())')
            plt.ylabel('frequency')
            plt.show()

"""
Stochastic Differential Equations can be seen as a stochastic method which is suitable to simulate implied volatility of ECO in BSM model.

"""
class Stochastic_Diff_Eqns:
        def __init__(self,I,M,r,sigma,T):
            self.I=I
            self.M=M
            self.r=r
            self.sigma=sigma
            self.T=T
        def SDE_(self):
            I,M,r,sigma,T=self.I,self.M,self.r,self.sigma,self.T
            S = np.zeros((M + 1, I))
            dt = T / M
            S[0]=S0
            A = (r - 0.5 * sigma ** 2) * dt
            B = sigma * math.sqrt(dt)
           
            for t in range(1, M + 1):
                S[t] = S[t - 1] * np.exp(A + B * npr.standard_normal(I))
            return S
        def plotting_at_maturity(f,SDE):
            plt.hist(SDE[-1], bins=50)
            plt.xlabel('index level')
            plt.title('Dynamically simulated geometric Brownian motion at maturity')
            plt.ylabel('frequency')
            plt.show()
        def plotting_paths(f,SDE):
            plt.plot(SDE[:, :10], lw=1.5)
            plt.xlabel('time')
            plt.title('Dynamically simulated geometric Brownian motion paths')
            plt.ylabel('index level');
            plt.show()

"""
Square root diffusion-Euler 

"""
class Square_root_diffusion_euler_Eqns:

      def __init__(self,x0,M,kappa,sigma,theta,I,T):
          self.x0=x0
          self.M=M
          self.kappa=kappa
          self.sigma=sigma
          self.theta=theta
          self.I=I
          self.T=T
          

      def srd_euler(self):
          x0,M,kappa,sigma,theta,I,T=self.x0,self.M,self.kappa,self.sigma,self.theta,self.I,self.T
          dt = T / M
          xh = np.zeros((M + 1, I))
          x = np.zeros_like(xh)
          xh[0] = x0
          x[0] = x0
          for t in range(1, M + 1):
              xh[t] = (xh[t - 1] +\
              kappa * (theta - np.maximum(xh[t - 1], 0)) * dt +\
              sigma * np.sqrt(np.maximum(xh[t - 1], 0)) *\
              math.sqrt(dt) * npr.standard_normal(I))
              x = np.maximum(xh, 0)
              return x
      def plotting_square_root_diffusion_at_maturity(f,x1):
           plt.hist(x1[-1], bins=50)
           plt.xlabel('value')
           plt.title('Dynamically simulated square-root diffusion at maturity (Euler scheme)')
           plt.ylabel('frequency');
      def plotting_square_root_diffusion_paths(f,x1):   
           plt.plot(x1[:, :10], lw=1.5)
           plt.xlabel('time')
           plt.title('Dynamically simulated square-root diffusion paths (Euler scheme)')
           plt.ylabel('index level');

       ###DISCRETIZATION###
      def srd_exact(self):
            x0,M,kappa,sigma,theta,I,T=self.x0,self.M,self.kappa,self.sigma,self.theta,self.I,self.T
            dt = T / M
            x = np.zeros((M + 1, I))
            x[0] = x0
            for t in range(1, M + 1):
                df = 4 * theta * kappa / sigma ** 2
                c = (sigma ** 2 * (1 - np.exp(-kappa * dt))) / (4 * kappa)
                nc = np.exp(-kappa * dt) / c * x[t - 1]
                x[t] = c * npr.noncentral_chisquare(df, nc, size=I)
            return x
      def plotting_square_root_diffusion_at_maturity_exact(f,x2): 

          plt.hist(x2[-1], bins=50)
          plt.xlabel('value')
          plt.title('Dynamically simulated square-root diffusion at maturity (exact scheme)')
          plt.ylabel('frequency')
          plt.show()

      def plotting_square_root_diffusion_paths_exact(f,x2): 
          plt.plot(x2[:, :10], lw=1.5)
          plt.title('Dynamically simulated square-root diffusion paths (exact scheme)')
          plt.xlabel('time')
          plt.ylabel('index level')
          plt.show()

'''
Stochastic volatility
One of the major simplifying assumptions of the Black-Scholes-Merton model is the constant volatility. However, volatility in general is neither constant nor deterministic — it is stochastic (i.e random).
'''
class Stochastic_volatility:
     
    
    def cho_mat(rho):
       corr_mat = np.zeros((2, 2))
       corr_mat[0, :] = [1.0, rho]
       corr_mat[1, :] = [rho, 1.0]
       cho_mat = np.linalg.cholesky(corr_mat)
       return cho_mat

    def changing_volatility_v(T,M,v0,kappa,sigma,theta,cho_mat,ran_num):

        dt = T / M
        v = np.zeros_like(ran_num[0])
        vh = np.zeros_like(v)
        v[0] = v0
        vh[0] = v0
        for t in range(1, M + 1):
            ran = np.dot(cho_mat, ran_num[:, t, :])
            vh[t] = (vh[t - 1] +
                     kappa * (theta - np.maximum(vh[t - 1], 0)) * dt +
                     sigma * np.sqrt(np.maximum(vh[t - 1], 0)) *
                     math.sqrt(dt) * ran[1])
        v = np.maximum(vh, 0)
        return v

    def changing_volatility_S(T,M,v,kappa,sigma,cho_mat,ran_num,S0):
        dt = T / M
        S = np.zeros_like(ran_num[0])
        S[0] = S0
        for t in range(1, M + 1):
            ran = np.dot(cho_mat, ran_num[:, t, :])
            S[t] = S[t - 1] * np.exp((r - 0.5 * v[t]) * dt +
                                     np.sqrt(v[t]) * ran[0] * np.sqrt(dt))
        return S
    def plotting_stochastic_volatility_at_maturity(S,v):
         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
         ax1.hist(S[-1], bins=50)
         ax1.set_xlabel('index level')
         ax1.set_ylabel('frequency')
         ax2.hist(v[-1], bins=50)
         ax2.set_xlabel('volatility')
         plt.suptitle('Dynamically simulated stochastic volatility process at maturity');
         plt.show()
    def plotting_paths(S,v):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
        figsize=(10, 6))
        ax1.plot(S[:, :10], lw=1.5)
        ax1.set_ylabel('index level')
        ax2.plot(v[:, :10], lw=1.5)
        ax2.set_xlabel('time')
        ax2.set_ylabel('volatility')
        ax2.axhline(theta)
        plt.suptitle('Dynamically simulated stochastic volatility process paths');
        plt.show()

"""
Stochastic volatility and the leverage effect are stylized (empirical) facts found in a number of markets. Another important stylized fact is the existence of jumps in asset prices and, for example, volatility. In 1976, Merton published his jump diffusion model, enhancing the Black-Scholes-Merton setup through a model component gen‐ erating jumps with log-normal distribution.
"""
class Jump_diffusion:
      def jump_diff_function(delta,mu,I,M,sigma,lamb,T):
          

        rj = lamb * (math.exp(mu + 0.5 * delta ** 2) - 1)
        dt = T / M
        S_jump = np.zeros((M + 1, I))
        S_jump[0] = S0
        sn1 = npr.standard_normal((M + 1, I))
        sn2 = npr.standard_normal((M + 1, I))
        poi = npr.poisson(lamb * dt, (M + 1, I))
        for t in range(1, M + 1, 1):
            S_jump[t] = S_jump[t - 1] * (np.exp((r - rj - 0.5 * sigma ** 2) * dt +\
            sigma * math.sqrt(dt) * sn1[t]) +\
            (np.exp(mu + delta * sn2[t]) - 1) *\
            poi[t])
            S_jump[t] = np.maximum(S_jump[t], 0)
        return S_jump
      def plotting_at_maturity(f,S_jump):
    
        plt.hist(S_jump[-1], bins=50)
        plt.xlabel('value')
        plt.ylabel('frequency')
        plt.title('Dynamically simulated jump diffusion process at maturity');
        plt.show()

if __name__=="__main__":
    S0 = 100
    r = 0.05
    sigma = 0.25
    T = 2.0
    I = 10000
    M=50
    ########
    x0 = 0.05
    kappa = 3.0
    theta=0.02
    ########
    v0 = 0.1
    rho = 0.6
    lamb = 0.75
    mu = -0.6
    delta = 0.25
    ran_num = npr.standard_normal((2, M + 1, I))
    #S = np.zeros((M + 1, I))
    f=plt.figure(figsize=(10, 6))
    geo_brownian_mot=Geometric_Brownian_Motion(S0, r, sigma, T, I)
    ST1=geo_brownian_mot.ST1_function()
    ST2=geo_brownian_mot.ST2_function()
    Geometric_Brownian_Motion.plotting(f,ST1)
    Geometric_Brownian_Motion.plotting(f,ST2)
    stoch_diff_eqns=Stochastic_Diff_Eqns(I, M, r, sigma, T)
    SDE=stoch_diff_eqns.SDE_()
    Stochastic_Diff_Eqns.plotting_at_maturity(f, SDE)
    Stochastic_Diff_Eqns.plotting_paths(f, SDE)
    square_root_diff_euler=Square_root_diffusion_euler_Eqns(x0, M, kappa, sigma, theta, I, T)
    x1=square_root_diff_euler.srd_euler()
    Square_root_diffusion_euler_Eqns.plotting_square_root_diffusion_at_maturity(f, x1)
    Square_root_diffusion_euler_Eqns.plotting_square_root_diffusion_paths(f, x1)
    x2=square_root_diff_euler.srd_exact()
    Square_root_diffusion_euler_Eqns.plotting_square_root_diffusion_at_maturity_exact(f, x2)
    Square_root_diffusion_euler_Eqns.plotting_square_root_diffusion_paths_exact(f, x2)
    cho_mat=Stochastic_volatility.cho_mat(rho)
    v=Stochastic_volatility.changing_volatility_v(T,M,v0,kappa,sigma,theta,cho_mat,ran_num)
    S=Stochastic_volatility.changing_volatility_S(T,M,v,kappa,sigma,cho_mat,ran_num,S0)
    Stochastic_volatility.plotting_paths(S, v)
    Stochastic_volatility.plotting_stochastic_volatility_at_maturity(S, v)
    S_jump=Jump_diffusion.jump_diff_function(delta, mu, I, M, sigma, lamb, T)
    Jump_diffusion.plotting_at_maturity(f, S_jump)
    
    
    
   
       