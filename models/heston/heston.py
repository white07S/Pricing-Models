import numpy as np
import pandas as pd

# For calculation and solvinge quations/optimazation
import scipy.stats as ss
import scipy.optimize as scpo
from scipy import sparse
from scipy.fftpack import ifft
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from functools import partial

# Result Presenting
import matplotlib.pyplot as plt

import warnings

class Diffusion_process():    
    def __init__(self, r=0.1, sig=0.2, mu=0.1):
        """    
        r: risk free interest rate(constant)
        sig: constant diffusion coefficient(positive)
        mu: constant drift coefficient
        """
        self.r, self.mu, self.sig = r, mu, sig

# Class of the Heston process that stores the parameters
class Heston_process():
    def __init__(self, mu=0.1, rho=0, sigma=0.2, theta=-0.1, kappa=0.1):
        """
        r: risk free constant rate
        rho: correlation between stock noise and variance noise (|rho| must be <=1)
        theta: long term mean of the variance process(positive)
        sigma: volatility coefficient(positive)
        kappa: mean reversion coefficient for the variance process(positive)
        """
        self.mu, self.rho, self.theta, self.sigma, self.kappa = mu, rho, theta, sigma, kappa   

def fft(K, S0, r, T, cf): # interp support cubic 
    """ 
    K = vector of strike
    S0 = spot price scalar
    cf = characteristic function
    """
    N=2**15                         # FFT more efficient for N power of 2
    B = 500                         # integration limit 
    
    dx = B/N
    x = np.arange(N) * dx

    weight = 3 + (-1)**(np.arange(N)+1) # Simpson weights
    weight[0] = 1; weight[N-1]=1

    dk = 2*np.pi/B
    b = N * dk /2
    ks = -b + dk * np.arange(N)

    integrand = np.exp(- 1j * b * np.arange(N)*dx) * cf(x - 0.5j) * 1/(x**2 + 0.25) * weight * dx/3
    integral_value = np.real( ifft(integrand)*N )
    spline_cub = interp1d(ks, integral_value, kind="cubic") # cubic will fit better than linear
    prices = S0 - np.sqrt(S0 * K) * np.exp(-r*T)/np.pi * spline_cub( np.log(S0/K) )
    
    return prices


class Option_param():  
    def __init__(self, S0=100, K=100, T=1, v0=0.04, payoff="call", exercise="European"):
        """
        S0: current stock price
        K: Strike price
        T: time to maturity
        v0: (optional) spot variance 
        exercise: European or American
        """
        self.S0, self.v0, self.K, self.T, self.exercise, self.payoff = S0, v0, K, T, exercise, payoff

class BS_pricer():
    """
    Finite-difference Black-Scholes PDE = df/dt + r df/dx + 1/2 sigma^2 d^f/dx^2 -rf = 0
    """
    def __init__(self, Option_info, Process_info ):
        """
        Process_info: a instance of "Diffusion_process.", which contains (r,mu, sig) 
        Option_info: of type Option_param, which contains (S0,K,T)
        """ 
        self.r = Process_info.r           # interest rate
        self.sig = Process_info.sig       # diffusion coefficient
        self.S0 = Option_info.S0          # current price
        self.K = Option_info.K            # strike
        self.T = Option_info.T            # maturity in years
        
        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
        self.exercise, self.payoff = Option_info.exercise, Option_info.payoff
        
    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum( S - self.K, 0 )
        elif self.payoff == "put":    
            Payoff = np.maximum( self.K - S, 0 )  
        return Payoff
        
    @staticmethod # Black Scholes closed formula
    def BlackScholes(payoff='call', S0=100., K=100., T=1., r=0.1, sigma=0.2 ):
        d1 = (np.log(S0/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S0/K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        if payoff=="call":
            return S0 * ss.norm.cdf( d1 ) - K * np.exp(-r * T) * ss.norm.cdf( d2 )
        elif payoff=="put":
            return K * np.exp(-r * T) * ss.norm.cdf( -d2 ) - S0 * ss.norm.cdf( -d1 )
    
    # FFT method that yields a vector of prices.
    def FFT(self, K): # K: an array of strikes
        
        # Characteristic function of a Normal random variable
        def cf_normal(u, mu=1, sig=2):
            return np.exp( 1j * u * mu - 0.5 * u**2 * sig**2 )
        
        K = np.array(K)
        cf_GBM = partial(cf_normal, mu=( self.r - 0.5 * self.sig**2 )*self.T, sig=self.sig*np.sqrt(self.T))  # function binding
        if self.payoff == "call":
            return fft(K, self.S0, self.r, self.T, cf_GBM)
        elif self.payoff == "put":    # put-call parity
            return fft(K, self.S0, self.r, self.T, cf_GBM) - self.S0 + K*np.exp(-self.r*self.T)

# Class to price the options with the Heston model by:
class Heston_pricer():
    
    def __init__(self, Option_info, Process_info ):
        """
        Process_info: a instance of "Heston_process.", which contains (mu, rho, sigma, theta, kappa)
        Option_info: of type Option_param, which contains (S0,K,T)
        """
        self.r = Process_info.mu              # interest rate
        self.sigma = Process_info.sigma       # Heston parameters
        self.theta = Process_info.theta       
        self.kappa = Process_info.kappa       
        self.rho = Process_info.rho           
        
        self.S0 = Option_info.S0          # current price
        self.v0 = Option_info.v0          # spot variance
        self.K = Option_info.K            # strike
        self.T = Option_info.T            # maturity(in years)
        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff
    
    # payoff function
    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum( S - self.K, 0 )
        elif self.payoff == "put":    
            Payoff = np.maximum( self.K - S, 0 )  
        return Payoff
    
    # FFT method. It returns a vector of prices.
    def FFT(self, K): # K: strikes
        K = np.array(K)
        
        # Heston characteristic function (proposed by Schoutens 2004)
        def cf_Heston_good(u, t, v0, mu, kappa, theta, sigma, rho):
            xi = kappa - sigma*rho*u*1j
            d = np.sqrt( xi**2 + sigma**2 * (u**2 + 1j*u) )
            g1 = (xi+d)/(xi-d)
            g2 = 1/g1
            cf = np.exp( 1j*u*mu*t + (kappa*theta)/(sigma**2) * ( (xi-d)*t - 2*np.log( (1-g2*np.exp(-d*t))/(1-g2) ))\
                      + (v0/sigma**2)*(xi-d) * (1-np.exp(-d*t))/(1-g2*np.exp(-d*t)) )
            return cf
        
        cf_H_b_good = partial(cf_Heston_good, t=self.T, v0=self.v0, mu=self.r, theta=self.theta, 
                                  sigma=self.sigma, kappa=self.kappa, rho=self.rho)
        if self.payoff == "call":
            return fft(K, self.S0, self.r, self.T, cf_H_b_good)
        elif self.payoff == "put":        # put-call parity
            return fft(K, self.S0, self.r, self.T, cf_H_b_good) - self.S0 + K*np.exp(-self.r*self.T)




