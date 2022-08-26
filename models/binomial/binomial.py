
import numpy as np
import math
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf
import arch
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


def binomial_model(N, S0, u, r, K):
    """
    N = number of binomial iterations
    S0 = initial stock price
    u = factor change of upstate
    r = risk free interest rate per annum
    K = strike price
    """
    d = 1 / u
    p = (1 + r - d) / (u - d)
    q = 1 - p

    # make stock price tree
    stock = np.zeros([N + 1, N + 1])
    for i in range(N + 1):
        for j in range(i + 1):
            stock[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Generate option prices recursively
    option = np.zeros([N + 1, N + 1])
    option[:, N] = np.maximum(np.zeros(N + 1), (stock[:, N] - K))
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option[j, i] = (
                1 / (1 + r) * (p * option[j, i + 1] + q * option[j + 1, i + 1])
            )
    return option






class stock_vol:

	def __init__(self, tk, start, end):
		self.tk = tk
		self.start = start
		self.end = end
		all_data = pdr.get_data_yahoo(self.tk, start=self.start, end=self.end)
		self.stock_data = pd.DataFrame(all_data['Adj Close'], columns=["Adj Close"])
		self.stock_data["log"] = np.log(self.stock_data)-np.log(self.stock_data.shift(1))

	def mean_sigma(self):
		return self.stock_data["log"].dropna().ewm(span=252).std()[-1]


	def garch_sigma(self): 
		model = arch.arch_model(self.stock_data["log"].dropna(), mean='Zero', vol='GARCH', p=1, q=1)
		model_fit = model.fit()
		forecast = model_fit.forecast(horizon=1)
		var = forecast.variance.iloc[-1]
		sigma = float(np.sqrt(var))
		return sigma





class stockoption():

	def __init__(self, S0, K, r, T, N, prm):
		"""
		Initialise parameters
		:param S0: initial stock price
		:param K: strike price
		:param r: risk free interest rate per year
		:param T: length of option in years
		:param N: number of binomial iterations
		:param prm: dictionary with additional parameters
		"""
		self.S0 = S0
		self.K = K
		self.r = r
		self.T = T
		self.N = N
		"""
		prm parameters:
		start = date from when you want to analyse stocks, "yyyy-mm-dd"
		end = date of final stock analysis (likely current date), "yyyy-mm-dd"
		tk = ticker label
		div = dividend paid
		is_calc = is volatility calculated using stock price history, boolean
		use_garch = use GARCH model, boolean
		sigma = volatility of stock
		is_call = is it a call option, boolean
		eu_option = European or American option, boolean
		"""
		self.tk = prm.get('tk', None)
		self.start = prm.get('start', None)
		self.end = prm.get('end', None)
		self.div = prm.get('div', 0)
		self.is_calc = prm.get('is_calc', False)
		self.use_garch = prm.get('use_garch', False)
		self.vol = stock_vol(self.tk, self.start, self.end)
		if self.is_calc:
			if self.use_garch:
				self.sigma = self.vol.garch_sigma()
			else:
				self.sigma = self.vol.mean_sigma()
		else:
			self.sigma = prm.get('sigma', 0)
		self.is_call = prm.get('is_call', True)
		self.eu_option = prm.get('eu_option', True)
		'''
		derived values:
		dt = time per step, in years
		df = discount factor
		'''
		self.dt = T/float(N)
		self.df = math.exp(-(r-self.div)*self.dt)


class euro_option(stockoption):
	'''
	calculate required preliminary parameters:
	u = factor change of upstate
	d = factor change of downstate
	qu = risk free upstate probability
	qd = risk free downstate probability
	M = number of nodes
	'''
	def __int_prms__(self):
		self.M = self.N + 1 
		self.u = math.exp(self.sigma*math.sqrt(self.dt))
		self.d = 1./self.u
		self.qu = (math.exp((self.r-self.div)*self.dt)-self.d)/(self.u-self.d)
		self.qd = 1-self.qu
		
	def stocktree(self):
		stocktree = np.zeros([self.M, self.M])
		for i in range(self.M):
			for j in range(self.M):
				stocktree[j, i] = self.S0*(self.u**(i-j))*(self.d**j)
		return stocktree

	def option_price(self, stocktree):
		option = np.zeros([self.M, self.M])
		if self.is_call:
			option[:, self.M-1] = np.maximum(np.zeros(self.M), (stocktree[:, self.N] - self.K))
		else:
			option[:, self.M-1] = np.maximum(np.zeros(self.M), (self.K - stocktree[:, self.N]))
		return option

	def optpricetree(self, option):
		for i in np.arange(self.M-2, -1, -1):
			for j in range(0, i+1):
				option[j, i] = math.exp(-self.r*self.dt) * (self.qu*option[j, i+1]+self.qd*option[j+1, i+1])
		return option

	def begin_tree(self):
		stocktree = self.stocktree()
		payoff = self.option_price(stocktree)
		return self.optpricetree(payoff)

	def price(self):
		self.__int_prms__()
		self.stocktree()
		payoff = self.begin_tree()
		return payoff[0, 0]
