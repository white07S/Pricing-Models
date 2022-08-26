from heston import * 
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("spy-options-exp-2020-08-14-weekly-show-all-stacked-08-07-2020.csv")
data = data.loc[:,['Strike', 'Bid', 'Midpoint', 'Ask',  'Type']]
data["Spread"] = (data.Ask - data.Bid)

CALL = data[data.Type=="Call"]
# PUT = data[data.Type=="Put"].reset_index(drop=True)
prices = CALL.Midpoint.values
strikes = CALL.Strike.values
spreads = CALL.Spread.values
payoff = "call"

def report_calibration(initial_guess, calibrated_params):
    report = pd.DataFrame({"Initial guess": initial_guess, "Calibrated": calibrated_params},
                          index=["rho", "sigma", "theta", "kappa", "v0"]).round(4).T
    return report

S0=334.33; K=S0; T=5/252; r=0.05
    # option price at Augest 7
    
# Objective function
def f_Hest(x, rho, sigma, theta, kappa, v0, r=0.05):
    Heston_param = Heston_process(mu=r, rho=rho, sigma=sigma, theta=theta, kappa=kappa)
    opt_param = Option_param(S0=S0, K=K, T=T, v0=v0, exercise="European", payoff=payoff )
    Hest = Heston_pricer(opt_param, Heston_param)
    return Hest.FFT(x)

init_vals = [-0.6, 1.0, 0.04, 2.5, 0.04] # rho, sigma, theta, kappa, v0
bounds = ( [-1, 1e-15, 1e-15, 1e-15, 1e-15], [1, np.inf, 2, np.inf, 2] )
params_Hest = scpo.curve_fit(f_Hest, strikes, prices, 
                             p0=init_vals, bounds=bounds, sigma=spreads,
                             xtol=1e-4, max_nfev=1000)[0]

# Result
report_calibration(init_vals, params_Hest)

def Feller(x):
    return 2*x[3] * x[2] - x[1]**2 - 1e-6
cons = ({"fun": Feller, "type": "ineq"})

def least_sq(x, prices, strikes, spread):
    """ Objective function """
    Heston_param = Heston_process(mu=r, rho=x[0], sigma=x[1], theta=x[2], kappa=x[3])
    opt_param = Option_param(S0=S0, K=K, T=T, v0=x[4], exercise="European", payoff="call" )
    Hest = Heston_pricer(opt_param, Heston_param)
    prices_calib = Hest.FFT(strikes)
    return np.sum( ((prices_calib - prices)/spread)**2 ) 

init_vals = [-0.4, 1.1, 0.1, 0.6, 0.02] # rho, sigma, theta, kappa, v0
bounds = ( (-1,1), (1e-15,np.inf), (1e-15, 50), (1e-15, 50), (1e-15, 10) )
params_Hest_con = scpo.minimize(least_sq, x0=init_vals, args=(prices, strikes, spreads),
                  method='SLSQP', bounds=bounds,
                  constraints=cons, tol=1e-4, options={"maxiter":500}).x

# Result
report_calibration(init_vals, params_Hest_con)

def implied_volatility( price, S0, K, T, r, payoff="call", disp=True ):

    def obj_fun(vol):
        return ( price - BS.BlackScholes(payoff=payoff, S0=S0, K=K, T=T, r=r, sigma=vol) )

    X0 = [0.19, 0.21, 0.5, 1, 2, 3]   # set of initial guess points
    for x0 in X0:
        x, _, solved, _ = scpo.fsolve( obj_fun, x0, full_output=1, xtol=1e-14)
        if solved == 1:
            return x[0]  

    if disp == True:
        print("Strike", K)
    return -1

def implied_vol_minimize( price, S0, K, T, r, payoff="call", disp=True ):
    
    n = 2     # must be even
    def obj_fun(vol):
        return ( BS.BlackScholes(payoff=payoff, S0=S0, K=K, T=T, r=r, sigma=vol) - price )**n
        
    res = scpo.minimize_scalar( obj_fun, bounds=(1e-14, 8), method='bounded')
    if res.success == True:
        return res.x       
    if disp == True:
        print("Strike", K)
    return -1
import time  

start = time.time()
opt_param = Option_param(S0=S0, K=K, T=T, v0=0.04, exercise="European", payoff="call" )
diff_param = Diffusion_process(r=0.05, sig=0.2)
Heston_param = Heston_process(mu=0.1, rho=-0.3, sigma=0.6, theta=0.04, kappa=5)

# Build the pricers
BS = BS_pricer(opt_param, diff_param)
Hest = Heston_pricer(opt_param, Heston_param)

# Compute the price
BS_prices = BS.FFT(strikes)
Hest_prices = Hest.FFT(strikes)
strikes = CALL.Strike.values

BS_compare = pd.DataFrame({"K": strikes, "BS_prices": BS_prices})
BS_compare["IV_BS"] = BS_compare.apply(lambda x: implied_volatility(x["BS_prices"], S0, K=x["K"], T=T, r=r), axis=1)
BS_compare["IV_BS_min"] = BS_compare.apply(lambda x: implied_vol_minimize(x["BS_prices"], S0, K=x["K"], T=T, r=r), axis=1)

plt = BS_compare[["IV_BS","IV_BS_min"]].plot(figsize=(16,5), title = "Compare Root method and loss-minimized method in Black-Scholes model")
plt.set_xlabel("Strike")
plt.set_ylabel("Implied volatility")
end = time.time()
print(end-start)
plt.show()
