[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![Forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)


![GitHub](https://img.shields.io/github/license/white07S/Pricing-Models)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/white07S/Pricing-Models)
![GitHub repo size](https://img.shields.io/github/repo-size/white07S/Pricing-Models)
![GitHub last commit](https://img.shields.io/github/last-commit/white07S/Pricing-Models)

# Pricing-Models

# Binomial Option Pricing Model
* To value American options, the binomial option pricing model employs an iterative approach that employs multiple periods.
* Each iteration of the model has two possible outcomes: a move up or a move down that follows a binomial tree.
* More details: https://en.wikipedia.org/wiki/Binomial_options_pricing_model
```
python3 Pricing-Models/binomial/test.py
```

# Heston model

The Heston process is described by the <font color=blue> system of </font> SDE<font color=blue>s</font>: 

$$ \begin{cases}
dS_t = \mu S_t dt + \sqrt{v_t} S_t dW^1_t \\
dv_t = \kappa (\theta - v_t) dt + \sigma \sqrt{v_t} dW^2_t 
\end{cases}$$

The stock price follows a "geometric Brownian motion" with a stochastic volatility. The square of the volatility (the variance) follows a CIR process.   


The parameters are:
- $\mu$ drift of the stock process
- $\kappa$ mean reversion coefficient of the variance process
- $\theta$ long term mean of the variance process 
- $\sigma$  volatility coefficient of the variance process
- $\rho$ correlation between $W^1$ and $W^2$ i.e. $dW^1_t dW^2_t = \rho dt$
* The option price can be calculated in a variety of ways, including the Lewis method, the Fourier-inversion method, and others. When we need to compute the price of a large number of options with the same maturity, however, the previous methods are no more efficient. To reduce the computational cost, one solution is to use the FFT (Fast Fourier Transform) algorithm.
* Brief Explnation about FFT for Options
  * The integral in the pricing function is(if you want to learn more to derive the integral formula refer: Martin Schmelze (2010), Option Pricing Formulae using Fourier Transform: Theory and Application.  ):
  
  * Note: The [scipy function ifft](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.ifft.html#scipy.fftpack.ifft) returns: `y(j) = (x * exp(2*pi*sqrt(-1)*j*np.arange(n)/n)).mean()` So we need to multiply the return by N.

$$ 
\begin{aligned}
I(k_k) &= \int_{0}^{\infty} e^{ixk_j} \phi_T \bigg(x-\frac{i}{2} \bigg) \frac{1}{x^2 + \frac{1}{4}} dx 
\quad \approx \quad \frac{\Delta x}{3} \sum_{n=0}^{N-1} w_n \, e^{ik_j x_n} \phi_T \bigg( x_n-\frac{i}{2} \bigg) \frac{1}{{x_n}^2 + \frac{1}{4}} \\
&= \frac{\Delta x}{3} \sum_{n=0}^{N-1} w_n \, e^{i (-b + j \Delta k) n \Delta x} \phi_T \bigg( x_n-\frac{i}{2} \bigg) \frac{1}{{x_n}^2 + \frac{1}{4}}. \\
&= \frac{\Delta x}{3} \sum_{n=0}^{N-1} \, e^{i 2\pi j \frac{n}{N}} \; w_n e^{-i b n \Delta x} \phi_T \bigg( x_n-\frac{i}{2} \bigg) \frac{1}{{x_n}^2 + \frac{1}{4}}.
\end{aligned}$$


```
python3 Pricing-Models/heston/test.py
```

# Markov switching multifractal

* The Markov-switching multifractal (MSM) is a model of asset returns created by Laurent E. Calvet and Adlai J. Fisher that combines stochastic volatility components of diverse durations in financial econometrics (the application of statistical methods to economic data).
* MSM has been utilized in financial economics to examine the effects of multifrequency risk on pricing. The excess volatility of stock returns relative to fundamentals and the negative skewness of equity returns have both been partially explained by the models. The creation of multifractal jump-diffusions has also been done using them.
* MSM is tested on equity market and verfied using the return and risk ratio.
* For more informaton: https://en.wikipedia.org/wiki/Markov_switching_multifractal

```
python3 Pricing-Models/markov-Switching/test.py
```
* This models is optimized using **Numba** so if you have any trouble installing numba refer its official documentation or remove the decorater from markov.py file.
* Results
 * Model performance for simuated data vs JPUS data is 3.25sec and 1.23 sec respectively.
 * ![Map](https://github.com/white07S/Pricing-Models/blob/main/models/markovSwitching/sim.png)
  
  
# Garman-Kohlhagen Model
* This approach compares foreign currencies to stocks offering a known dividend return and was created to evaluate currency choices. A "dividend yield" equal to the risk-free interest rate offered in that foreign currency is given to the currency's owner. The Black-Scholes model's stochastic process is also assumed to govern pricing in this model.
* In this project, you can find the comparision between Black-Scholes and Garman-Kohlhagen Model.
* For more information: https://repository.arizona.edu/handle/10150/321771




