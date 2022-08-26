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
  



