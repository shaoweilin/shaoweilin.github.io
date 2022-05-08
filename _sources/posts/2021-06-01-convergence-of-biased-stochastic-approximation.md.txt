---
date: 2021-06-01
excerpts: 2
---

# Convergence of biased stochastic approximation

Using techniques from [biased](2020-12-01-biased-stochastic-approximation/) stochastic approximation {cite}`karimi2019non`, we prove under some regularity conditions the convergence of the online learning algorithm proposed [previously](2021-03-23-biased-stochastic-approximation-with-mutable-processes/) for mutable Markov processes. 

Recall that the algorithm is described by the following updates.

$$ \displaystyle 
X_{n+1} \sim Q_*(X_{n+1} \vert X_{n})$$

$$ \displaystyle 
Z_{n+1} \sim Q_{\lambda_{n}}(Z_{n+1} \vert Z_{n}, X_{n})$$

$$ \displaystyle 
\theta_{n+1} = \theta_{n} + \eta_{n+1} \left.\frac{d}{d\theta} \log P_\theta(Z_{n+1}, X_{n+1} \vert Z_{n},X_{n}) \right\vert _{\theta = \theta_{n}}$$

$$ \displaystyle 
\alpha_{n+1} = \alpha_{n} + \left.\frac{d}{d\lambda} \log Q_{\lambda}(Z_{n+1} \vert  Z_{n},X_{n})\right\vert _{\lambda=\lambda_{n}}$$

$$ \displaystyle 
\gamma_{n+1} = \xi(X_{n+1} \vert X_n) + \log \frac{Q_{\lambda_{n}}(Z_{n+1}\vert Z_{n},X_{n})}{P_{\theta_{n}}(Z_{n+1},X_{n+1}\vert Z_{n},X_{n})} $$

$$ \displaystyle 
\lambda_{n+1} = \lambda_{n} - \eta_{n+1} \alpha_{n+1} \gamma_{n+1}$$ 

where $\xi(X_{n+1} \vert X_n)$ is a function of the environment $X_{n+1}, X_n$ that is independent of the parameters $\theta,\lambda$ and the beliefs $Z_{n+1},Z_n.$

This post is a continuation from our [series](2020-08-28-motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.


### How do we frame the problem in the language of {cite}`karimi2019non`?

To prove the convergence of our [biased](2020-12-01-biased-stochastic-approximation/) stochastic approximation, we cannot apply the standard unbiased stochastic approximation theory of Robbins and Monro. We can however apply the work of {cite}`karimi2019non` which gives some guarantees for biased stochastic approximation involving Markov updates. In this section, we will now derive sufficient conditions for the [convergence](2020-12-01-biased-stochastic-approximation/#theorem-convergence-of-biased-stochastic-approximation) of our biased stochastic approximation.

We [recall](2021-03-23-biased-stochastic-approximation-with-mutable-processes/#what-do-we-assume-about-the-true-distribution-the-model-and-the-learning-objective) some key assumptions about our disciminative model $\{Q_\lambda:\lambda\in\Lambda\},$ generative model $\{P_\theta:\theta \in \Theta\}$ and true distribution $Q_*.$

----

**C1 (Markov property and stationarity).** Under the true distribution $Q_*,$ the process $\{X_t\}$ is Markov. Under each $Q_\lambda$ and each $P_\theta,$ the process $\{(Z_t, X_t)\}$ is Markov, and $Z_t$ and $X_t$ are independent given their past. The true distribution $Q_*$ has a stationary distribution $\bar{\pi}_*,$ and each $Q_\lambda$ has a stationary distribution $\bar{\pi}_\lambda.$

----


First, let $W_n \in \mathcal{W}$ denote 

$$W_n = (W_{n1}, W_{n2}, W_{n3}, W_{n4}, W_{n5}) := (Z_n,X_n,Z_{n-1},X_{n-1}, \alpha_n).$$

Then $W_n$ is a $\lambda$-controlled Markov process. Abusing notation, we write the distribution of $W_n$ as

$$ \begin{array}{rl} & \displaystyle 
Q_\lambda(W_{n+1} \vert W_n) 
\\ & \\ & = \displaystyle
Q_\lambda\left((\,Z_{n+1},X_{n+1},Z_n,X_n,\alpha_{n+1})\, \vert\, (Z_n,X_n,Z_{n-1},X_{n-1},\alpha_n) \, \right)
\\ & \\ & = \displaystyle
Q_*(X_{n+1}\vert X_n) Q_\lambda(Z_{n+1}\vert Z_n, X_n)
\end{array} $$

where we require

$$\alpha_{n+1} = \alpha_n + \frac{d}{d\lambda} \log Q_{\lambda}(Z_{n+1} \vert  Z_{n},X_{n}).$$

Our Lyapunov function is the learning objective

$$ \displaystyle
V(\lambda,\theta) := I_{\bar{Q}_\lambda \Vert P_\theta}(Z_1, X_1 \vert Z_0, X_0).$$

where $\bar{Q}_\lambda$ is the Markov chain with initial distribution $\bar{\pi}_\lambda$ and the same transition probabilities as $Q_\lambda.$

Note that this Lyapunov function is bounded below by zero, because it is a form of relative information. We assume some standard regularity condition on $V(\lambda,\theta).$

----
**C2 ($\ell$-smoothness).** There exists $\ell < \infty$ such that for all $\lambda,\lambda' \in \Lambda$ and $\theta, \theta' \in \Theta,$

$$ \displaystyle 
\left\Vert \frac{\partial V}{\partial \theta}(\lambda,\theta) - \frac{\partial V}{\partial \theta}(\lambda,\theta') \right\Vert \leq \ell \Vert \theta - \theta' \Vert,$$

$$ \displaystyle 
\left\Vert \frac{\partial V}{\partial \lambda}(\lambda,\theta) - \frac{\partial V}{\partial \lambda}(\lambda',\theta) \right\Vert \leq \ell \Vert \lambda - \lambda' \Vert.$$

----

We write the stochastic updates as

$$ \begin{array}{rl} 
W_{n+1} &\sim Q_{\lambda_{n}}(W_{n+1} \vert W_n).
\\ & \\
\displaystyle \theta_{n+1} & 
\displaystyle = \theta_n - \eta_{n+1} G_\theta(W_{n+1};\lambda_n,\theta_n)
\\ & \\ 
\displaystyle \lambda_{n+1} & 
\displaystyle = \lambda_n - \eta_{n+1} G_\lambda(W_{n+1};\lambda_n,\theta_{n})
\end{array} $$

where we have

$$ \begin{array}{rl} 
\displaystyle G_\theta(W_{n+1};\lambda_n,\theta_n) &= 
\displaystyle - \left. \frac{d}{d\theta} \log P_\theta(Z_{n+1}, X_{n+1} \vert Z_{n},X_{n}) \right\vert _{\theta = \theta_n}
\\ & \\
\displaystyle G_\lambda(W_{n+1};\lambda_n,\theta_{n}) &= 
\displaystyle \alpha_{n+1}\left( \log \frac{Q_{\lambda_n}(Z_{n+1}\vert Z_{n},X_{n})}{P_{\theta_{n}}(Z_{n+1},X_{n+1}\vert Z_{n},X_{n})} +\xi(X_{n+1} \vert X_n) \right)
. \end{array}$$

The mean fields of the updates are given by

$$ \begin{array}{rl} 
\displaystyle g_\theta(\lambda_n, \theta_n) & =
\displaystyle - \lim_{T\rightarrow \infty} \mathbb{E}_{Q_{\lambda_n}} \left[ \left.\frac{d}{d\theta} \log P_\theta(Z_{T+1}, X_{T+1} \vert Z_{T},X_{T})\right\vert _{\theta = \theta_n} \right]
\\ & \\ & =
\displaystyle \frac{\partial V}{\partial\theta}(\lambda_n,\theta_n) 
\\ & \\
\displaystyle g_\lambda(\lambda_n,\theta_{n}) & =
\displaystyle
\lim_{T\rightarrow \infty} \mathbb{E}_{Q_{\lambda_n}} \Bigg[ \left( \log \frac{Q_{\lambda_n}(Z_{T+1}\vert Z_{T},X_{T})}{P_{\theta_{n}}(Z_{T+1},X_{T+1}\vert Z_{T},X_{T})} +\xi(X_{T+1} \vert X_T)\right) 
\\ & \\ & \quad\quad \displaystyle \times \sum_{t=0}^{T} \left. \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \right\vert_{\lambda=\lambda_n} \Bigg]
\\ & \\ & =
\displaystyle
\lim_{T\rightarrow \infty} \mathbb{E}_{Q_{\lambda_n}} \Bigg[ \left( \log \frac{Q_{\lambda_n}(Z_{T+1},X_{T+1}\vert Z_{T},X_{T})}{P_{\theta_{n}}(Z_{T+1},X_{T+1}\vert Z_{T},X_{T})} -\log Q_*(X_{T+1}\vert X_T) + \xi(X_{T+1} \vert X_T)\right)  
\\ & \\ & \quad\quad 
\displaystyle \times \sum_{t=0}^{T} \left. \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \right\vert_{\lambda=\lambda_n} \Bigg]
\\ & \\ & =
\displaystyle \frac{\partial V}{\partial\lambda}(\lambda_n,\theta_{n}) 
+ \lim_{T\rightarrow \infty} \mathbb{E}_{Q_{\lambda_n}} \Bigg[ \left( -\log Q_*(X_{T+1}\vert X_T) + \xi(X_{T+1} \vert X_T) \right) 
\\ & \\ & \quad \quad
\displaystyle \times \sum_{t=0}^{T} \left. \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \right\vert_{\lambda=\lambda_n} \Bigg]
\\ & \\ & =
\displaystyle \frac{\partial V}{\partial\lambda}(\lambda_n,\theta_{n+1}) 
. \end{array}$$

where the last equality follows from a [formula](2021-03-23-biased-stochastic-approximation-with-mutable-processes/#appendix-discriminative-model-update) for integrals over the derivative of a stationary distribution:

$$ \begin{array}{rl} &
\displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_{\lambda_n}} \Bigg[ \left( -\log Q_*(X_{T+1}\vert X_T) + \xi(X_{T+1} \vert X_T) \right) \sum_{t=0}^{T} \left. \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \right\vert_{\lambda=\lambda_n} \Bigg]
\\ & \\ & =
\displaystyle \frac{\partial }{\partial\lambda} \int \bar{\pi}_\lambda(dZ_0,dX_0) Q_\lambda(dZ_1,dX_1\vert Z_0,X_0) \left( -\log Q_*(X_{1}\vert X_0) + \xi(X_1 \vert X_0) \right)
\\ & \\ & =
\displaystyle \frac{\partial }{\partial\lambda} \int \bar{\pi}_*(dX_0) \bar{\pi}_\lambda(dZ_0\vert X_0) Q_*(dX_1\vert X_0)Q_\lambda(dZ_1\vert Z_0,X_0) \left( -\log Q_*(X_{1}\vert X_0) + \xi(X_1 \vert X_0) \right)
\\ & \\ & =
\displaystyle \frac{\partial }{\partial\lambda} \int \bar{\pi}_*(dX_0) Q_*(dX_1\vert X_0)\left( -\log Q_*(X_{1}\vert X_0) + \xi(X_1 \vert X_0) \right)
\\ & \\ & =
0 . \end{array}$$

It should not be surprising that terms depending only on the $X_t$ drop out of the mean field, because the mean field involves an integral over the _derivative_ of a stationary distribution with respect to $\lambda$ but the distribution of the $X_t$ is independent of $\lambda.$ The situation is similar to that of the following simplified example.

$$ \begin{array}{rl} &
\displaystyle \int r(X) \,\frac{d }{d\lambda} \bar{\pi}_\lambda(dZ\vert X)\bar{\pi}(dX)  
\\ & \\ & =
\displaystyle \int \bar{\pi}(dX) r(X) \, \frac{d}{d\lambda} \int  \bar{\pi}_\lambda (dZ\vert X) 
\\ & \\ & =
\displaystyle \int \bar{\pi}(dX) r(X) \,  \frac{d}{d\lambda} 1 
\\ & \\ & =
0 . \end{array}$$

From the mean fields, we have the following correction terms. 

$$\begin{array}{rl} 
\displaystyle E_\theta(W_{n+1};\lambda_n,\theta_n) &= 
\displaystyle G_\theta(W_{n+1};\lambda_n,\theta_n) - g_\theta(\lambda_n, \theta_n)
\\ & \\
\displaystyle E_\lambda(W_{n+1};\lambda_n,\theta_{n}) &= 
\displaystyle G_\lambda(W_{n+1};\lambda_n,\theta_{n}) - g_\lambda(\lambda_n,\theta_{n})
. \end{array}$$

Note that the _total expectation_ (with respect to the stationary distribution $\bar{\pi}_\lambda$ on $W_{n+1}$) of each correction term is zero by definition. Now, if the _conditional expectation_ (given the past $W_{n}, \ldots, W_1$) is zero, then the stochastic approximation is _unbiased_. Otherwise, the stochastic approximation is _biased_, which is the case for our problem. 


### How do we prove convergence?

First, we make some common assumptions about the step sizes and the stop time for the stochastic approximation.

----

**<a id="assumption-step-sizes-and-stop-time"></a>C3 (Step sizes and stop time).** For all $k \geq 1,$ the step sizes are given by

$$ \displaystyle
\eta_k = \eta_1 k^{-1/2}$$

for some fixed $\eta_1 > 0.$ Given a fixed time limit $T > 0,$ the stochastic approximation stops randomly after $N \in \{0, \ldots, T\}$ steps according to the probabilities

$$ \displaystyle 
\mathbb{P}[N=k] = \eta_{k+1} \left(\sum_{j=0}^T \eta_{j+1} \right)^{-1}.$$

----

Similar to a previous [analysis](2020-12-01-biased-stochastic-approximation/#theorem-convergence-of-biased-stochastic-approximation), our goal is to find an upper bound such that

$$ \displaystyle 
\mathbb{E}\left[\left\Vert g_\theta(\lambda_N, \theta_N) \right\Vert^2\right] + \mathbb{E}\left[\left\Vert g_\lambda(\lambda_N,\theta_{N}) \right\Vert^2\right] \leq C(T) \rightarrow 0 $$

as $T \rightarrow \infty,$ where the expectations are taken over stochastic updates in the stochastic approximation and over the random stopping time. 

Sometimes, it is not possible to prove convergence of the above upper bound to zero because of certain relaxations (e.g. estimating the true conditional entropy) in the stochastic approximation. Instead, we may prove that the upper bound converges to some positive constant.

Either way, we get some control on the mean and variance of the lengths $\left\Vert g_\theta(\lambda_N, \theta_N) \right\Vert$ and $\left\Vert g_\lambda(\lambda_N,\theta_{N}) \right\Vert$ of the mean fields or total expectations of the stochastic updates, despite the fact that the parameters $\theta_n$ and $\lambda_n$ are continually fluctuating because of the stochastic updates.

Note that by definition

$$ \displaystyle
\mathbb{E}\left[\left\Vert g_\theta(\lambda_N, \theta_N) \right\Vert^2\right] = \frac{\sum_{n=0}^T \eta_{n+1} \mathbb{E}\left[ \left\Vert g_\theta(\lambda_n, \theta_n) \right\Vert^2 \right]}{\sum_{n=0}^T \eta_{n+1}}, $$

$$ \displaystyle
\mathbb{E}\left[\left\Vert g_\lambda(\lambda_N, \theta_{N}) \right\Vert^2\right] = \frac{\sum_{n=0}^T \eta_{n+1} \mathbb{E}\left[ \left\Vert g_\lambda(\lambda_n, \theta_{n}) \right\Vert^2 \right]}{\sum_{n=0}^T \eta_{n+1}}. $$

Following the previous [analysis](2020-12-01-biased-stochastic-approximation/#theorem-convergence-of-biased-stochastic-approximation), the proof starts with the $\ell$-smoothness of $V(\lambda,\theta)$ which implies 

$$ \begin{array}{rl} &
\displaystyle
V(\lambda_{n+1},\theta_{n+1}) - V(\lambda_n,\theta_n) 
\\ & \\ & \leq  
\displaystyle -\eta_{n+1}\left\langle \frac{\partial V}{\partial\lambda}(\lambda_n,\theta_{n}) , G_\lambda(W_{n+1}; \lambda_n,\theta_{n}) \right\rangle
\\ & \\ & \quad
\displaystyle -\eta_{n+1}\left\langle \frac{\partial V}{\partial\theta}(\lambda_n,\theta_n) , G_\theta(W_{n+1}; \lambda_n,\theta_{n}) \right\rangle
\\ & \\ & \quad
\displaystyle + \frac{\ell}{2} \eta_{n+1}^2 \left( \left\Vert G_\lambda(W_{n+1}; \lambda_n,\theta_{n}) \right\Vert^2 + \left\Vert G_\theta(W_{n+1}; \lambda_n,\theta_n) \right\Vert^2 \right)
.\end{array} $$

We now substitute the mean fields and correction terms. After some rearranging and summing from $n=0$ to $n=T$, we have

$$ \begin{array}{rl} &
\displaystyle
\sum_{n=0}^T \eta_{n+1}\left\langle \frac{\partial V}{\partial\lambda}(\lambda_n,\theta_{n}) , g_\lambda(\lambda_n,\theta_{n}) \right\rangle
\\ & \\ & +
\displaystyle
\sum_{n=0}^T \eta_{n+1}\left\langle \frac{\partial V}{\partial\theta}(\lambda_n,\theta_n) , g_\theta(\lambda_n,\theta_{n}) \right\rangle 
\\ & \\ & \leq  
V(\lambda_{0},\theta_{0}) - V(\lambda_{T+1},\theta_{T+1})
\\ & \\ & \quad
\displaystyle -\eta_{n+1}\left\langle \frac{\partial V}{\partial\lambda}(\lambda_n,\theta_{n}) , \sum_{n=0}^T E_\lambda(W_{n+1}; \lambda_n,\theta_{n}) \right\rangle
\\ & \\ & \quad
\displaystyle -\eta_{n+1}\left\langle \frac{\partial V}{\partial\theta}(\lambda_n,\theta_n) , \sum_{n=0}^T E_\theta(W_{n+1}; \lambda_n,\theta_{n}) \right\rangle
\\ & \\ & \quad
\displaystyle + \frac{\ell}{2} \eta_{n+1}^2 \sum_{n=0}^T \left( \left\Vert g_\lambda(\lambda_n,\theta_{n}) \right\Vert^2 + \left\Vert E_\lambda(W_{n+1}; \lambda_n,\theta_{n}) \right\Vert^2 \right)
\\ & \\ & \quad
\displaystyle + \frac{\ell}{2} \eta_{n+1}^2 \sum_{n=0}^T \left( \left\Vert g_\theta(\lambda_n,\theta_{n}) \right\Vert^2 + \left\Vert E_\theta(W_{n+1}; \lambda_n,\theta_n) \right\Vert^2 \right)
.\end{array} $$

The left-hand-side of the inequality can be bounded below by some affine function of $\left\Vert g_\lambda(\lambda_n,\theta_{n}) \right\Vert^2$ and $ \left\Vert g_\theta(\lambda_n,\theta_{n}) \right\Vert^2.$ On the right-hand-side of the inequality, we assume that the Lyapunov difference $V(\lambda_{0},\theta_{0}) - V(\lambda_{T+1},\theta_{T+1})$ and the squared corrections $\left\Vert E_\lambda(W_{n+1}; \lambda_n,\theta_{n}) \right\Vert^2$ and $\left\Vert E_\theta(W_{n+1}; \lambda_n,\theta_n) \right\Vert^2$ are bounded above by constants. 

----

**C4 (Correction bound).** There exists $\sigma < \infty$ such that for all $\lambda \in \Lambda, \theta \in \Theta, w \in \mathcal{W},$

$$ \displaystyle
\Vert E_\lambda(w; \lambda,\theta) \Vert \leq \sigma,$$

$$ \displaystyle
\Vert E_\theta(w; \lambda,\theta) \Vert \leq \sigma.$$

----

Overall, if we also have control over the terms

$$ \displaystyle
\left\langle \frac{\partial V}{\partial\lambda}(\lambda_n,\theta_{n}) , \sum_{n=0}^T E_\lambda(W_{n+1}; \lambda_n,\theta_{n}) \right\rangle, $$

$$ \displaystyle
\left\langle \frac{\partial V}{\partial\theta}(\lambda_n,\theta_n) , \sum_{n=0}^T E_\theta(W_{n+1}; \lambda_n,\theta_{n}) \right\rangle, $$

then we can bound $\sum_{n=0}^T \eta_{n+1}\left\Vert g_\lambda(\lambda_n,\theta_{n}) \right\Vert^2$ and $\sum_{n=0}^T \eta_{n+1} \left\Vert g_\theta(\lambda_n,\theta_{n}) \right\Vert^2,$ which in turn gives us a bound on $ \mathbb{E}[\left\Vert g_\theta(\lambda_N, \theta_N) \right\Vert^2] + \mathbb{E}[\left\Vert g_\lambda(\lambda_N,\theta_{N}) \right\Vert^2].$

Bounding the above two terms will require solutions of the Poisson equations for $E_\lambda(W_{n+1}; \lambda_n,\theta_{n})$ and $E_\theta(W_{n+1}; \lambda_n,\theta_{n}).$

### What are the corresponding Poisson equations?

We study the first Poisson equation

$$ \displaystyle
L_{\lambda} \hat{E}_{\theta} (w; \lambda,\theta) = E_{\theta}(w; \lambda,\theta)$$

where $w = (z_1, x_1, z_0, x_0,\alpha_1),$

$$ \displaystyle
L_{\lambda} \hat{E}_{\theta} (w;\lambda,\theta) = Q_{\lambda} \hat{E}_{\theta} (w;\lambda,\theta)- \hat{E}_{\theta} (w;\lambda,\theta).$$

A candidate solution (if it is well-defined) of the Poisson equation is

$$ \displaystyle
\hat{E}_{\theta} (w;\lambda,\theta) = - \displaystyle \sum_{t=0}^\infty Q_{\lambda}^t E_{\theta}(w;\lambda,\theta)$$

where $Q^t_\lambda$ denotes $t$ applications of the Markov kernel $Q_\lambda.$

Now, for all $t \geq 1,$

$$ \begin{array}{rl} & 
\displaystyle Q_\lambda^t G_{\theta}(w;\lambda,\theta) 
\\ & \\ & = 
\displaystyle \frac{\partial}{\partial\theta} \int \hat{Q}_\lambda(dZ_0,dX_0)\hat{Q}_\lambda^t(dZ_{t}, dX_{t} \vert Z_0, X_0) \log \frac{\hat{Q}_\lambda(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})}{P_\theta(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})} 
\\ & \\ & = 
\displaystyle \frac{\partial}{\partial\theta}  I_{\hat{Q}_\lambda \Vert P_\theta}(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})  \end{array}$$

where $\hat{Q}_\lambda$ is the distribution of the Markov chain that initializes $(Z_0,X_0)$ with the state $(z_1, x_1)$ and has transition probabilities $Q_\lambda.$ Therefore,

$$ \begin{array}{rl} & 
\displaystyle
Q_{\lambda} \hat{E}_{\theta} (w;\lambda,\theta)
\\ & \\ & = 
\displaystyle
-\lim_{T \rightarrow \infty} \sum_{t=1}^{T} Q_\lambda^{t} E_{\theta}(w;\lambda,\theta)  
\\ & \\ & = 
\displaystyle
- \lim_{T \rightarrow \infty} \sum_{t=1}^{T} Q_\lambda^{t} \left(G_{\theta}(w;\lambda,\theta)-g_{\theta}(w;\lambda,\theta)  \right)
\\ & \\ & = 
\displaystyle
\lim_{T \rightarrow \infty} \sum_{t=1}^{T} g_{\theta}(w;\lambda,\theta)- Q_\lambda^{t}G_{\theta}(w;\lambda,\theta) 
\\ & \\ & = 
\displaystyle \lim_{T \rightarrow \infty} \frac{\partial}{\partial\theta} \Big( \sum_{t=1}^{T} I_{\bar{Q}_\lambda \Vert P_\theta}(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})
 - \sum_{t=1}^{T}  I_{\hat{Q}_\lambda \Vert P_\theta}(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1}) \Big) 
 \\ & \\ & = 
 \displaystyle \lim_{T \rightarrow \infty} \frac{\partial}{\partial\theta} \Big( I_{\bar{Q}_\lambda \Vert P_\theta}(Z_{[1..T]}, X_{[1..T]} \vert Z_{0}, X_{0}) -  I_{\hat{Q}_\lambda \Vert P_\theta}(Z_{[1..T]}, X_{[1..T]} \vert Z_{0}, X_{0}) \Big) 
\end{array}$$

where the last equality follows from the chain rule for conditional relative information. 

As for the second Poisson equation

$$ \displaystyle
L_{\lambda} \hat{E}_{\lambda} (w; \lambda,\theta) = E_{\lambda}(w; \lambda,\theta)$$

a candidate solution is

$$ \displaystyle
\hat{E}_{\lambda} (w;\lambda,\theta) = - \displaystyle \sum_{t=0}^\infty Q_{\lambda}^t E_{\lambda}(w;\lambda,\theta).$$

Now, as shown in the [appendix](2021-06-01-convergence-of-biased-stochastic-approximation/#appendix--poisson-equation-for-discriminative-model-update), for all $t \geq 1,$

$$ \begin{array}{rl} & 
\displaystyle Q_\lambda^t G_{\lambda}(w;\lambda,\theta) 
\\ & \\ & = 
\left( I_{\hat{Q}_\lambda\Vert P_\theta}(Z_t,X_t\vert Z_{t-1},X_{t-1}) + H_{\hat{Q}_*}(X_t\vert X_{t-1}) + \xi_{\hat{Q}_*}(X_t \vert X_{t-1}) \right) \alpha_1
\\ & \\ & \quad 
\displaystyle + \frac{\partial}{\partial \lambda} I_{\hat{Q}_\lambda\Vert P_\theta}(Z_t,X_t\vert Z_{t-1},X_{t-1})
. \end{array}$$

Therefore, 

$$ \begin{array}{rl} & 
\displaystyle
Q_{\lambda} \hat{E}_{\lambda} (w;\lambda,\theta)
\\ & \\ & = 
\displaystyle
\lim_{T \rightarrow \infty} \sum_{t=1}^{T} g_{\lambda}(w;\lambda,\theta)- Q_\lambda^{t}G_{\lambda}(w;\lambda,\theta) 
\\ & \\ & = 
\displaystyle \lim_{T \rightarrow \infty} \frac{\partial}{\partial\lambda} \Big( \sum_{t=1}^{T} I_{\bar{Q}_\lambda \Vert P_\theta}(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})
 - \sum_{t=1}^{T}  I_{\hat{Q}_\lambda \Vert P_\theta}(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1}) \Big) 
 \\ & \\ & \quad \quad
 \displaystyle - \left( \sum_{t=1}^{T}  I_{\hat{Q}_\lambda\Vert P_\theta}(Z_t,X_t\vert Z_{t-1},X_{t-1}) + \sum_{t=1}^{T} H_{\hat{Q}_*}(X_t\vert X_{t-1}) + \sum_{t=1}^T \xi_{\hat{Q}_*}(X_t \vert X_{t-1}) \right) \alpha_1
 \\ & \\ & = 
 \displaystyle \lim_{T \rightarrow \infty} \frac{\partial}{\partial\lambda} \Big( I_{\bar{Q}_\lambda \Vert P_\theta}(Z_{[1..T]}, X_{[1..T]} \vert Z_{0}, X_{0}) -  I_{\hat{Q}_\lambda \Vert P_\theta}(Z_{[1..T]}, X_{[1..T]} \vert Z_{0}, X_{0}) \Big) 
 \\ & \\ & \quad \quad
 \displaystyle - \left( I_{\hat{Q}_\lambda\Vert P_\theta}(Z_{[1..T]},X_{[1..T]}\vert Z_0,X_0) +  H_{\hat{Q}_*}(X_{[1..T]}\vert X_0) + \xi_{\hat{Q}_*}(X_{[1..T]} \vert X_{0})\right) \alpha_1
\end{array}$$

where

$$ \displaystyle 
\xi_{\hat{Q}_*}(X_{[1..T]} \vert X_{0}) = \sum_{t=1}^T \xi_{\hat{Q}_*}(X_t \vert X_{t-1}).$$

Bringing them all together,

$$ \begin{array}{rl}
V(\lambda,\theta) &= I_{\bar{Q}_\lambda \Vert P_\theta}(Z_1, X_1 \vert Z_0, X_0) 
\\ & \\ 
g_\theta(\lambda, \theta) & = 
\displaystyle \frac{\partial V}{\partial \theta}(\lambda, \theta) 
\\ & \\ 
g_\lambda(\lambda, \theta) & = 
\displaystyle \frac{\partial V}{\partial \lambda}(\lambda, \theta) 
\\ & \\ 
E_{\theta} (w;\lambda,\theta) & = 
\displaystyle - \frac{\partial}{\partial\theta} \Big( V(\lambda,\theta)+ \log P_\theta(z_1, x_1 \vert z_0, x_0) \Big) 
\\ & \\ 
E_\lambda(w;\lambda,\theta) &= 
\displaystyle \alpha_1 \left( \log \frac{Q_{\lambda}(z_1\vert z_0,x_0)}{P_{\theta}(z_1,x_1\vert z_0,x_0)} -\xi(x_1 \vert x_0) \right) - \frac{\partial V}{\partial \lambda}(\lambda, \theta) 
\\ & \\
Q_{\lambda} \hat{E}_{\theta} (w;\lambda,\theta) & =
\displaystyle \lim_{T \rightarrow \infty} \frac{\partial}{\partial\theta} \Big( I_{\bar{Q}_\lambda \Vert P_\theta}(Z_{[1..T]}, X_{[1..T]} \vert Z_{0}, X_{0})  
-  I_{\hat{Q}_\lambda \Vert P_\theta}(Z_{[1..T]}, X_{[1..T]} \vert Z_{0}, X_{0}) \Big)
\\ & \\ 
Q_{\lambda} \hat{E}_{\lambda} (w;\lambda,\theta) & = 
 \displaystyle \lim_{T \rightarrow \infty} \frac{\partial}{\partial\lambda} \Big( I_{\bar{Q}_\lambda \Vert P_\theta}(Z_{[1..T]}, X_{[1..T]} \vert Z_{0}, X_{0}) -  I_{\hat{Q}_\lambda \Vert P_\theta}(Z_{[1..T]}, X_{[1..T]} \vert Z_{0}, X_{0}) \Big) 
 \\ & \\ & \quad \quad
 \displaystyle - \left( I_{\hat{Q}_\lambda\Vert P_\theta}(Z_{[1..T]},X_{[1..T]}\vert Z_0,X_0) +  H_{\hat{Q}_*}(X_{[1..T]}\vert X_0) + \xi_{\hat{Q}_*}(X_{[1..T]} \vert X_{0}) \right) \alpha_1
\\ & \\ 
\hat{E}_{\theta} (w;\lambda,\theta) & = 
Q_{\lambda} \hat{E}_{\theta} (w;\lambda,\theta) - E_{\theta} (w;\lambda,\theta)
\\ & \\ 
\hat{E}_{\lambda} (w;\lambda,\theta) & = 
Q_{\lambda} \hat{E}_{\lambda} (w;\lambda,\theta) - E_{\lambda} (w;\lambda,\theta) 
\end{array}$$


We impose the following regularity conditions.

----
**C5 (Solution of Poisson equation).** The limits 

$$ \displaystyle
Q_\lambda\hat{E}_\lambda : \mathcal{W} \times \Lambda \times \Theta \rightarrow \Lambda $$ 

$$ \displaystyle
Q_\lambda\hat{E}_\theta : \mathcal{W} \times \Lambda \times \Theta \rightarrow \Theta$$ 

are well-defined and measurable.

**C6 (Regularity of solution).** There exists $\ell_0, \ell_1 < \infty$ such that for all $\lambda,\lambda' \in \Lambda$ and $\theta, \theta' \in \Theta$ and $w \in \mathcal{W},$

$$ \displaystyle
\Vert \hat{E}_{\lambda} (w;\lambda,\theta) \Vert \leq \ell_0, \quad \Vert Q_\lambda \hat{E}_{\lambda} (w;\lambda,\theta)  \Vert \leq \ell_0,$$

$$ \displaystyle
\Vert \hat{E}_{\theta} (w;\lambda,\theta) \Vert \leq \ell_0, \quad \Vert Q_\lambda \hat{E}_{\theta} (w;\lambda,\theta)  \Vert \leq \ell_0,$$

$$ \displaystyle
\Vert Q_\lambda \hat{E}_{\lambda} (w;\lambda,\theta) -Q_{\lambda'} \hat{E}_{\lambda} (w;\lambda',\theta)  \Vert \leq \ell_1 \Vert \lambda - \lambda' \Vert,$$

$$ \displaystyle
\Vert Q_\lambda \hat{E}_{\lambda} (w;\lambda,\theta) -Q_{\lambda} \hat{E}_{\lambda} (w;\lambda,\theta')  \Vert \leq \ell_1 \Vert \theta - \theta' \Vert.$$

----

**<a id="theorem-convergence-of-online-learning"></a>Theorem (Convergence of online learning).** Suppose that we have stochastic approximation

$$ \displaystyle 
X_{n+1} \sim Q_*(X_{n+1} \vert X_{n})$$

$$ \displaystyle 
Z_{n+1} \sim Q_{\lambda_{n}}(Z_{n+1} \vert Z_{n}, X_{n})$$

$$ \displaystyle 
\theta_{n+1} = \theta_{n} + \eta_{n+1} \left.\frac{d}{d\theta} \log P_\theta(Z_{n+1}, X_{n+1} \vert Z_{n},X_{n}) \right\vert _{\theta = \theta_{n}}$$

$$ \displaystyle 
\alpha_{n+1} = \alpha_{n} + \left.\frac{d}{d\lambda} \log Q_{\lambda}(Z_{n+1} \vert  Z_{n},X_{n})\right\vert _{\lambda=\lambda_{n}}$$

$$ \displaystyle 
\gamma_{n+1} = \xi(X_{n+1} \vert X_n) + \log \frac{Q_{\lambda_{n}}(Z_{n+1}\vert Z_{n},X_{n})}{P_{\theta_{n}}(Z_{n+1},X_{n+1}\vert Z_{n},X_{n})} $$

$$ \displaystyle 
\lambda_{n+1} = \lambda_{n} - \eta_{n+1} \alpha_{n+1} \gamma_{n+1}$$ 

Then assuming C1-C6 and sufficiently small initial step size $\eta_1,$
we have

$$ \displaystyle
\mathbb{E}\left[\left\Vert g_\theta(\lambda_N, \theta_N) \right\Vert^2\right] + \mathbb{E}\left[\left\Vert g_\lambda(\lambda_N,\theta_{N}) \right\Vert^2\right] = O(\log T / \sqrt{T} )$$

where the maximum time $T$ was defined in [C3](#assumption-step-sizes-and-stop-time).

----

### Appendix : Poisson equation for discriminative model update

In this section, we derive the terms $Q_\lambda^t G_{\lambda}(w;\lambda,\theta)$ appearing in our candidate solution to the Poisson equation for the discriminative model update.

Given $w = (z_1, x_1, z_0, x_0,\alpha_1),$ for all $t \geq 1$ we have

$$ \begin{array}{rl} & 
\displaystyle Q_\lambda^t G_{\lambda}(w;\lambda,\theta) 
\\ & \\ & = 
\displaystyle \int \hat{Q}_\lambda(dZ_0,dX_0)Q_\lambda^t(dZ_{t}, dX_{t} \vert Z_0, X_0) 
\\ & \\ & \quad \quad 
\displaystyle \times \left(\log \frac{Q_\lambda(Z_{t}, X_t \vert Z_{t-1}, X_{t-1})}{P_\theta(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})} - \log Q_*(X_t\vert X_{t-1}) +\xi(X_{t} \vert X_{t-1})\right)
\\ & \\ & \quad \quad 
\displaystyle \times \left( \alpha_1 + \sum_{s=1}^t \frac{d}{d\lambda} \log Q_\lambda(Z_{s}\vert Z_{s-1}, X_{s-1} ) \right)
. \end{array}$$


We observe that

$$ \begin{array}{rl} & 
\displaystyle \int \hat{Q}_\lambda(dZ_0,dX_0) Q_\lambda^t(dZ_{t}, dX_{t} \vert Z_0, X_0) \left(\log \frac{Q_\lambda(Z_{t}, X_t \vert Z_{t-1}, X_{t-1})}{P_\theta(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})} \right) 
\\ & \\ & = 
\displaystyle I_{\hat{Q}_\lambda\Vert P_\theta}(Z_t,X_t\vert Z_{t-1},X_{t-1})
\end{array}$$

and that

$$ \begin{array}{rl} & 
\displaystyle \int \hat{Q}_\lambda(dZ_0,dX_0) Q_\lambda^t(dZ_{t}, dX_{t} \vert Z_0, X_0) \left(- \log Q_*(X_t\vert X_{t-1}) +\xi(X_{t} \vert X_{t-1}) \right) 
\\ & \\ & = 
\displaystyle
\int \hat{Q}_*(dX_0) Q_*^t(dX_t\vert X_0) \left(- \log Q_*(X_t\vert X_{t-1}) +\xi(X_{t} \vert X_{t-1}) \right) 
\\ & \\ & =
H_{\hat{Q}_*}(X_t\vert X_{t-1}) + \xi_{\hat{Q}_*}(X_{t} \vert X_{t-1})
. \end{array}$$

where $\hat{Q}_*$ is the distribution of the Markov chain that initializes $X_0$ with the state $x_1$ and has transition probabilities $Q_*,$ $H_{\hat{Q}_*}$ is the [conditional entropy](2020-09-08-building-foundations-of-information-theory-on-relative-information/#how-do-we-derive-entropy-from-relative-information), and

$$ \displaystyle
\xi_{\hat{Q}_*}(X_{t} \vert X_{t-1}) = \mathbb{E}_{\hat{Q}_*}\left[\xi(X_{t} \vert X_{t-1})\right]. $$ 

We also note that

$$ \begin{array}{rl} &
\displaystyle
\frac{\partial}{\partial \lambda} I_{\hat{Q}_\lambda\Vert P_\theta}(Z_t,X_t\vert Z_{t-1},X_{t-1})
\\ & \\ & =
\displaystyle \int \hat{Q}_\lambda(dZ_0,dX_0) \left(\frac{\partial}{\partial \lambda}Q_\lambda^t(dZ_{t}, dX_{t} \vert Z_0, X_0) \right) \log \frac{Q_\lambda(Z_{t}, X_t \vert Z_{t-1}, X_{t-1})}{P_\theta(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})} 
\\ & \\ & \quad
\displaystyle + \int \hat{Q}_\lambda(dZ_0,dX_0) Q_\lambda^t(dZ_{t}, dX_{t} \vert Z_0, X_0) \left(\frac{\partial}{\partial \lambda} \log \frac{Q_\lambda(Z_{t}, X_t \vert Z_{t-1}, X_{t-1})}{P_\theta(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})} \right)
\\ & \\ & =
\displaystyle \int \hat{Q}_\lambda(dZ_0,dX_0)  \left(Q_\lambda^t(dZ_{t}, dX_{t} \vert Z_0, X_0)\sum_{s=1}^t \frac{d}{d\lambda} \log Q_\lambda(Z_{s},X_s\vert Z_{s-1}, X_{s-1} ) \right)
\\ & \\ & \quad \quad 
\displaystyle \times 
\left(\log \frac{Q_\lambda(Z_{t}, X_t \vert Z_{t-1}, X_{t-1})}{P_\theta(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})} \right)
\\ & \\ & \quad
\displaystyle + \int \hat{Q}_\lambda(dZ_0,dX_0) Q_\lambda^{t-1}(dZ_{t-1}, dX_{t-1} \vert Z_0, X_0) \left(\frac{d}{d \lambda} \int Q_\lambda(dZ_{t}, dX_t \vert Z_{t-1}, X_{t-1}) \right)
\\ & \\ & =
\displaystyle \int \hat{Q}_\lambda(dZ_0,dX_0)Q_\lambda^t(dZ_{t}, dX_{t} \vert Z_0, X_0) \left(\log \frac{Q_\lambda(Z_{t}, X_t \vert Z_{t-1}, X_{t-1})}{P_\theta(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})} \right)
\\ & \\ & \quad \quad 
\displaystyle \times \left(\sum_{s=1}^t \frac{d}{d\lambda} \log Q_\lambda(Z_{s}\vert Z_{s-1}, X_{s-1} ) \right)
. \end{array}$$

Lastly, 

$$ \begin{array}{rl} &
\displaystyle \int \hat{Q}_\lambda(dZ_0,dX_0) \left( Q_\lambda^t(dZ_{t}, dX_{t} \vert Z_0, X_0) \sum_{s=1}^t \frac{d}{d\lambda} \log Q_\lambda(Z_{s}\vert Z_{s-1}, X_{s-1} ) \right)
\\ & \\ & \quad \quad 
\displaystyle \times \left( - \log Q_*(X_t\vert X_{t-1}) +\xi(X_t \vert X_{t-1})\right)
\\ & \\ & =
\displaystyle 
\frac{d}{d\lambda} \int
\hat{Q}_\lambda(dZ_0,dX_0)  Q_\lambda^t(dZ_{t}, dX_{t} \vert Z_0, X_0)  \left( - \log Q_*(X_t\vert X_{t-1})+\xi(X_t \vert X_{t-1})\right)
\\ & \\ & =
\displaystyle 
\frac{d}{d\lambda} \int
\hat{Q}_*(dX_0)  Q_*^t( dX_{t} \vert X_0)  \left( - \log Q_*(X_t\vert X_{t-1}) +\xi(X_t \vert X_{t-1})\right)
\\ & \\ & =
0 . \end{array}$$

Therefore, 

$$ \begin{array}{rl} & 
\displaystyle Q_\lambda^t G_{\lambda}(w;\lambda,\theta) 
\\ & \\ & = 
\left( I_{\hat{Q}_\lambda\Vert P_\theta}(Z_t,X_t\vert Z_{t-1},X_{t-1}) + H_{\hat{Q}_*}(X_t\vert X_{t-1}) + \xi_{\hat{Q}_*}(X_t \vert X_{t-1}) \right) \alpha_1
\\ & \\ & \quad 
\displaystyle + \frac{\partial}{\partial \lambda} I_{\hat{Q}_\lambda\Vert P_\theta}(Z_t,X_t\vert Z_{t-1},X_{t-1})
. \end{array}$$

### References

```{bibliography}
:filter: docname in docnames
```