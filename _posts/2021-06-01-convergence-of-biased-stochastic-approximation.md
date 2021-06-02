---
layout: post
title: Convergence of biased stochastic approximation
excerpt_separator: <!--more-->
---

Using techniques from [KMMW19], we prove under some regularity conditions the convergence of the [biased](https://shaoweilin.github.io/biased-stochastic-approximation/) approximation proposed [previously](https://shaoweilin.github.io/biased-stochastic-approximation-for-latent-processes/) for latent Markov processes. 

Recall that the biased stochastic approximation is described by the following updates.

$$\displaystyle \theta_{n+1} = \theta_n + \eta_{n+1} \left.\frac{d}{d\theta} \log P_\theta(Z_{n}, X_{n} \vert Z_{n-1},X_{n-1}) \right\vert _{\theta = \theta_n}$$

$$\displaystyle \alpha_{n+1} = \gamma \alpha_n + \left.\frac{d}{d\lambda} \log Q_{\lambda}(Z_{n} \vert  Z_{n-1},X_{n-1})\right\vert _{\lambda=\lambda_n}$$

$$\displaystyle \beta_{n+1} =  \log \frac{Q_{\lambda_n}(Z_{n}\vert Z_{n-1},X_{n-1})}{P_{\theta_{n+1}}(Z_{n}\vert Z_{n-1},X_{n-1})} - \log P_{\theta_{n+1}}(X_{n}\vert Z_{n-1},X_{n-1}) -\xi$$

$$\displaystyle \lambda_{n+1} = \lambda_n - \eta_{n+1} \alpha_{n+1} \beta_{n+1}$$ 

$$\displaystyle X_{n+1} \sim Q_*(X_{n+1} \vert X_{n})$$

$$\displaystyle Z_{n+1} \sim Q_{\lambda_{n+1}}(Z_{n+1} \vert Z_{n}, X_{n})$$

This post is a continuation from our [series](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

<!--more-->

## How do we frame the problem in the language of [KMMW19]?

To prove the convergence of our [biased](https://shaoweilin.github.io/biased-stochastic-approximation/) stochastic approximation, we cannot apply the standard unbiased stochastic approximation theory of Robbins and Monro. We can however apply the work of [KMMW19] which gives some guarantees for biased stochastic approximation involving Markov updates. In this section, we will now derive sufficient conditions for the [convergence](https://shaoweilin.github.io/biased-stochastic-approximation/#theorem-convergence-of-biased-stochastic-approximation) of our biased stochastic approximation.

First, let $$W_n$$ denote 

$$W_n = (W_{n1}, W_{n2}, W_{n3}, W_{n4}) := (Z_n,X_n,Z_{n-1},X_{n-1}).$$

Then $$W_n$$ is a $$\lambda$$-controlled Markov process. Abusing notation, we write the distribution of $$W_n$$ as

$$ \begin{array}{rl} & \displaystyle 
Q_\lambda(W_{n+1} \vert W_n) 
\\ & \\ & = \displaystyle
Q_\lambda\left((\,Z_{n+1},X_{n+1},Z_n,X_n)\, \vert\, (Z_n,X_n,Z_{n-1},X_{n-1}) \, \right)
\\ & \\ & = \displaystyle
Q_*(X_{n+1}\vert X_n) Q_\lambda(Z_{n+1}\vert Z_n, X_n)
. \end{array}
$$

Our Lyapunov function is the learning objective

$$ \displaystyle
V(\lambda,\theta) := H_{\bar{Q}_\lambda \Vert P_\theta}(Z_1, X_1 \vert Z_0, X_0).
$$

We write the stochastic updates as

$$ \begin{array}{rl} 
\displaystyle \theta_{n+1} & 
\displaystyle = \theta_n - \eta_{n+1} G_\theta(W_n;\lambda_n,\theta_n)
\\ & \\ 
\displaystyle \lambda_{n+1} & 
\displaystyle = \lambda_n - \eta_{n+1} G_\lambda(W_n;\lambda_n,\theta_{n+1})
\\ & \\
W_{n+1} &\sim \mathbb{P}(W_{n+1} \vert W_n; \lambda_{n+1}).
\end{array}
$$

where using $$\alpha_n, \beta_n$$ defined previously we have

$$\begin{array}{rl} 
\displaystyle G_\theta(W_n;\lambda_n,\theta_n) &= 
\displaystyle - \left. \frac{d}{d\theta} \log P_\theta(Z_{n}, X_{n} \vert Z_{n-1},X_{n-1}) \right\vert _{\theta = \theta_n}
\\ & \\
\displaystyle G_\lambda(W_n;\lambda_n,\theta_{n+1}) &= 
\displaystyle \alpha_{n+1}\beta_{n+1}
. \end{array}$$

The mean fields of the updates are given by

$$\begin{array}{rl} 
\displaystyle g_\theta(\lambda_n, \theta_n) & =
\displaystyle - \sum_w \bar{\pi}_{\lambda_n}(w) \left. \frac{d}{d\theta} \log P_\theta(w_1, w_2 \vert w_3,w_4) \right\vert _{\theta = \theta_n}
\\ & \\ & =
\displaystyle \frac{\partial}{\partial\theta}V(\lambda_n,\theta_n) 
\\ & \\
\displaystyle g_\lambda(\lambda_n,\theta_{n+1}) & =
\displaystyle \sum_{t=0}^\infty \sum_{w',w} \bar{\pi}_{\lambda_n}(w) Q_{\lambda_n}^t(w'\vert w) \,\,\times
\\ & \\ & \quad \quad
\displaystyle \gamma^t \left( \log \frac{Q_{\lambda_n}(w'_1 \vert w'_3,w'_4)}{P_{\theta_{n+1}}(w'_1 \vert w'_3,w'_4)} - \log P_{\theta_{n+1}}(w'_2 \vert w'_3, w'_4) -\xi \right) \,\,\times 
\\ & \\ & \quad \quad
\displaystyle
\left.\frac{d}{d\lambda} \log Q_\lambda(w_1 \vert  w_3,w_4) \right\vert_{\lambda=\lambda_n}
\\ & \\ & \approx
\displaystyle \frac{\partial}{\partial\lambda}V(\lambda_n,\theta_{n+1}) 
\end{array}$$

where $$Q^t_\lambda$$ denotes the transition probabilities after $$t$$ steps of the Markov chain with distribution $$Q_\lambda.$$

From the mean fields, we have the following correction terms. 

$$\begin{array}{rl} 
\displaystyle E_\theta(W_n;\lambda_n,\theta_n) &= 
\displaystyle G_\theta(W_n;\lambda_n,\theta_n) - g_\theta(\lambda_n, \theta_n)
\\ & \\
\displaystyle E_\lambda(W_n;\lambda_n,\theta_{n+1}) &= 
\displaystyle G_\lambda(W_n;\lambda_n,\theta_{n+1}) - g_\lambda(\lambda_n,\theta_{n+1})
. \end{array}$$

Note that the _total expectation_ (with respect to the stationary distribution $$\bar{\pi}_\lambda$$ on $$W_n$$) of each correction terms is zero by definition. On the other hand, if the _conditional expectation_ (given the past $$W_{n-1}, \ldots, W_0$$) is zero, then the stochastic approximation is _unbiased_. Otherwise, the stochastic approximation is biases, which is the case for our problem. 


## How do we prove convergence?

Given a computation time limit $$T > 0$$ and learning rates $$\eta_1, \ldots, \eta_{T+1},$$ a common assumption [KMMW19] for the termination of the stochastic approximation is to have a random stopping time $$N \in \{0,\ldots, T\}$$ satisfying

$$\displaystyle \mathbb{P}[N=k] \quad \propto \quad \eta_{k+1}.$$

Similar to a previous [analysis](https://shaoweilin.github.io/biased-stochastic-approximation/#theorem-convergence-of-biased-stochastic-approximation), our goal is to find an upper bound such that

$$
\displaystyle \mathbb{E}\left[\left\Vert g_\theta(\lambda_N, \theta_N) \right\Vert^2\right] + \mathbb{E}\left[\left\Vert g_\lambda(\lambda_N,\theta_{N+1}) \right\Vert^2\right] \leq C(T) \rightarrow 0 
$$

as $$T \rightarrow \infty,$$ where the expectations are taken over stochastic updates in the stochastic approximation and over the random stopping time. 

Sometimes, it is not possible to prove convergence of the above upper bound to zero because of certain relaxations (e.g. discounted gradient updates) in the stochastic approximation. Instead, we may prove that the upper bound converges to some positive constant.

Either way, we get some control on the mean and variance of the lengths $$\left\Vert g_\theta(\lambda_N, \theta_N) \right\Vert$$ and $$\left\Vert g_\lambda(\lambda_N,\theta_{N+1}) \right\Vert$$ of the mean fields or total expectations of the stochastic updates, despite the fact that the parameters $$\theta_n$$ and $$\lambda_n$$ are continually fluctuating because of the stochastic updates.

Note that by definition

$$
\mathbb{E}\left[\left\Vert g_\theta(\lambda_N, \theta_N) \right\Vert^2\right] = \frac{\sum_{n=0}^T \eta_{n+1} \mathbb{E}\left[ \left\Vert g_\theta(\lambda_n, \theta_n) \right\Vert^2 \right]}{\sum_{n=0}^T \eta_{n+1}},
$$

$$
\mathbb{E}\left[\left\Vert g_\lambda(\lambda_N, \theta_{N+1}) \right\Vert^2\right] = \frac{\sum_{n=0}^T \eta_{n+1} \mathbb{E}\left[ \left\Vert g_\lambda(\lambda_n, \theta_{n+1}) \right\Vert^2 \right]}{\sum_{n=0}^T \eta_{n+1}}.
$$

Following the previous [analysis](https://shaoweilin.github.io/biased-stochastic-approximation/#theorem-convergence-of-biased-stochastic-approximation), the proof starts with the $$\ell$$-smoothness of $$V(\lambda,\theta)$$ which implies 

$$\begin{array}{rl} &
\displaystyle
V(\lambda_{n+1},\eta_{n+1}) - V(\lambda_n,\theta_n) 
\\ & \\ & \leq  
\displaystyle -\eta_{n+1}\left\langle \frac{\partial}{\partial\lambda} V(\lambda_n,\theta_{n+1}) , G_\lambda(W_n; \lambda_n,\theta_{n+1}) \right\rangle
\\ & \\ & \quad
\displaystyle -\eta_{n+1}\left\langle \frac{\partial}{\partial\theta} V(\lambda_n,\theta_n) , G_\theta(W_n; \lambda_n,\theta_{n}) \right\rangle
\\ & \\ & \quad
\displaystyle + \frac{\ell}{2} \eta_{n+1}^2 \left( \left\Vert G_\lambda(W_n; \lambda_n,\theta_{n+1}) \right\Vert^2 + \left\Vert G_\theta(W_n; \lambda_n,\theta_n) \right\Vert^2 \right)
.\end{array}
$$

We now substitute the mean fields and correction terms. After some rearranging and summing from $$n=0$$ to $$n=T$$, we have

$$\begin{array}{rl} &
\displaystyle
\sum_{n=0}^T \eta_{n+1}\left\langle \frac{\partial}{\partial\lambda} V(\lambda_n,\theta_{n+1}) , g_\lambda(\lambda_n,\theta_{n+1}) \right\rangle
\\ & \\ & +
\displaystyle
\sum_{n=0}^T \eta_{n+1}\left\langle \frac{\partial}{\partial\theta} V(\lambda_n,\theta_n) , g_\theta(\lambda_n,\theta_{n}) \right\rangle 
\\ & \\ & \leq  
V(\lambda_{0},\eta_{0}) - V(\lambda_{T+1},\theta_{T+1})
\\ & \\ & \quad
\displaystyle -\eta_{n+1}\left\langle \frac{\partial}{\partial\lambda} V(\lambda_n,\theta_{n+1}) , \sum_{n=0}^T E_\lambda(W_n; \lambda_n,\theta_{n+1}) \right\rangle
\\ & \\ & \quad
\displaystyle -\eta_{n+1}\left\langle \frac{\partial}{\partial\theta} V(\lambda_n,\theta_n) , \sum_{n=0}^T E_\theta(W_n; \lambda_n,\theta_{n}) \right\rangle
\\ & \\ & \quad
\displaystyle + \frac{\ell}{2} \eta_{n+1}^2 \sum_{n=0}^T \left( \left\Vert g_\lambda(\lambda_n,\theta_{n+1}) \right\Vert^2 + \left\Vert E_\lambda(W_n; \lambda_n,\theta_{n+1}) \right\Vert^2 \right)
\\ & \\ & \quad
\displaystyle + \frac{\ell}{2} \eta_{n+1}^2 \sum_{n=0}^T \left( \left\Vert g_\theta(\lambda_n,\theta_{n}) \right\Vert^2 + \left\Vert E_\theta(W_n; \lambda_n,\theta_n) \right\Vert^2 \right)
.\end{array}
$$

The left-hand-side of the inequality can be bounded below by some affine function of $$\left\Vert g_\lambda(\lambda_n,\theta_{n+1}) \right\Vert^2$$ and $$ \left\Vert g_\theta(\lambda_n,\theta_{n}) \right\Vert^2.$$ On the right-hand-side of the inequality, we may assume that the Lyapunov difference $$V(\lambda_{0},\eta_{0}) - V(\lambda_{T+1},\theta_{T+1})$$ and the squared corrections $$\left\Vert E_\lambda(W_n; \lambda_n,\theta_{n+1}) \right\Vert^2$$ and $$\left\Vert E_\theta(W_n; \lambda_n,\theta_n) \right\Vert^2$$ are bounded above by constants. 

Overall, if we also have control over the terms

$$
\displaystyle \left\langle \frac{\partial}{\partial\lambda} V(\lambda_n,\theta_{n+1}) , \sum_{n=0}^T E_\lambda(W_n; \lambda_n,\theta_{n+1}) \right\rangle,
$$

$$
\displaystyle \left\langle \frac{\partial}{\partial\theta} V(\lambda_n,\theta_n) , \sum_{n=0}^T E_\theta(W_n; \lambda_n,\theta_{n}) \right\rangle,
$$

then we can bound $$\sum_{n=0}^T \eta_{n+1}\left\Vert g_\lambda(\lambda_n,\theta_{n+1}) \right\Vert^2$$ and $$\sum_{n=0}^T \eta_{n+1} \left\Vert g_\theta(\lambda_n,\theta_{n}) \right\Vert^2,$$ which in turn gives us a bound on $$ \mathbb{E}[\left\Vert g_\theta(\lambda_N, \theta_N) \right\Vert^2] + \mathbb{E}[\left\Vert g_\lambda(\lambda_N,\theta_{N+1}) \right\Vert^2].$$

Bounding the above two terms will require solutions of the Poisson equations for $$E_\lambda(W_n; \lambda_n,\theta_{n+1})$$ and $$E_\theta(W_n; \lambda_n,\theta_{n}).$$

## What are the corresponding Poisson equations?

We study the first Poisson equation

$$L_{\lambda} \hat{E}_{\theta} (w_0; \lambda,\theta) = E_{\theta}(w_0; \lambda,\theta)$$

where $$w_0 = (z_1, x_1, z_0, x_0),$$

$$L_{\lambda} \hat{E}_{\theta} (w_0;\lambda,\theta) = Q_{\lambda} \hat{E}_{\theta} (w_0;\lambda,\theta)- \hat{E}_{\theta} (w_0;\lambda,\theta).$$

The solution of the Poisson equation is given by

$$\hat{E}_{\theta} (w_0;\lambda,\theta) = - \displaystyle \sum_{t=0}^\infty Q_{\lambda}^t E_{\theta}(w_0;\lambda,\theta).$$

Now, let us write

$$ \displaystyle
E_{\theta}(w_0; \lambda,\theta) = -  \frac{\partial}{\partial\theta} \left( H_{\bar{Q}_\lambda \Vert P_\theta}(Z_1, X_1 \vert Z_0, X_0) - \log \frac{Q_\lambda(z_1,x_1\vert z_0, x_0)}{P_\theta(z_1,x_1 \vert z_0,x_0)} \right).
$$

It follows that for $$t \geq 1,$$

$$\begin{array}{rl} & 
\displaystyle Q_\lambda^t E_{\theta}(w_0;\lambda,\theta) 
\\ & \\ & = 
\displaystyle -\frac{\partial}{\partial\theta} \Bigg( H_{\bar{Q}_\lambda \Vert P_\theta}(Z_1, X_1 \vert Z_0, X_0) 
\\ & \\ & \quad
\displaystyle - \int \hat{Q}_\lambda(dZ_0,dX_0)\hat{Q}_\lambda^t(dZ_{t}, dX_{t} \vert Z_0, X_0) \log \frac{\hat{Q}_\lambda(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})}{P_\theta(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})} \Bigg) 
\\ & \\ & = 
\displaystyle - \frac{\partial}{\partial\theta} \left( H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})  - H_{\hat{Q}_\lambda \Vert P_\theta}(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})  \right) \end{array}$$

where $$\hat{Q}_\lambda$$ is the distribution of the Markov chain that initializes $$(Z_0,X_0)$$ with the state $$(z_1, x_1)$$ and has transition probabilities $$Q_\lambda.$$ Therefore,

$$\begin{array}{rl} & 
\displaystyle
Q_{\lambda} \hat{E}_{\theta} (w_0;\lambda,\theta)
\\ & \\ & = 
\displaystyle
- \sum_{t=1}^{\infty} Q_\lambda^{t} E_{\theta}(w_0;\lambda,\theta)  
\\ & \\ & = 
\displaystyle \lim_{T \rightarrow \infty} \frac{\partial}{\partial\theta} \Big( \sum_{t=1}^{T} H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1})
\\ & \\ & \quad
 \displaystyle - \sum_{t=1}^{T}  H_{\hat{Q}_\lambda \Vert P_\theta}(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1}) \Big) 
 \\ & \\ & = 
 \displaystyle \lim_{T \rightarrow \infty} \frac{\partial}{\partial\theta} \Big( H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{T}, X_{T} \vert Z_{0}, X_{0}) -  H_{\hat{Q}_\lambda \Vert P_\theta}(Z_{T}, X_{T} \vert Z_{0}, X_{0}) \Big) 
\end{array}$$

where the last equality follows from the chain rule for conditional relative entropy. 

As for the second Poisson equation

$$L_{\lambda} \hat{E}_{\lambda} (w_0; \lambda,\theta) = E_{\lambda}(w_0; \lambda,\theta)$$

its solution is

$$\hat{E}_{\lambda} (w_0;\lambda,\theta) = - \displaystyle \sum_{t=0}^\infty Q_{\lambda}^t E_{\lambda}(w_0;\lambda,\theta).$$

Bringing them all together,

$$\begin{array}{rl}
V(\lambda,\theta) &= H_{\bar{Q}_\lambda \Vert P_\theta}(Z_1, X_1 \vert Z_0, X_0) 
\\ & \\ 
g_\theta(\lambda, \theta) & = 
\displaystyle \frac{\partial}{\partial \theta} V(\lambda, \theta) 
\\ & \\ 
E_{\theta} (w_0;\lambda,\theta) & = 
\displaystyle - \frac{\partial}{\partial\theta} \Big( V(\lambda,\theta)+ \log P_\theta(z_1, x_1 \vert z_0, x_0) \Big) 
\\ & \\ 
Q_{\lambda} \hat{E}_{\theta} (w_0;\lambda,\theta) & =
\displaystyle \lim_{T \rightarrow \infty} \frac{\partial}{\partial\theta} \Big( H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{T}, X_{T} \vert Z_{0}, X_{0})  
\\ & \\ & \qquad \qquad 
\displaystyle -  H_{\hat{Q}_\lambda \Vert P_\theta}(Z_{T}, X_{T} \vert Z_{0}, X_{0}) \Big)
\\ & \\ 
\hat{E}_{\theta} (w_0;\lambda,\theta) & = 
Q_{\lambda} \hat{E}_{\theta} (w_0;\lambda,\theta) - E_{\theta} (w_0;\lambda,\theta) 
\end{array}$$

We impose the following regularity conditions.

----

**C1 (Stationarity).** For all $$Q \in \Delta_\mathcal{M}, \theta \in \Theta,$$ the Markov kernel $$\mathcal{P}_{F(Q,\theta)}$$ has a unique stationary distribution $$\bar{\pi}_{F(Q,\theta)}$$

**C2 (Exploitation and Exploration).** For all $$Q \in \Delta_\mathcal{M}, \theta \in \Theta,$$

$$V(F(Q, \theta),\theta) \leq V(Q,\theta).$$

**C3 ($$\ell$$-smoothness).** There exists $$\ell < \infty$$ such that for all $$Q \in \Delta_\mathcal{M}, \theta, \theta' \in \Theta,$$

$$\displaystyle \left\Vert \frac{\partial V}{\partial \theta}(Q,\theta) - \frac{\partial V}{\partial \theta}(Q,\theta') \right\Vert \leq \ell \Vert \eta - \eta' \Vert.$$

**C4 (Regularity of solution of Poisson equation).** There exists $$\ell_0, \ell_1 < \infty$$ such that for all $$Q \in \Delta_\mathcal{M}, \theta, \theta' \in \Theta, w \in \mathcal{W},$$

$$\Vert \hat{E}_{Q, \theta} (w) \Vert \leq \ell_0, \quad \Vert \mathcal{P}_{Q,\eta} \hat{E}_{Q,\theta}(w) \Vert \leq \ell_0,$$

$$\Vert \mathcal{P}_{Q,\theta} \hat{E}_{Q,\theta}(w) - \mathcal{P}_{Q,\theta'} \hat{E}_{Q,\theta'} (w) \Vert \leq \ell_1 \Vert \theta - \theta' \Vert.$$

**C5 (Boundedness of correction term).** There exists $$\sigma < \infty$$ such that for all $$Q \in \Delta_\mathcal{M}, \theta \in \Theta, w \in \mathcal{W},$$

$$\Vert E_{Q,\theta} (w) \Vert \leq \sigma.$$

----

**Theorem (Convergence of Biased Stochastic Approximation).** Suppose that we have state updates

$$(Y_{k+1}, U_{k+1}) \sim Q_*(Y_{k+1}, U_{k+1} \vert Y_k, U_k),$$

$$Z_{k+1} \sim Q_k(Z_{k+1} \vert Z_k, U_k),$$

and parameter updates

$$\begin{array}{rl} \displaystyle \theta_{k+1} &= \displaystyle \theta_k + \eta_{k+1} \left. \frac{d}{d\theta} \log P_\theta(Z_{k+1}, U_{k+1} \vert Z_k, U_k) \right\vert _{\theta = \theta_k}, \end{array}$$

$$Q_{k+1} = F(Q_k, \theta_{k+1}).$$

for $$0 \leq k \leq n,$$ using step sizes $$\eta_k = \eta_0 k^{-1/2}$$ for sufficiently small $$\eta_0 \geq 0,$$ and using a random stop time $$0 \leq N \leq n$$ with $$\mathbb{P}(N = l) := (\sum_{k=0}^n \eta_{k+1})^{-1} \eta_{l+1}.$$ Then assuming C1-C5, we have

$$\mathbb{E}(\Vert g(Q_N, \theta_N) \Vert^2) = O(\log n / \sqrt{n} ).$$

----

## References

[BB01] Baxter, Jonathan, and Peter L. Bartlett. "Infinite-horizon policy-gradient estimation." _Journal of Artificial Intelligence Research_ 15 (2001): 319-350.

[JG14] Jimenez Rezende, Danilo, and Wulfram Gerstner. "Stochastic variational learning in recurrent spiking networks." _Frontiers in computational neuroscience_ 8 (2014): 38.

[KMMW19] Karimi, Belhal, Blazej Miasojedow, Éric Moulines, and Hoi-To Wai. "Non-asymptotic analysis of biased stochastic approximation scheme." _arXiv preprint arXiv:1902.00629_ (2019).

[L92] Leroux, Brian G. "Maximum-likelihood estimation for hidden Markov models." _Stochastic processes and their applications_ 40, no. 1 (1992): 127-143.

[S01] Sato, Masa-Aki. "Online model selection based on the variational Bayes." _Neural computation_ 13, no. 7 (2001): 1649-1681.

