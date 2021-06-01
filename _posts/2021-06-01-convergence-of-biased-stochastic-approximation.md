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
\displaystyle \frac{d}{d\theta}V(\lambda_n,\theta_n) 
\\ & \\
\displaystyle g_\lambda(\lambda_n,\theta_{n+1}) & =
\displaystyle \sum_{t=0}^\infty \sum_{w',w} \bar{\pi}_{\lambda_n}(w) Q_{\lambda_n}^t(w'\vert w) \,\,\times
\\ & \\ & \quad \quad
\displaystyle \gamma^t \left( \log \frac{Q_{\lambda_n}(w'_1 \vert w'_3,w'_4)}{P_{\theta_{n+1}}(w'_1 \vert w'_3,w'_4)} - \log P_{\theta_{n+1}}(w'_2 \vert w'_3, w'_4) -\xi \right) \,\,\times 
\\ & \\ & \quad \quad
\displaystyle
\left.\frac{d}{d\lambda} \log Q_\lambda(w_1 \vert  w_3,w_4) \right\vert_{\lambda=\lambda_n}
\\ & \\ & \approx
\displaystyle \frac{d}{d\lambda}V(\lambda_n,\theta_{n+1}) 
\end{array}$$

where $$Q^t_\lambda$$ denotes the transition probabilities after $$t$$ steps of the Markov chain with distribution $$Q_\lambda.$$

## What are the Poisson equations here?

**TODO**

Therefore, the mean field satisfies

$$g(Q_n, \theta_n) = \displaystyle \frac{\partial V}{\partial \theta}(Q_n, \theta_n),$$

so assumptions A1 and A2 of the [convergence theorem](https://shaoweilin.github.io/biased-stochastic-approximation/) are automatically satisfied.


We now study the Poisson equation

$$L_{Q,\theta} \hat{E}_{Q, \theta} (w_0) = E_{Q,\theta}(w_0)$$

where $$w_0 = (z_1, x_1, z_0, x_0),$$

$$\begin{array}{rl} E_{Q,\theta}(w_0) & = \displaystyle - g(Q,\theta) - \frac{d}{d\theta}\log P_\theta(z_1, x_1 \vert z_0, x_0) \\ & \\ & = - \displaystyle  \frac{d}{d\theta} \Big( V(Q,\theta)+ \log P_\theta(z_1, x_1 \vert z_0, x_0) \Big) , \end{array} $$

$$L_{Q,\theta} \hat{E}_{Q, \theta} (w_0) = \mathcal{P}_{Q,\theta} \hat{E}_{Q, \theta} (w_0)- \hat{E}_{Q, \theta} (w_0).$$

The solution of the Poisson equation is given by

$$\hat{E}_{Q, \theta} (w_0) = - \displaystyle \lim_{n \rightarrow \infty} \sum_{k=0}^n \mathcal{P}_{Q,\theta}^k E_{Q,\theta}(w_0).$$

We observe that

$$\begin{array}{rl} & \mathcal{P}_{Q,\theta} E_{Q,\theta}(w_0) \\ & \\ & = \displaystyle -\frac{d}{d\theta} \Bigg( V(Q,\theta) - \int Q(dZ_2, dX_2 \vert z_1, x_1) \log \frac{Q(Z_{2}, X_{2} \vert z_1, x_1)}{P_\theta(Z_{2}, X_{2} \vert z_1, x_1)} \Bigg) \\ & \\ & = \displaystyle - \frac{d}{d\theta} \Big( H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_1, X_1 \vert Z_0, X_0) \\ & \\ & \qquad \qquad - H_{\mathcal{M}(\mathcal{P}_{Q},\pi_{z_1, x_1}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_1, X_1 \vert Z_0, X_0) \Big) \end{array}$$

where $$\pi_{z_1, x_1}$$ is the initial distribution with $$Z_1 = z_1, X_1 = x_1$$ almost surely. Then,

$$\begin{array}{rl} & \mathcal{P}_{Q,\theta} \hat{E}_{Q, \theta} (w_0) \\ & \\ & = E_{Q,\theta}(w_0) + \hat{E}_{Q, \theta} (w_0) \\ & \\ & = - \displaystyle \lim_{n \rightarrow \infty} \sum_{k=0}^{n-1} \mathcal{P}_{Q,\theta}^{k+1} E_{Q,\theta}(w_0) \\ & \\ & = \displaystyle \lim_{n \rightarrow \infty} \frac{d}{d\theta} \Big( \sum_{k=0}^{n-1} H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{k+1}, X_{k+1} \vert Z_k, X_k) \\ & \\ & \quad \displaystyle - \sum_{k=0}^{n-1} H_{\mathcal{M}(\mathcal{P}_{Q},\pi_{z_1, x_1}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{k+1}, X_{k+1} \vert Z_k, X_k) \Big) \\ & \\ & = \displaystyle \lim_{n \rightarrow \infty} \frac{d}{d\theta} \Big( H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{n}, X_{n} \vert Z_0, X_0) \\ & \\ & \quad \displaystyle - H_{\mathcal{M}(\mathcal{P}_{Q},\pi_{z_1, x_1}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{n}, X_{n} \vert Z_0, X_0) \Big) \end{array}$$

Bringing them all together,

$$\begin{array}{rl} V(Q,\theta) &= H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_1, X_1 \vert Z_0, X_0) \\ & \\ g(Q, \theta) & = \displaystyle \frac{\partial V}{\partial \theta}(Q, \theta) \\ & \\ E_{Q,\theta}(w_0) & = - \displaystyle  \frac{d}{d\theta} \Big( V(Q,\theta)+ \log P_\theta(z_1, x_1 \vert z_0, x_0) \Big) \\ & \\ \mathcal{P}_{Q,\theta} \hat{E}_{Q, \theta} (w_0) & = \displaystyle \lim_{n \rightarrow \infty} \frac{d}{d\theta} \Big( H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{n}, X_{n} \vert Z_0, X_0) \\ & \\ & \quad \displaystyle - H_{\mathcal{M}(\mathcal{P}_{Q},\pi_{z_1, x_1}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{n}, X_{n} \vert Z_0, X_0) \Big) \\ & \\ \hat{E}_{Q, \theta} (w_0) & = \mathcal{P}_{Q,\theta} \hat{E}_{Q, \theta} (w_0) - E_{Q,\theta}(w_0) \end{array}$$

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

