---
layout: post
title: Convergence of biased stochastic approximation
excerpt_separator: <!--more-->
---

Using techniques from [biased](https://shaoweilin.github.io/biased-stochastic-approximation/) stochastic approximation [KMMW19], we prove under some regularity conditions the convergence of the online learning algorithm proposed [previously](https://shaoweilin.github.io/biased-stochastic-approximation-for-latent-processes/) for latent Markov processes. 

Recall that the algorithm is described by the following updates.

$$\displaystyle \theta_{n+1} = \theta_n + \eta_{n+1} \left.\frac{d}{d\theta} \log P_\theta(Z_{n}, X_{n} \vert Z_{n-1},X_{n-1}) \right\vert _{\theta = \theta_n}$$

$$\displaystyle \lambda_{n+1} = \lambda_n - \eta_{n+1} \alpha_{n} \left( \log \frac{Q_{\lambda_n}(Z_{n}\vert Z_{n-1},X_{n-1})}{P_{\theta_{n+1}}(Z_{n},X_{n}\vert Z_{n-1},X_{n-1})} -\hat{\xi} \right)$$ 

$$\displaystyle X_{n+1} \sim Q_*(X_{n+1} \vert X_{n})$$

$$\displaystyle Z_{n+1} \sim Q_{\lambda_{n+1}}(Z_{n+1} \vert Z_{n}, X_{n})$$

$$\displaystyle \alpha_{n+1} = \gamma \alpha_n + \left.\frac{d}{d\lambda} \log Q_{\lambda}(Z_{n+1} \vert  Z_{n},X_{n})\right\vert _{\lambda=\lambda_{n+1}}$$

This post is a continuation from our [series](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

<!--more-->

## How do we frame the problem in the language of [KMMW19]?

To prove the convergence of our [biased](https://shaoweilin.github.io/biased-stochastic-approximation/) stochastic approximation, we cannot apply the standard unbiased stochastic approximation theory of Robbins and Monro. We can however apply the work of [KMMW19] which gives some guarantees for biased stochastic approximation involving Markov updates. In this section, we will now derive sufficient conditions for the [convergence](https://shaoweilin.github.io/biased-stochastic-approximation/#theorem-convergence-of-biased-stochastic-approximation) of our biased stochastic approximation.

We [recall](https://shaoweilin.github.io/biased-stochastic-approximation-for-latent-processes/#what-do-we-assume-about-the-true-distribution-the-model-and-the-learning-objective) some key assumptions about our disciminative model $$\{Q_\lambda:\lambda\in\Lambda\},$$ generative model $$\{P_\theta:\theta \in \Theta\}$$ and true distribution $$Q_*$$

----

**C1 (Markov property and stationarity).** Under the true distribution $$Q_*,$$ the process $$\{X_t\}$$ is Markov. Under each $$Q_\lambda$$ and each $$P_\theta,$$ the process $$\{(Z_t, X_t)\}$$ is Markov, and $$Z_t$$ and $$X_t$$ are independent given their past. The true distribution $$Q_*$$ has a stationary distribution $$\bar{\pi}_*,$$ and each discriminative model distribution $$Q_\lambda$$ has a stationary distribution $$\bar{\pi}_\lambda.$$

----


First, let $$W_n \in \mathcal{W}$$ denote 

$$W_n = (W_{n1}, W_{n2}, W_{n3}, W_{n4}, W_{n5}) := (Z_n,X_n,Z_{n-1},X_{n-1}, \alpha_n).$$

Then $$W_n$$ is a $$\lambda$$-controlled Markov process. Abusing notation, we write the distribution of $$W_n$$ as

$$ \begin{array}{rl} & \displaystyle 
Q_\lambda(W_{n+1} \vert W_n) 
\\ & \\ & = \displaystyle
Q_\lambda\left((\,Z_{n+1},X_{n+1},Z_n,X_n,\alpha_{n+1})\, \vert\, (Z_n,X_n,Z_{n-1},X_{n-1},\alpha_n) \, \right)
\\ & \\ & = \displaystyle
Q_*(X_{n+1}\vert X_n) Q_\lambda(Z_{n+1}\vert Z_n, X_n)
\end{array}
$$

where we require

$$\alpha_{n+1} = \gamma \alpha_n + \frac{d}{d\lambda} \log Q_{\lambda}(Z_{n+1} \vert  Z_{n},X_{n}).$$

Our Lyapunov function is the learning objective

$$ \displaystyle
V(\lambda,\theta) := H_{\bar{Q}_\lambda \Vert P_\theta}(Z_1, X_1 \vert Z_0, X_0).
$$

We assume some standard regularity condition on $$V(\lambda,\theta).$$

----
**C2 ($$\ell$$-smoothness).** There exists $$\ell < \infty$$ such that for all $$\lambda,\lambda' \in \Lambda$$ and $$\theta, \theta' \in \Theta,$$

$$\displaystyle \left\Vert \frac{\partial V}{\partial \theta}(\lambda,\theta) - \frac{\partial V}{\partial \theta}(\lambda,\theta') \right\Vert \leq \ell \Vert \theta - \theta' \Vert,$$

$$\displaystyle \left\Vert \frac{\partial V}{\partial \lambda}(\lambda,\theta) - \frac{\partial V}{\partial \lambda}(\lambda',\theta) \right\Vert \leq \ell \Vert \lambda - \lambda' \Vert.$$

----

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

where we have

$$\begin{array}{rl} 
\displaystyle G_\theta(W_n;\lambda_n,\theta_n) &= 
\displaystyle - \left. \frac{d}{d\theta} \log P_\theta(Z_{n}, X_{n} \vert Z_{n-1},X_{n-1}) \right\vert _{\theta = \theta_n}
\\ & \\
\displaystyle G_\lambda(W_n;\lambda_n,\theta_{n+1}) &= 
\displaystyle \alpha_{n}\left( \log \frac{Q_{\lambda_n}(Z_{n}\vert Z_{n-1},X_{n-1})}{P_{\theta_{n+1}}(Z_{n},X_{n}\vert Z_{n-1},X_{n-1})} -\hat{\xi} \right)
. \end{array}$$

The mean fields of the updates are given by

$$\begin{array}{rl} 
\displaystyle g_\theta(\lambda_n, \theta_n) & =
\displaystyle - \lim_{T\rightarrow \infty} \mathbb{E}_{Q_{\lambda_n}} \left[ \left.\frac{d}{d\theta} \log P_\theta(Z_{T+1}, X_{T+1} \vert Z_{T},X_{T})\right\vert _{\theta = \theta_n} \right]
\\ & \\ & =
\displaystyle \frac{\partial V}{\partial\theta}(\lambda_n,\theta_n) 
\\ & \\
\displaystyle g_\lambda(\lambda_n,\theta_{n+1}) & =
\displaystyle
\lim_{T\rightarrow \infty} \mathbb{E}_{Q_{\lambda_n}} \Bigg[ \left( \log \frac{Q_{\lambda_n}(Z_{T+1}\vert Z_{T},X_{T})}{P_{\theta_{n+1}}(Z_{T+1},X_{T+1}\vert Z_{T},X_{T})} -\hat{\xi}\right) \,\,\times 
\\ & \\ & \quad\quad \displaystyle \sum_{t=0}^{T} \gamma^{T-t} \left. \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \right\vert_{\lambda=\lambda_n} \Bigg]
\\ & \\ & \approx
\displaystyle \frac{\partial V}{\partial\lambda}(\lambda_n,\theta_{n+1}) 
. \end{array}$$

Because the mean field $$g_\lambda(\lambda,\theta)$$ is not exactly equal to the gradient of Lyapunov function $$V(\lambda,\theta),$$ we need some assumptions to align its behavior with the Lyapunov gradient.

----

**<a id="assumption-direction-of-mean-field"></a>C3 (Direction of mean field).** There exists $$c_0, c_1 \geq 0$$ such that for all $$\lambda \in \Lambda$$ and $$\theta \in \Theta,$$

$$\displaystyle c_0 + c_1 \left\langle \frac{\partial V}{\partial \lambda}(\lambda,\theta) , g_\lambda(\lambda,\theta) \right\rangle \geq \Vert g_\lambda(\lambda,\theta) \Vert^2.$$

**C4 (Length of mean field).** There exists $$d_0, d_1 \geq 0$$ such that for all $$\lambda \in \Lambda$$ and $$\theta \in \Theta,$$

$$\displaystyle d_0 + d_1 \Vert g_\lambda(\lambda,\theta) \Vert \geq \left\Vert \frac{\partial V}{\partial \lambda}(\lambda,\theta) \right\Vert.$$

----

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

First, we make some common assumptions about the step sizes and the stop time for the stochastic approximation.

----

**C5 (Step sizes and stop time).** For all $$k \geq 1,$$ the step sizes are given by

$$\eta_k = \eta_1 k^{-1/2}$$

for some fixed $$\eta_1 > 0.$$ Given a fixed time limit $$T > 0,$$ the stochastic approximation stops randomly after $$N \in \{0, \ldots, T\}$$ steps according to the probabilities

$$\displaystyle \mathbb{P}[N=k] = \eta_{k+1} \left(\sum_{j=0}^T \eta_{j+1} \right)^{-1}.$$

----

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
\displaystyle -\eta_{n+1}\left\langle \frac{\partial V}{\partial\lambda}(\lambda_n,\theta_{n+1}) , G_\lambda(W_n; \lambda_n,\theta_{n+1}) \right\rangle
\\ & \\ & \quad
\displaystyle -\eta_{n+1}\left\langle \frac{\partial V}{\partial\theta}(\lambda_n,\theta_n) , G_\theta(W_n; \lambda_n,\theta_{n}) \right\rangle
\\ & \\ & \quad
\displaystyle + \frac{\ell}{2} \eta_{n+1}^2 \left( \left\Vert G_\lambda(W_n; \lambda_n,\theta_{n+1}) \right\Vert^2 + \left\Vert G_\theta(W_n; \lambda_n,\theta_n) \right\Vert^2 \right)
.\end{array}
$$

We now substitute the mean fields and correction terms. After some rearranging and summing from $$n=0$$ to $$n=T$$, we have

$$\begin{array}{rl} &
\displaystyle
\sum_{n=0}^T \eta_{n+1}\left\langle \frac{\partial V}{\partial\lambda}(\lambda_n,\theta_{n+1}) , g_\lambda(\lambda_n,\theta_{n+1}) \right\rangle
\\ & \\ & +
\displaystyle
\sum_{n=0}^T \eta_{n+1}\left\langle \frac{\partial V}{\partial\theta}(\lambda_n,\theta_n) , g_\theta(\lambda_n,\theta_{n}) \right\rangle 
\\ & \\ & \leq  
V(\lambda_{0},\eta_{0}) - V(\lambda_{T+1},\theta_{T+1})
\\ & \\ & \quad
\displaystyle -\eta_{n+1}\left\langle \frac{\partial V}{\partial\lambda}(\lambda_n,\theta_{n+1}) , \sum_{n=0}^T E_\lambda(W_n; \lambda_n,\theta_{n+1}) \right\rangle
\\ & \\ & \quad
\displaystyle -\eta_{n+1}\left\langle \frac{\partial V}{\partial\theta}(\lambda_n,\theta_n) , \sum_{n=0}^T E_\theta(W_n; \lambda_n,\theta_{n}) \right\rangle
\\ & \\ & \quad
\displaystyle + \frac{\ell}{2} \eta_{n+1}^2 \sum_{n=0}^T \left( \left\Vert g_\lambda(\lambda_n,\theta_{n+1}) \right\Vert^2 + \left\Vert E_\lambda(W_n; \lambda_n,\theta_{n+1}) \right\Vert^2 \right)
\\ & \\ & \quad
\displaystyle + \frac{\ell}{2} \eta_{n+1}^2 \sum_{n=0}^T \left( \left\Vert g_\theta(\lambda_n,\theta_{n}) \right\Vert^2 + \left\Vert E_\theta(W_n; \lambda_n,\theta_n) \right\Vert^2 \right)
.\end{array}
$$

The left-hand-side of the inequality can be bounded below by some affine function of $$\left\Vert g_\lambda(\lambda_n,\theta_{n+1}) \right\Vert^2$$ and $$ \left\Vert g_\theta(\lambda_n,\theta_{n}) \right\Vert^2.$$ On the right-hand-side of the inequality, we may assume that the Lyapunov difference $$V(\lambda_{0},\eta_{0}) - V(\lambda_{T+1},\theta_{T+1})$$ and the squared corrections $$\left\Vert E_\lambda(W_n; \lambda_n,\theta_{n+1}) \right\Vert^2$$ and $$\left\Vert E_\theta(W_n; \lambda_n,\theta_n) \right\Vert^2$$ are bounded above by constants. 

----

**C6 (Correction bound).** There exists $$\sigma < \infty$$ such that for all $$\lambda \in \Lambda, \theta \in \Theta, w \in \mathcal{W},$$

$$\Vert E_\lambda(w; \lambda,\theta) \Vert \leq \sigma,$$

$$\Vert E_\theta(w; \lambda,\theta) \Vert \leq \sigma.$$

----

Overall, if we also have control over the terms

$$
\displaystyle \left\langle \frac{\partial V}{\partial\lambda}(\lambda_n,\theta_{n+1}) , \sum_{n=0}^T E_\lambda(W_n; \lambda_n,\theta_{n+1}) \right\rangle,
$$

$$
\displaystyle \left\langle \frac{\partial V}{\partial\theta}(\lambda_n,\theta_n) , \sum_{n=0}^T E_\theta(W_n; \lambda_n,\theta_{n}) \right\rangle,
$$

then we can bound $$\sum_{n=0}^T \eta_{n+1}\left\Vert g_\lambda(\lambda_n,\theta_{n+1}) \right\Vert^2$$ and $$\sum_{n=0}^T \eta_{n+1} \left\Vert g_\theta(\lambda_n,\theta_{n}) \right\Vert^2,$$ which in turn gives us a bound on $$ \mathbb{E}[\left\Vert g_\theta(\lambda_N, \theta_N) \right\Vert^2] + \mathbb{E}[\left\Vert g_\lambda(\lambda_N,\theta_{N+1}) \right\Vert^2].$$

Bounding the above two terms will require solutions of the Poisson equations for $$E_\lambda(W_n; \lambda_n,\theta_{n+1})$$ and $$E_\theta(W_n; \lambda_n,\theta_{n}).$$

## What are the corresponding Poisson equations?

We study the first Poisson equation

$$L_{\lambda} \hat{E}_{\theta} (w_0; \lambda,\theta) = E_{\theta}(w_0; \lambda,\theta)$$

where $$w_0 = (z_1, x_1, z_0, x_0),$$

$$L_{\lambda} \hat{E}_{\theta} (w_0;\lambda,\theta) = Q_{\lambda} \hat{E}_{\theta} (w_0;\lambda,\theta)- \hat{E}_{\theta} (w_0;\lambda,\theta).$$

A candidate solution (if it is well-defined) of the Poisson equation is

$$\hat{E}_{\theta} (w_0;\lambda,\theta) = - \displaystyle \sum_{t=0}^\infty Q_{\lambda}^t E_{\theta}(w_0;\lambda,\theta)$$

where $$Q^t_\lambda$$ denotes $$t$$ applications of the Markov kernel $$Q_\lambda.$$

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
 - \sum_{t=1}^{T}  H_{\hat{Q}_\lambda \Vert P_\theta}(Z_{t}, X_{t} \vert Z_{t-1}, X_{t-1}) \Big) 
 \\ & \\ & = 
 \displaystyle \lim_{T \rightarrow \infty} \frac{\partial}{\partial\theta} \Big( H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{T}, X_{T} \vert Z_{0}, X_{0}) -  H_{\hat{Q}_\lambda \Vert P_\theta}(Z_{T}, X_{T} \vert Z_{0}, X_{0}) \Big) 
\end{array}$$

where the last equality follows from the chain rule for conditional relative entropy. 

Bringing them all together,

$$\begin{array}{rl}
V(\lambda,\theta) &= H_{\bar{Q}_\lambda \Vert P_\theta}(Z_1, X_1 \vert Z_0, X_0) 
\\ & \\ 
g_\theta(\lambda, \theta) & = 
\displaystyle \frac{\partial V}{\partial \theta}(\lambda, \theta) 
\\ & \\ 
E_{\theta} (w;\lambda,\theta) & = 
\displaystyle - \frac{\partial}{\partial\theta} \Big( V(\lambda,\theta)+ \log P_\theta(z_1, x_1 \vert z_0, x_0) \Big) 
\\ & \\ 
Q_{\lambda} \hat{E}_{\theta} (w;\lambda,\theta) & =
\displaystyle \lim_{T \rightarrow \infty} \frac{\partial}{\partial\theta} \Big( H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{T}, X_{T} \vert Z_{0}, X_{0})  
-  H_{\hat{Q}_\lambda \Vert P_\theta}(Z_{T}, X_{T} \vert Z_{0}, X_{0}) \Big)
\\ & \\ 
\hat{E}_{\theta} (w;\lambda,\theta) & = 
Q_{\lambda} \hat{E}_{\theta} (w;\lambda,\theta) - E_{\theta} (w;\lambda,\theta) 
\end{array}$$

As for the second Poisson equation

$$L_{\lambda} \hat{E}_{\lambda} (w_0; \lambda,\theta) = E_{\lambda}(w_0; \lambda,\theta)$$

a candidate solution is

$$\hat{E}_{\lambda} (w_0;\lambda,\theta) = - \displaystyle \sum_{t=0}^\infty Q_{\lambda}^t E_{\lambda}(w_0;\lambda,\theta).$$

**TODO**

We impose the following regularity conditions.

----
**C7 (Solution of Poisson equation).** The functions $$\hat{E}_\lambda : \mathcal{W} \times \Lambda \times \Theta \rightarrow \Lambda$$ and $$\hat{E}_\theta : \mathcal{W} \times \Lambda \times \Theta \rightarrow \Theta$$ given by

$$\hat{E}_{\lambda} (w;\lambda,\theta) = - \displaystyle \lim_{T\rightarrow \infty} \sum_{t=0}^T Q_{\lambda}^t E_{\lambda}(w;\lambda,\theta),$$

$$\hat{E}_{\theta} (w;\lambda,\theta) = - \displaystyle \lim_{T\rightarrow \infty} \sum_{t=0}^T Q_{\lambda}^t E_{\theta}(w;\lambda,\theta)$$

are measurable and well-defined.

**C8 (Regularity of solution).** There exists $$\ell_0, \ell_1 < \infty$$ such that for all $$\lambda,\lambda' \in \Lambda$$ and $$\theta, \theta' \in \Theta$$ and $$w \in \mathcal{W},$$

$$\Vert \hat{E}_{\lambda} (w;\lambda,\theta) \Vert \leq \ell_0, \quad \Vert Q_\lambda \hat{E}_{\lambda} (w;\lambda,\theta)  \Vert \leq \ell_0,$$

$$\Vert \hat{E}_{\theta} (w;\lambda,\theta) \Vert \leq \ell_0, \quad \Vert Q_\lambda \hat{E}_{\theta} (w;\lambda,\theta)  \Vert \leq \ell_0,$$

$$\Vert Q_\lambda \hat{E}_{\lambda} (w;\lambda,\theta) -Q_{\lambda'} \hat{E}_{\lambda} (w;\lambda',\theta)  \Vert \leq \ell_1 \Vert \lambda - \lambda' \Vert,$$

$$\Vert Q_\lambda \hat{E}_{\lambda} (w;\lambda,\theta) -Q_{\lambda} \hat{E}_{\lambda} (w;\lambda,\theta')  \Vert \leq \ell_1 \Vert \theta - \theta' \Vert.$$

----

**Theorem (Convergence of online learning).** Suppose that we have stochastic approximation

$$\displaystyle \theta_{n+1} = \theta_n + \eta_{n+1} \left.\frac{d}{d\theta} \log P_\theta(Z_{n}, X_{n} \vert Z_{n-1},X_{n-1}) \right\vert _{\theta = \theta_n}$$

$$\displaystyle \lambda_{n+1} = \lambda_n - \eta_{n+1} \alpha_{n} \left( \log \frac{Q_{\lambda_n}(Z_{n}\vert Z_{n-1},X_{n-1})}{P_{\theta_{n+1}}(Z_{n},X_{n}\vert Z_{n-1},X_{n-1})} -\hat{\xi} \right)$$ 

$$\displaystyle X_{n+1} \sim Q_*(X_{n+1} \vert X_{n})$$

$$\displaystyle Z_{n+1} \sim Q_{\lambda_{n+1}}(Z_{n+1} \vert Z_{n}, X_{n})$$

$$\displaystyle \alpha_{n+1} = \gamma \alpha_n + \left.\frac{d}{d\lambda} \log Q_{\lambda}(Z_{n+1} \vert  Z_{n},X_{n})\right\vert _{\lambda=\lambda_{n+1}}.$$

Then assuming C1-C8 and sufficiently small initial step size $$\eta_1,$$
we have

$$\mathbb{E}\left[\left\Vert g_\theta(\lambda_N, \theta_N) \right\Vert^2\right] + \mathbb{E}\left[\left\Vert g_\lambda(\lambda_N,\theta_{N+1}) \right\Vert^2\right] = O(c_0 + \log T / \sqrt{T} )$$

where $$c_0$$ was defined in assumption [C3](#assumption-direction-of-mean-field).

----

## References

[BB01] Baxter, Jonathan, and Peter L. Bartlett. "Infinite-horizon policy-gradient estimation." _Journal of Artificial Intelligence Research_ 15 (2001): 319-350.

[JG14] Jimenez Rezende, Danilo, and Wulfram Gerstner. "Stochastic variational learning in recurrent spiking networks." _Frontiers in computational neuroscience_ 8 (2014): 38.

[KMMW19] Karimi, Belhal, Blazej Miasojedow, Éric Moulines, and Hoi-To Wai. "Non-asymptotic analysis of biased stochastic approximation scheme." _arXiv preprint arXiv:1902.00629_ (2019).

[L92] Leroux, Brian G. "Maximum-likelihood estimation for hidden Markov models." _Stochastic processes and their applications_ 40, no. 1 (1992): 127-143.

[S01] Sato, Masa-Aki. "Online model selection based on the variational Bayes." _Neural computation_ 13, no. 7 (2001): 1649-1681.

