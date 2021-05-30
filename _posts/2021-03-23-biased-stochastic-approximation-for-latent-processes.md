---
layout: post
title: Biased stochastic approximation for latent processes
---

We apply biased stochastic approximation and variational inference to optimize a relative entropy objective for latent Markov processes. Using this technique, we prove under some regularity conditions that the learning algorithm converges to a local minima.

We will be using [biased](https://shaoweilin.github.io/biased-stochastic-approximation/) stochastic approximation [KMMW19] where the stochastic updates are dependent on the past but the conditional expectation of the stochastic updates given the past is not equal to the mean field. These biased stochastic approximation schemes generalize the classical expectation maximization algorithm [KMMW19].

This post is a continuation from our [series](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

## What do we assume about the true distribution, the model and the learning objective?

As [before](https://shaoweilin.github.io/variational-inference-for-latent-processes/), we assume that the universe is a Markov process $$\{X_t\},$$ and let its true distribution be the path measure $$Q_*.$$

Suppose that we have a parametric discriminative model $$\{Q_\lambda : \lambda \in \Lambda\}$$ and a parametric generative model $$\{P_\theta : \theta \in \Theta\}$$ where the distributions $$Q_\lambda$$ and $$P_\theta$$ are path measures on some joint process $$\{(Z_t, X_t)\}.$$ The random variables $$Z_t$$ represent computational states in this discriminative-generative model.

We assume that in both models, the distributions are Markov and each $$Z_t$$ and $$X_t$$ are conditionally independent given their past.  We also assume that marginals $$Q(X_{0\ldots T})$$ of the discriminative model distributions $$Q_\lambda(Z_{0 \ldots T}, X_{0\ldots T})$$ are all equal to the true distribution $$Q_*(X_{0\ldots T}).$$
 
Some parts of universe $$\{X_t\}$$ are observed and other parts are unobserved. We will impose these conditions by putting constraints on the structure of the models $$\{Q_\lambda\}$$ and $$\{P_\theta\}$$, as described in this [article](https://shaoweilin.github.io/variational-inference-for-latent-processes/). 

Our goal is to train the models by minimizing the asymptotic relative entropy rate (continuous time) 

$$\lim_{T\rightarrow \infty} \frac{d}{dT}H_{Q \Vert P}(Z_{0\ldots T}, X_{0\ldots T})$$

or asymptotic conditional relative entropy (discrete time)

$$\lim_{n \rightarrow \infty} H_{Q \Vert P}(Z_{n+1}, X_{n+1} \vert Z_{n}, X_{n}).$$

over $$\{Q_\lambda\}$$ and $$\{P_\theta\}$$. We first explore the problem in discrete time, before discussing the analogous results in continuous time.

We assume that $$Q_\lambda$$ has a stationary distribution $$\bar{\pi}_\lambda,$$ and let $$\bar{Q}_\lambda$$ be the distribution of a Markov chain that has the same transition probabilities as $$Q_\lambda$$ but has the initial distribution $$\bar{\pi}_\lambda.$$ Then,

$$\lim_{n \rightarrow \infty} H_{Q_\lambda \Vert P_\theta}(Z_{n+1}, X_{n+1} \vert Z_{n}, X_{n}) = H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{1}, X_{1} \vert Z_{0}, X_{0}).$$

## What is the general intuition behind online learning for latent processes?

To minimize the conditional relative entropy objective, we adopt an approach similar to the expectation-maximization (EM) or exponential-mixture (em) [algorithm](https://shaoweilin.github.io/machine-learning-with-relative-entropy/). More precisely, we iteratively optimize for the discriminative model distribution $$Q_\lambda$$ and for the generative model distribution $$P_\theta$$ while holding the other constant. 

First, we pick some initial generative model distribution $$P_{\theta_0}$$ and discriminative model distribution $$Q_{\lambda_0}.$$ Then, for $$n = 0, 1, \ldots,$$ we repeat the next two steps.

----

**Step 1 (generative model update).** Fixing the discriminative model distribution $$Q_{\lambda_{n}}(Z_{1} \vert Z_{0}, X_{0}),$$ minimize $$H_{\bar{Q}_{\lambda_{n}}\Vert P_{\theta}}(Z_{1}, X_{1} \vert Z_{0}, X_{0})$$ over generative model distributions $$P_{\theta}$$.

By definition,

$$\begin{array}{rl} & 
H_{\bar{Q}_{\lambda_{n}}\Vert P_{\theta}}(Z_{1}, X_{1} \vert Z_{0}, X_{0})
\\ & \\ & = 
\mathbb{E}_{\bar{Q}_{\lambda_{n}}} [\log Q_{\lambda_{n}}(Z_{1}, X_{1} \vert Z_{0},X_{0})] 
\\ & \\ & 
\quad - \mathbb{E}_{\bar{Q}_{\lambda_{n}}} [\log P_\theta(Z_{1}, X_{1} \vert Z_{0},X_{0})], 
\end{array}$$

where we note that the first term is independent of $$\theta$$.

We update the parameter $$\theta$$ using the gradient

$$\displaystyle \theta_{n+1} = \theta_n + \eta_{n+1} \mathbb{E}_{\bar{Q}_{\lambda_{n}}} \left[\left.\frac{d}{d\theta} \log P_\theta(Z_{1}, X_{1} \vert Z_{0},X_{0})\right\vert _{\theta = \theta_n}\right].$$

where we can also write

$$ \begin{array}{rl} &
\displaystyle \mathbb{E}_{\bar{Q}_{\lambda}} \left[\left.\frac{d}{d\theta} \log P_\theta(Z_{1}, X_{1} \vert Z_{0},X_{0})\right\vert _{\theta = \theta_n}\right]
\\ & \\ & =
\displaystyle
\lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+1)},X_{0..(T+1)})} \left[ \left.\frac{d}{d\theta} \log P_\theta(Z_{T+1}, X_{T+1} \vert Z_{T},X_{T})\right\vert _{\theta = \theta_n} \right]
. \end{array}
$$

**Step 2 (discriminative model update).** Fixing the generative model distribution $$P_{\theta_{n+1}},$$ minimize $$H_{\bar{Q}_\lambda \Vert P_{\theta_{n+1}}}(Z_{1}, X_{1} \vert Z_{0}, X_{0})$$ over discriminative model distributions $$Q_\lambda.$$

Because $$Z_{1}$$ and $$X_{1}$$ are conditionally independent given the past,

$$\begin{array}{rl} & H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{1}, X_{1} \vert Z_{0}, X_{0}) \\ & \\ &= H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{1} \vert Z_{0}, X_{0}) + H_{\bar{Q}_\lambda \Vert P_\theta}(X_{1} \vert Z_{0}, X_{0}). \end{array}$$

The second term is independent of $$\lambda$$ because it depends only on the true distribution $$Q_*$$ of the observables. 

We update the parameter $$\lambda$$ using the gradient

$$\lambda_{n+1} = \displaystyle \lambda_n - \eta_{n+1} \left.\frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert P_{\theta_{n+1}}}(Z_{1} \vert Z_{0}, X_{0})\right\vert _{\lambda = \lambda_n}$$

where, as shown in the [appendix](https://shaoweilin.github.io/biased-stochastic-approximation-for-latent-processes/#appendix-discriminative-model-update), we have

$$\begin{array}{rl} &
\displaystyle \frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{1} \vert Z_{0}, X_{0}) 
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+1)},X_{0..(T+1)})} \Bigg[ \left( \log \frac{Q_\lambda(Z_{T+1}\vert Z_{T},X_{T})}{P_\theta(Z_{T+1}\vert Z_{T},X_{T})} \right) \,\,\times 
\\ & \\ & \quad\quad \displaystyle \sum_{t=0}^{T} \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \Bigg]
. \end{array}$$

Following [BB1] and [KMMW19], we approximate this gradient with the following numerically stable estimator

$$\begin{array}{rl} &
\displaystyle \frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{1} \vert Z_{0}, X_{0}) 
\\ & \\ & \approx 
\displaystyle \left( \log \frac{Q_\lambda(Z_{T+1}\vert Z_{T},X_{T})}{P_\theta(Z_{T+1}\vert Z_{T},X_{T})} \right) \sum_{t=0}^{T} \gamma^{T-t} \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t})
. \end{array}$$

where $$0 < \gamma < 1$$ is a discount factor, $$T$$ is sufficiently large and $$Z_{0\ldots (T+1)}, X_{0\ldots (T+1)}$$ is drawn from $$Q_{\lambda}.$$

----

## Is there a stochastic approximation of the above procedure?

The above two-step procedure has the following stochastic approximation. 

$$\displaystyle \theta_{n+1} = \theta_n + \eta_{n+1} \left.\frac{d}{d\theta} \log P_\theta(Z_{n}, X_{n} \vert Z_{n-1},X_{n-1}) \right\vert _{\theta = \theta_n}$$

$$\displaystyle G_{n+1} = \gamma G_n + \left.\frac{d}{d\lambda} \log Q_{\lambda}(Z_{n} \vert  Z_{n-1},X_{n-1})\right|_{\lambda=\lambda_n}$$

$$\displaystyle F_{n+1} = \log \frac{Q_{\lambda_n}(Z_{n}\vert Z_{n-1},X_{n-1})}{P_{\theta_{n+1}}(Z_{n}\vert Z_{n-1},X_{n-1})}$$

$$\displaystyle \lambda_{n+1} = \lambda_n - \eta_{n+1} G_{n+1} F_{n+1}$$ 

$$\displaystyle X_{n+1} \sim Q_*(X_{n+1} \vert X_{n})$$

$$\displaystyle Z_{n+1} \sim Q_{\lambda_{n+1}}(Z_{n+1} \vert Z_{n}, X_{n})$$

In continuous time, the above updates will become differential equations. The samples $$Z_t$$ would be driven by a Poisson process, and the transition probabilities appearing in the updates for $$\theta_t$$, $$G_t$$ and $$F_t$$ would be replaced by transition rates. 

Before we make some preliminary observations about this stochastic approximation, let us introduce some terminology. Given $$(Z_{n-1}, X_{n-1}),$$ suppose we sample $$(Z_n, X_n)$$ from $$Q_\lambda(Z_n,X_n \vert Z_{n-1}, X_{n-1}).$$ The _conditional expectation_ of a function $$r(Z_n, X_n, Z_{n-1}, X_{n-1})$$ is the expectation of $$r$$ conditioned on some given values of $$(Z_{n-1}, X_{n-1}).$$ The _mean field_ or _total expectation_ of $$r$$ is the expectation of its conditional expectation over the stationary distribution $$\bar{\pi}_\lambda$$ on $$(Z_{n-1}, X_{n-1}).$$

In the above stochastic approximation, the mean fields of the updates for $$\theta_{n}$$ and $$\lambda_n$$ are (possibly discounted versions of) the corresponding derivatives of $$H_{\bar{Q}_\lambda \Vert P_{\theta}}(Z_{1}, X_{1} \vert Z_{0}, X_{0}).$$ However, the conditional expectations of the updates depend on $$(Z_{n-1}, X_{n-1})$$ and are not necessarily equal to their mean fields. In this case, we say that the stochastic approximation is _biased_.

In continuous time, the mean fields will be derivatives of relative entropy rates. The conditional expectations which depend on the current states $$(Z_t,X_t)$$ will be biased estimates of the mean fields.

## How do we prove convergence using biased stochastic approximation?

To prove the convergence of our [biased](https://shaoweilin.github.io/biased-stochastic-approximation/) stochastic approximation, we cannot apply the standard unbiased stochastic approximation theory of Robbins and Monro. We can however apply the work of [KMMW19] which gives some guarantees for biased stochastic approximation involving Markov updates. In this section, we will now derive sufficient conditions for the [convergence](https://shaoweilin.github.io/biased-stochastic-approximation/#theorem-convergence-of-biased-stochastic-approximation) of our biased stochastic approximation.

First, we note that $$\{(Z_n,X_n)\}$$ is a $$Q_{\lambda_n}$$-controlled Markov process with

$$(Z_n,X_n) \sim Q_{\lambda_n} (Z_{n}, X_{n} \vert Z_{n-1}, X_{n-1}) = Q_*(X_n \vert X_{n-1}) Q_{\lambda_n}(Z_n \vert Z_{n-1}, X_{n-1}). $$

The mean field of $$G_{Q_n, \theta_n}(W_{n+1})$$ is given by

$$\begin{array}{rl} g(Q_n,\theta_n) & =- \displaystyle \int \bar{\pi}_{Q_n} (dZ_0, dX_0) Q_n(dZ_1, dX_1 \vert Z_0, X_0) \\ & \\ & \quad \displaystyle \left. \frac{d}{d\theta}\log P_\theta(Z_{1}, X_{1} \vert Z_0, X_0) \right|_{\theta = \theta_n} \\ & \\ & = \displaystyle \frac{d}{d\theta} \int \bar{\pi}_{Q_n} (dZ_0, dX_0) Q_n(dZ_1, dX_1 \vert Z_0, X_0) \\ & \\ & \quad \displaystyle \left. \log \frac{Q_n(Z_{1}, X_{1} \vert Z_0, X_0)}{\mathcal{P}_\theta(Z_{1}, X_{1} \vert Z_0, X_0)}\right|_{\theta = \theta_n} \\ & \\ & = \displaystyle \left. \frac{d}{d\theta} H_{\mathcal{M}(\mathcal{P}_{Q_n},\bar{\pi}_{Q_n}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1}, X_{1} \vert Z_0, X_0) \right|_{\theta = \theta_n} \end{array}$$

where $$\mathcal{M}(\mathcal{P},\bar{\pi})$$ denotes the Markov distribution with transition kernel $$\mathcal{P}$$ and initial distribution $$\bar{\pi}$$; and where $$\mathcal{M}(\mathcal{P}, -)$$ is any Markov distribution with transition kernel $$\mathcal{P}$$ if the initial distribution is irrelevant.

We now define the Lyapunov function

$$\begin{array}{rl} \displaystyle V(Q,\theta) & := \displaystyle \lim_{T \rightarrow \infty} \frac{1}{T} H_{Q \Vert P_\theta}(Z_{0\ldots T},X_{0\ldots T}) \\ & \\ & = \displaystyle \lim_{T \rightarrow \infty} H_{Q \Vert P_\theta}(Z_T, X_T \vert Z_{0\ldots (T-1)},X_{0\ldots (T-1)}) \\ & \\ & = \displaystyle \lim_{T \rightarrow \infty} H_{Q \Vert P_\theta}(Z_T, X_T \vert Z_{T-1},X_{T-1}) \\ & \\ & = \displaystyle H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1}, X_{1} \vert Z_0, X_0) \end{array}$$

where $$\mathcal{P}_{Q}$$ and $$\bar{\pi}_{Q}$$ are respectively the transition kernel and the unique stationary distribution of $$Q.$$ Here, the second equality is the asymptotic relationship between relative entropy and relative entropy rate; the third equality follows from the Markov property on $$Q$$ and $$P_\theta$$.

Therefore, the mean field satisfies

$$g(Q_n, \theta_n) = \displaystyle \frac{\partial V}{\partial \theta}(Q_n, \theta_n),$$

so assumptions A1 and A2 of the [convergence theorem](https://shaoweilin.github.io/biased-stochastic-approximation/) are automatically satisfied.

Now, keeping $$\theta$$ fixed, let us minimize $$V(Q,\theta)$$ over $$Q \in \Delta_\mathcal{M}$$ so that we may update $$Q_n$$. In the training algorithm, $$Q_n(Z_{t+1}\vert Z_t, X_t)$$ is the discriminative model distribution that spits out possible explanations $$Z_{t+1}$$ of the observations $$X_t$$ based on prior knowledge $$Z_t.$$

Since

$$\begin{array}{rl} V(Q,\theta) &= H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1}, X_{1} \vert Z_0, X_0) \\ & \\ & = H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1} \vert Z_0, X_0) \\ & \\ & \quad + H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (X_{1} \vert Z_0, X_0) \\ & \\ & = H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1} \vert Z_0, X_0) \\ & \\ & \quad + \displaystyle \int \bar{\pi}_{Q} (dZ_0, dX_0) H_{Q_*(X_1 \vert X_0) \Vert \mathcal{P}_\theta(X_1 \vert Z_0, X_0)} (X_{1}) \\ & \\ & = H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1} \vert Z_0, X_0) \\ & \\ & \quad + \displaystyle \int \bar{\pi}_{Q} (dX_0) \bar{\pi}_{Q} (dZ_0 \vert X_0) H_{Q_*(X_1 \vert X_0) \Vert \mathcal{P}_\theta(X_1 \vert Z_0, X_0)} (X_{1}) , \end{array}$$

where $$\bar{\pi}_Q$$ is the stationary distribution for the true distribution $$Q$$ on $$\{X_t\}$$, we may extinguish the first term by setting

$$Q(Z_1 \vert Z_0, X_0) = P_\theta(Z_1 \vert Z_0, X_0)$$

which was proposed in the previous section. This first term represents the sampler's incentive to _exploit_ the current model distribution $$P_\theta(Z_1 \vert Z_0, X_0)$$ for most likely explanations of the observations.

As for the second term, it decreases if for each $$X_0$$, we assign a larger weight $$\bar{\pi}_{Q}(dZ_0 \vert X_0)$$ to $$Z_0$$ where the relative entropy $$H_{Q_*(X_1 \vert X_0) \Vert \mathcal{P}_\theta(X_1 \vert Z_0, X_0)} (X_{1})$$ is smaller. In other words, the second term shrinks if for each observation $$X_0,$$ the stationary distribution assigns a larger probability to hidden states $$Z_0$$ which can explain the next observation $$X_1$$ under the given model distribution $$P_\theta.$$ This second term represents the sampler's incentive to _explore_ new explanations which fit the observations better as evaluated by $$P_\theta(X_1 \vert Z_0, X_0)$$, even if those new explanations are less probable under $$P_\theta(Z_1 \vert Z_0, X_0)$$.

In the previous section, we chose a sampler $$F(-,\theta)$$ which has a strong tendency to exploit. In practical applications, the design of a good sampler $$F(Q,\theta)$$ may depend on the application, or on some additional mechanism to compare predictions against explanations, or on some kind of curriculum to guide the learning machine so that it may arrive quickly at the most useful explanations.

In this section, we will just assume that the Lyapunov function does not increase under the update $$Q_{n+1} = F(Q_n, \theta_{n+1})$$. The update could for instance compare $$V(Q_n,\theta_{n+1})$$ against $$V(Q_{n+1}, \theta_{n+1})$$ for some proposal $$Q_{n+1},$$ and only accept the proposal if there is an improvement.

We now study the Poisson equation

$$L_{Q,\theta} \hat{E}_{Q, \theta} (w_0) = E_{Q,\theta}(w_0)$$

where $$w_0 = (z_1, x_1, z_0, x_0),$$

$$\begin{array}{rl} E_{Q,\theta}(w_0) & = \displaystyle - g(Q,\theta) - \frac{d}{d\theta}\log P_\theta(z_{1}, x_{1} \vert z_0, x_0) \\ & \\ & = - \displaystyle  \frac{d}{d\theta} \Big( V(Q,\theta)+ \log P_\theta(z_{1}, x_{1} \vert z_0, x_0) \Big) , \end{array} $$

$$L_{Q,\theta} \hat{E}_{Q, \theta} (w_0) = \mathcal{P}_{Q,\theta} \hat{E}_{Q, \theta} (w_0)- \hat{E}_{Q, \theta} (w_0).$$

The solution of the Poisson equation is given by

$$\hat{E}_{Q, \theta} (w_0) = - \displaystyle \lim_{n \rightarrow \infty} \sum_{k=0}^n \mathcal{P}_{Q,\theta}^k E_{Q,\theta}(w_0).$$

We observe that

$$\begin{array}{rl} & \mathcal{P}_{Q,\theta} E_{Q,\theta}(w_0) \\ & \\ & = \displaystyle -\frac{d}{d\theta} \Bigg( V(Q,\theta) - \int Q(dZ_2, dX_2 \vert z_1, x_1) \log \frac{Q(Z_{2}, X_{2} \vert z_1, x_1)}{P_\theta(Z_{2}, X_{2} \vert z_1, x_1)} \Bigg) \\ & \\ & = \displaystyle - \frac{d}{d\theta} \Big( H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1}, X_{1} \vert Z_0, X_0) \\ & \\ & \qquad \qquad - H_{\mathcal{M}(\mathcal{P}_{Q},\pi_{z_1, x_1}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_1, X_1 \vert Z_0, X_0) \Big) \end{array}$$

where $$\pi_{z_1, x_1}$$ is the initial distribution with $$Z_1 = z_1, X_1 = x_1$$ almost surely. Then,

$$\begin{array}{rl} & \mathcal{P}_{Q,\theta} \hat{E}_{Q, \theta} (w_0) \\ & \\ & = E_{Q,\theta}(w_0) + \hat{E}_{Q, \theta} (w_0) \\ & \\ & = - \displaystyle \lim_{n \rightarrow \infty} \sum_{k=0}^{n-1} \mathcal{P}_{Q,\theta}^{k+1} E_{Q,\theta}(w_0) \\ & \\ & = \displaystyle \lim_{n \rightarrow \infty} \frac{d}{d\theta} \Big( \sum_{k=0}^{n-1} H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{k+1}, X_{k+1} \vert Z_k, X_k) \\ & \\ & \quad \displaystyle - \sum_{k=0}^{n-1} H_{\mathcal{M}(\mathcal{P}_{Q},\pi_{z_1, x_1}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{k+1}, X_{k+1} \vert Z_k, X_k) \Big) \\ & \\ & = \displaystyle \lim_{n \rightarrow \infty} \frac{d}{d\theta} \Big( H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{n}, X_{n} \vert Z_0, X_0) \\ & \\ & \quad \displaystyle - H_{\mathcal{M}(\mathcal{P}_{Q},\pi_{z_1, x_1}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{n}, X_{n} \vert Z_0, X_0) \Big) \end{array}$$

Bringing them all together,

$$\begin{array}{rl} V(Q,\theta) &= H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1}, X_{1} \vert Z_0, X_0) \\ & \\ g(Q, \theta) & = \displaystyle \frac{\partial V}{\partial \theta}(Q, \theta) \\ & \\ E_{Q,\theta}(w_0) & = - \displaystyle  \frac{d}{d\theta} \Big( V(Q,\theta)+ \log P_\theta(z_{1}, x_{1} \vert z_0, x_0) \Big) \\ & \\ \mathcal{P}_{Q,\theta} \hat{E}_{Q, \theta} (w_0) & = \displaystyle \lim_{n \rightarrow \infty} \frac{d}{d\theta} \Big( H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{n}, X_{n} \vert Z_0, X_0) \\ & \\ & \quad \displaystyle - H_{\mathcal{M}(\mathcal{P}_{Q},\pi_{z_1, x_1}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{n}, X_{n} \vert Z_0, X_0) \Big) \\ & \\ \hat{E}_{Q, \theta} (w_0) & = \mathcal{P}_{Q,\theta} \hat{E}_{Q, \theta} (w_0) - E_{Q,\theta}(w_0) \end{array}$$

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

$$\begin{array}{rl} \displaystyle \theta_{k+1} &= \displaystyle \theta_k + \eta_{k+1} \left. \frac{d}{d\theta} \log P_\theta(Z_{k+1}, U_{k+1} \vert Z_k, U_k) \right|_{\theta = \theta_k}, \end{array}$$

$$Q_{k+1} = F(Q_k, \theta_{k+1}).$$

for $$0 \leq k \leq n,$$ using step sizes $$\eta_k = \eta_0 k^{-1/2}$$ for sufficiently small $$\eta_0 \geq 0,$$ and using a random stop time $$0 \leq N \leq n$$ with $$\mathbb{P}(N = l) := (\sum_{k=0}^n \eta_{k+1})^{-1} \eta_{l+1}.$$ Then assuming C1-C5, we have

$$\mathbb{E}(\Vert g(Q_N, \theta_N) \Vert^2) = O(\log n / \sqrt{n} ).$$

----

## Appendix: Discriminative model update

In this appendix, we derive the gradient

$$\frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{1} \vert Z_{0}, X_{0})$$

used in the discriminative model update.

We start with the following formula from [BB1] and [KMMW19] for the integral of a function $$r(Y)$$ with respect to the derivative of the stationary distribution $$\bar{\pi}_\lambda(Y)$$.

$$\begin{array}{rl} &
\displaystyle \int r(Y) \frac{d}{d\lambda} \bar{\pi}_\lambda(dY) 
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \sum_{t=0}^T \int \bar{\pi}_\lambda(dY_0) \int  \prod_{i=0}^{t} Q_\lambda(dY_{i+1}\vert Y_i)     \,\,\times
\\ & \\ &
\quad \quad \displaystyle  r(Y_{t+1}) \frac{d}{d\lambda} \log Q_\lambda(Y_1 \vert Y_0)  
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \sum_{t=0}^T \int \bar{\pi}_\lambda(dY_{T-t}) \int  \prod_{i=T-t}^{T} Q_\lambda(dY_{i+1}\vert Y_i)   
\\ & \\ &
\quad \quad \displaystyle  r(Y_{T+1}) \frac{d}{d\lambda} \log Q_\lambda(Y_{T-t+1} \vert Y_{T-t})
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \sum_{t=0}^T \mathbb{E}_{Q_\lambda(Y_{0..(T+1)})} \left[ r(Y_{T+1})  \frac{d}{d\lambda} \log Q_\lambda(Y_{T-t+1} \vert  Y_{T-t}) \right]
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Y_{0..(T+1)})} \left[ r(Y_{T+1}) \sum_{t=0}^T \frac{d}{d\lambda} \log Q_\lambda(Y_{t+1} \vert Y_{t}) \right]
. \end{array}$$

We now derive the discriminative model update. By the product rule,

$$\begin{array}{rl} & 
\displaystyle \frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{1} \vert Z_{0}, X_{0}) 
\\ & \\ &= 
\displaystyle \frac{d}{d\lambda} \int \bar{\pi}_\lambda(dZ_0,dX_0) \int Q_\lambda(dZ_1\vert Z_0,X_0) \log \frac{Q_\lambda(Z_1 \vert Z_0,X_0)}{P_\theta(Z_1 \vert Z_0,X_0)} 
\\ & \\ & = 
\displaystyle  \int  \frac{d}{d\lambda} \bar{\pi}_\lambda(dZ_0,dX_0) \int Q_\lambda(dZ_1 \vert Z_0,X_0) \log \frac{Q_\lambda(Z_1\vert Z_0,X_0)}{P_\theta(Z_1\vert Z_0,X_0)} 
\\ & \\ & 
\quad +  \displaystyle \int \bar{\pi}_\lambda(dZ_0,dX_0) \int \frac{d}{d\lambda} Q_\lambda(dZ_1 \vert Z_0,X_0) \log \frac{Q_\lambda(Z_1\vert Z_0,X_0)}{P_\theta(Z_1\vert Z_0,X_0)} 
\\ & \\ & 
\quad + \displaystyle  \int \bar{\pi}_\lambda(dZ_0,dX_0) \int Q_\lambda(dZ_1 \vert Z_0,X_0) \frac{d}{d\lambda} \log \frac{Q_\lambda(Z_1\vert Z_0,X_0)}{P_\theta(Z_1\vert Z_0,X_0)}
. \end{array}$$

The third term equals 

$$\begin{array}{rl} &
\displaystyle \int \bar{\pi}_\lambda(dZ_0,dX_0)  \int Q_\lambda(dZ_1 \vert Z_0,X_0) \frac{\frac{d}{d\lambda} Q_\lambda(Z_1\vert Z_0,X_0)}{Q_\lambda(Z_1\vert Z_0,X_0)} \\ & \\ & = \displaystyle \int \bar{\pi}_\lambda(dZ_0,dX_0) \frac{d}{d\lambda}\int Q_\lambda(dZ_1\vert Z_0,X_0) \\ & \\ & = \displaystyle \int \bar{\pi}_\lambda(dZ_0,dX_0) \frac{d}{d\lambda} 1 \\ & \\ & = 0. \end{array}$$

The second terms equals

$$\begin{array}{rl} &
\displaystyle \int \bar{\pi}_\lambda(dZ_0,dX_0) \int  Q_\lambda(dZ_1 \vert Z_0,X_0) \,\,\times \\ &  \\ & \quad \displaystyle \left(\log \frac{Q_\lambda(Z_1\vert Z_0,X_0)}{P_\theta(Z_1\vert Z_0,X_0)}\right) \frac{\frac{d}{d\lambda} \log Q_\lambda(dZ_1 \vert Z_0,X_0)}{ Q_\lambda(dZ_1 \vert Z_0,X_0)}
\\ & \\ &
= \displaystyle \int \bar{\pi}_\lambda(dZ_0,dX_0) \int  Q_\lambda(dZ_1 \vert Z_0,X_0) \,\,\times \\ &  \\ & \quad \displaystyle \left(\log \frac{Q_\lambda(Z_1\vert Z_0,X_0)}{P_\theta(Z_1\vert Z_0,X_0)}\right) \frac{d}{d\lambda} \log Q_\lambda(dZ_1 \vert Z_0,X_0)
\\ & \\ &
= \displaystyle \lim_{T \rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+1)},X_{0..(T+1)})} \Bigg[ \left(\log \frac{Q_\lambda(Z_{T+1}\vert Z_T,X_T)}{P_\theta(Z_{T+1}\vert Z_T,X_T)}\right) \frac{d}{d\lambda} \log Q_\lambda(dZ_{T+1} \vert Z_T,X_T) \Bigg]
. \end{array}$$

Taking derivatives of the stationary distribution, the first term becomes

$$\begin{array}{rl} &
\displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+1)},X_{0..(T+1)})} \Bigg[ \int Q_\lambda(dZ_{T+2}\vert Z_{T+1},X_{T+1}) \log \frac{Q_\lambda(Z_{T+2} \vert Z_{T+1},X_{T+1})}{P_\theta(Z_{T+2} \vert Z_{T+1},X_{T+1})} \,\,\times 
\\ & \\ & \quad\quad \displaystyle \sum_{t=0}^T \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1},X_{t+1} \vert  Z_{t},X_{t}) \Bigg]
\\ & \\ & =
\displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+2)},X_{0..(T+2)})} \Bigg[ \left( \log \frac{Q_\lambda(Z_{T+2}\vert Z_{T+1},X_{T+1})}{P_\theta(Z_{T+2}\vert Z_{T+1},X_{T+1})} \right) \,\,\times 
\\ & \\ & \quad\quad \displaystyle \sum_{t=0}^T \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1},X_{t+1} \vert  Z_{t},X_{t}) \Bigg]
\\ & \\ & =
\displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+2)},X_{0..(T+2)})} \Bigg[ \left( \log \frac{Q_\lambda(Z_{T+2}\vert Z_{T+1},X_{T+1})}{P_\theta(Z_{T+2}\vert Z_{T+1},X_{T+1})} \right) \,\,\times 
\\ & \\ & \quad\quad \displaystyle \sum_{t=0}^T \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \Bigg]
\end{array}$$

where the last equality follows because

$$
\begin{array}{rl} &
\displaystyle \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1},X_{t+1} \vert  Z_{t},X_{t}) 
\\ & \\ &
= \displaystyle \frac{d}{d\lambda} \left( \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) + \log Q_*(X_{t+1} \vert X_t) \right)
\\ & \\ &
= \displaystyle \frac{d}{d\lambda}  \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t})
. \end{array}
$$

Combining this with the second term, we get the gradient 

$$\begin{array}{rl} &
\displaystyle \frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert P_\theta}(Z_{1} \vert Z_{0}, X_{0}) 
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+2)},X_{0..(T+2)})} \Bigg[ \left( \log \frac{Q_\lambda(Z_{T+2}\vert Z_{T+1},X_{T+1})}{P_\theta(Z_{T+2}\vert Z_{T+1},X_{T+1})} \right) \,\,\times 
\\ & \\ & \quad\quad \displaystyle \sum_{t=0}^{T+1} \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \Bigg]
. \end{array}$$

## References

[BB01] Baxter, Jonathan, and Peter L. Bartlett. "Infinite-horizon policy-gradient estimation." _Journal of Artificial Intelligence Research_ 15 (2001): 319-350.

[L92] Leroux, Brian G. "Maximum-likelihood estimation for hidden Markov models." _Stochastic processes and their applications_ 40, no. 1 (1992): 127-143.

[S01] Sato, Masa-Aki. "Online model selection based on the variational Bayes." _Neural computation_ 13, no. 7 (2001): 1649-1681.

[KMMW19] Karimi, Belhal, Blazej Miasojedow, Éric Moulines, and Hoi-To Wai. "Non-asymptotic analysis of biased stochastic approximation scheme." _arXiv preprint arXiv:1902.00629_ (2019).