---
date: 2021-03-23
excerpts: 2
---

# Biased stochastic approximation with mutable processes

The goal of this post is to derive a general online learning recipe for training a [mutable](2021-03-22-relative-inference-with-mutable-processes/#what-is-a-mutable-process) process $\{Z_t,X_t\}$ to learn the true distribution $Q_*(X)$ of a partially-observed Markov process $\{X_t\}$. The recipe returns a generative distribution $P(Z,X)$ whose marginal $P(X)$ approximates $Q_*(X).$ 

The variables $Z$ of the mutable process are auxiliary variables that assist in inference and computation. During training, the distribution of $Z$ given $X$ is controlled by a discriminative model $\{Q(Z\vert X)\}.$ Our method works in both discrete time and continuous time. We assume in the mutable process that for each time $t,$ the variables $Z_t$ and $X_t$ are conditionally independent of each other given their past.

Our strategy is relative inference, where we use a relative information objective that measures the divergence between the discriminative distribution $Q(Z,X)$ and the generative distribution $P(Z,X).$ We minimize this objective by coordinate-wise updates to the discriminative and generative distributions using stochastic gradients. 

We will be using [biased](2020-12-01-biased-stochastic-approximation/) stochastic approximation {cite}`karimi2019non` where the stochastic updates are dependent on the past but the conditional expectation of the stochastic updates given the past is not equal to the mean field. These biased stochastic approximation schemes for mutable processes generalize the classical expectation maximization algorithm for mutable models.

This post is a continuation from our [series](2020-08-28-motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

## What do we assume about the true distribution, the model and the learning objective?

As [before](2021-03-22-relative-inference-with-mutable-processes/), we assume that the universe is a Markov process $\{X_t\},$ and let its true distribution be the path measure $Q_*.$

Suppose that we have a parametric discriminative model $\{Q_\lambda : \lambda \in \Lambda\}$ and a parametric generative model $\{P_\theta : \theta \in \Theta\}$ where the distributions $Q_\lambda$ and $P_\theta$ are path measures on some joint process $\{(Z_t, X_t)\}.$ The random variables $Z_t$ represent computational states in this discriminative-generative model. We can also interpret the $Z_t$ as sample beliefs from belief distributions $Q_\lambda(Z_t\vert Z_{t-1},X_{t-1}).$

We assume that in both models, the distributions are Markov and each $Z_t$ and $X_t$ are conditionally independent given their past.  We also assume that marginals $Q(X_{0\ldots T})$ of the discriminative model distributions $Q_\lambda(Z_{0 \ldots T}, X_{0\ldots T})$ are all equal to the true distribution $Q_*(X_{0\ldots T}).$
 
Some parts of universe $\{X_t\}$ are observed and other parts are hidden. We will impose these conditions by putting constraints on the structure of the models $\{Q_\lambda\}$ and $\{P_\theta\}$, as described in this [post](2021-03-22-relative-inference-with-mutable-processes/). 

Our goal is to train the models by minimizing the asymptotic relative information rate (continuous time) 

$$ \displaystyle
\lim_{T\rightarrow \infty} \frac{d}{dT}I_{Q \Vert P}(Z_{0\ldots T}, X_{0\ldots T})$$

or asymptotic conditional relative information (discrete time)

$$ \displaystyle
\lim_{n \rightarrow \infty} I_{Q \Vert P}(Z_{n+1}, X_{n+1} \vert Z_{n}, X_{n}).$$

over $\{Q_\lambda\}$ and $\{P_\theta\}$. We first explore the problem in discrete time, before discussing the analogous results in continuous time.

We assume that $Q_\lambda$ has a stationary distribution $\bar{\pi}_\lambda,$ and let $\bar{Q}_\lambda$ be the distribution of a Markov chain that has the same transition probabilities as $Q_\lambda$ but has the initial distribution $\bar{\pi}_\lambda.$ Then,

$$ \displaystyle
\lim_{n \rightarrow \infty} I_{Q_\lambda \Vert P_\theta}(Z_{n+1}, X_{n+1} \vert Z_{n}, X_{n}) = I_{\bar{Q}_\lambda \Vert P_\theta}(Z_1, X_1 \vert Z_0, X_0).$$

## What is the general intuition behind online learning for mutable processes?

To minimize the relative information objective, we adopt an approach similar to the expectation-maximization (EM) or exponential-mixture (em) [algorithm](2020-10-23-machine-learning-with-relative-information/). Specifically, we perform coordinate-wise minimization for the discriminative distribution $Q_\lambda$ and for the generative distribution $P_\theta$, updating one distribution while holding the other constant. 

First, we pick some initial generative model distribution $P_{\theta_0}$ and discriminative model distribution $Q_{\lambda_0}.$ Then, for $n = 0, 1, \ldots,$ we repeat the next two steps. Here, we will perform both steps in parallel rather than in an alternating fashion.

----

**Step 1 (generative model update).** Fixing the discriminative model distribution $Q_{\lambda_{n}}(Z_1 \vert Z_0, X_0),$ minimize $I_{\bar{Q}_{\lambda_{n}}\Vert P_{\theta}}(Z_1, X_1 \vert Z_0, X_0)$ over generative model distributions $P_{\theta}$.

By definition,

$$ \begin{array}{rl} & 
I_{\bar{Q}_{\lambda_{n}}\Vert P_{\theta}}(Z_1, X_1 \vert Z_0, X_0)
\\ & \\ & = 
\mathbb{E}_{\bar{Q}_{\lambda_{n}}} [\log Q_{\lambda_{n}}(Z_1, X_1 \vert Z_0,X_0)] 
\\ & \\ & 
\quad - \mathbb{E}_{\bar{Q}_{\lambda_{n}}} [\log P_\theta(Z_1, X_1 \vert Z_0,X_0)], 
\end{array}$$

where we note that the first term is independent of $\theta$.

We update the parameter $\theta$ using the gradient

$$ \displaystyle 
\theta_{n+1} = \theta_n + \eta_{n+1} \mathbb{E}_{\bar{Q}_{\lambda_{n}}} \left[\left.\frac{d}{d\theta} \log P_\theta(Z_1, X_1 \vert Z_0,X_0)\right\vert _{\theta = \theta_n}\right].$$

where we can also write

$$ \begin{array}{rl} &
\displaystyle \mathbb{E}_{\bar{Q}_{\lambda}} \left[\left.\frac{d}{d\theta} \log P_\theta(Z_1, X_1 \vert Z_0,X_0)\right\vert _{\theta = \theta_n}\right]
\\ & \\ & =
\displaystyle
\lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+1)},X_{0..(T+1)})} \left[ \left.\frac{d}{d\theta} \log P_\theta(Z_{T+1}, X_{T+1} \vert Z_{T},X_{T})\right\vert _{\theta = \theta_n} \right]
. \end{array} $$

**Step 2 (discriminative model update).** Fixing the generative model distribution $P_{\theta_{n}},$ minimize $I_{\bar{Q}_\lambda \Vert P_{\theta_{n}}}(Z_1, X_1 \vert Z_0, X_0)$ over discriminative model distributions $Q_\lambda.$

We update the parameter $\lambda$ using the gradient

$$ \displaystyle
\lambda_{n+1} = \displaystyle \lambda_n - \eta_{n+1} \left.\frac{d}{d\lambda} I_{\bar{Q}_\lambda \Vert P_{\theta_{n}}}(Z_1 , X_1 \vert Z_0, X_0)\right\vert _{\lambda = \lambda_n}$$

where, as shown in the [appendix](2021-03-23-biased-stochastic-approximation-with-mutable-processes/#appendix-discriminative-model-update), we have

$$ \begin{array}{rl} &
\displaystyle \frac{d}{d\lambda} I_{\bar{Q}_\lambda \Vert P_\theta}(Z_1, X_1 \vert Z_0, X_0) 
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+1)},X_{0..(T+1)})} \Bigg[ \left( \log \frac{Q_\lambda(Z_{T+1}, X_{T+1}\vert Z_{T},X_{T})}{P_\theta(Z_{T+1},X_{T+1}\vert Z_{T},X_{T})} \right)
\\ & \\ & \quad\quad \displaystyle \times \sum_{t=0}^{T} \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \Bigg]
. \end{array}$$

----

## Is there a stochastic approximation of the above procedure?

In the above two-step procedure, the term

$$ \begin{array}{rl} &
\displaystyle \log \frac{Q_\lambda(Z_{T+1},X_{T+1}\vert Z_{T},X_{T})}{P_\theta(Z_{T+1},X_{T+1}\vert Z_{T},X_{T})} 
\\ & \\ & =
\displaystyle \log \frac{Q_\lambda(Z_{T+1}\vert Z_{T},X_{T})}{P_\theta(Z_{T+1}, X_{T+1}\vert Z_{T},X_{T})} + \log Q_*(X_{T+1}\vert X_{T})
. \end{array}$$

cannot be evaluated because it depends on the true distribution $Q_*.$ Fortunately, this term only scales the discriminative model update; it does not change the direction of the update. We will then replace the unknown $\log Q_*(X_{T+1}\vert X_{T})$ with an estimate.

Suppose we study the asymptotic time-average 

$$ \displaystyle
H = -\lim_{T\rightarrow \infty} \frac{1}{T} \sum_{t=0}^T \log Q_*(X_{t+1}\vert X_{t}) $$

of the negative log-transition of the true distribution. Under mild regularity conditions, we have the ergodic relationship

$$ \displaystyle
H = -\int \bar{\pi}_*(dX_0)Q_*(dX_1\vert X_0) \log Q_*(X_1\vert X_0) $$

where $\bar{\pi}_*$ is the stationary distribution of $Q_*.$ Let $\bar{Q}_*$ be the distribution of the _true stationary process_ with initial distribution $\bar{\pi}_*$ and transition probabilies $Q_*.$ The asymptotic time-average $H$ is therefore the _true conditional entropy_ of $X_1$ given $X_0$ under the true stationary process. 

More [precisely](2020-09-08-building-foundations-of-information-theory-on-relative-information/#how-do-we-derive-entropy-from-relative-information), given random variables $X_0, X_1, X_1',$ we construct two distributions, namely

$$ \displaystyle
\bar{Q}_* \!\times\!\bar{Q}_* (X_1, X_1',X_0) = \bar{\pi}_*(X_0) Q_*(X_1 \vert X_0) Q_*(X_1'\vert X_0),$$

$$ \displaystyle
\bar{Q}_{**} (X_1, X_1', X_0) = \bar{\pi}_*(X_0) Q_*(X_1 \vert X_0) \,\mathbb{I}(X_1 = X_1').$$

where $\mathbb{I}(X_1 = X_1')$ is the indicator function that ensures that $X_1$ and $X_1'$ are copies of each other. Then, the true conditional entropy is

$$ \displaystyle
H = I_{\bar{Q}_{**} \Vert \bar{Q}_* \!\times\! \bar{Q}_*} (X_1 \vert X_0). $$

Let $-\xi$ be an estimate of this true conditional entropy. We can substitute the unknown $\log Q_*(X_{T+1}\vert X_{T})$ with this constant without affecting the convergence of the algorithm, as we shall see in another post. More generally, we can replace the unknown with any estimate $\xi(X_{T+1} \vert X_T)$ that does not depend on parameters $\theta, \lambda$ or beliefs $Z_{T+1}, Z_T.$

Now, the above two-step procedure has the following stochastic approximation. 

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

In continuous time, the above updates will become differential equations. The samples $Z_t$ would be driven by a Poisson process, and the transition probabilities appearing in the updates for $\theta_t$, $\alpha_t$, $\gamma_t$ would be replaced by transition rates. 

Before we make some preliminary observations about this stochastic approximation, let us introduce some terminology. Given $(Z_{n}, X_{n}),$ suppose we sample $(Z_{n+1}, X_{n+1})$ from $Q_\lambda(Z_{n+1},X_{n+1} \vert Z_{n}, X_{n}).$ The _conditional expectation_ of a function $r(Z_{n+1}, X_{n+1}, Z_{n}, X_{n})$ is the expectation of $r$ conditioned on some given values of $(Z_{n}, X_{n}).$ The _mean field_ or _total expectation_ of $r$ is the expectation of its conditional expectation over the stationary distribution $\bar{\pi}_\lambda$ on $(Z_{n}, X_{n}).$

If the conditional expectations of the updates are independent of $(Z_{n}, X_{n})$, then they will be equal to their mean fields. In this case, we say that the stochastic approximation is _unbiased_. On the other hand, if the conditional expectations depend on $(Z_{n}, X_{n})$, we say that the stochastic approximation is _biased_.

In continuous time, the mean fields will be derivatives of relative information rates. The conditional expectations which depend on the current states $(Z_t,X_t)$ will be biased estimates of the mean fields.

## How can we interpret the discriminative model update?

For a fixed generative model $P_\theta,$ the discriminative model update looks for a distribution $Q_\lambda(Z_{n+1}\vert Z_{n},X_{n})$ that minimizes the learning objective $I_{\bar{Q}_\lambda \Vert P_\theta}(Z_{n+1}, X_{n+1} \vert Z_{n}, X_{n}).$ Intuitively, we can think of the update as looking for good belief $Z_{n+1}$ given the previous belief $Z_{n}$ and observation $X_{n}.$

Because $Z_{n+1}$ and $X_{n+1}$ are conditionally independent given the past, the learning objective decomposes as a sum of two terms.

$$ \begin{array}{rl} & 
I_{\bar{Q}_\lambda \Vert P_\theta}(Z_{n+1}, X_{n+1} \vert Z_{n}, X_{n}) \\ & \\ &= I_{\bar{Q}_\lambda \Vert P_\theta}(Z_{n+1} \vert Z_{n}, X_{n}) + I_{\bar{Q}_\lambda \Vert P_\theta}(X_{n+1} \vert Z_{n}, X_{n}) \end{array}$$

The first term vanishes when 

$$ \displaystyle
Q_\lambda(Z_{n+1} \vert Z_{n}, X_{n}) = P_\theta(Z_{n+1} \vert Z_{n}, X_{n}).$$

This term shows that the discriminative model update tends to _exploit_ the generative model $P_\theta(Z_{n+1}\vert Z_{n}, X_{n})$ in generating a belief $Z_{n+1}.$ 

The second term vanishes when $Q_\lambda(X_{n+1}\vert Z_{n},X_{n}) = Q_*(X_{n+1}\vert X_{n})$ equals $P_\theta(X_{n+1}\vert Z_{n},X_{n}),$ but this is clearly impossible because the true distribution is fixed. 

Instead, note that

$$ \begin{array}{rl} & 
I_{\bar{Q}_\lambda \Vert P_\theta}(X_{n+1} \vert Z_n, X_n) 
\\ & \\ & = 
\displaystyle \int \bar{\pi}_*(dX_n) \bar{\pi}_\lambda(dZ_n\vert X_n) I_{Q_*(X_{n+1} \vert X_n) \Vert \mathcal{P}_\theta(X_{n+1} \vert Z_n, X_n)} (X_{n+1})
\end{array}$$

so the parameter $\lambda$ has an effect only on the stationary transition $\bar{\pi}_\lambda(dZ_n\vert dX_n).$ Thus, in the long run, the discriminative model update tends to pair beliefs $Z_n$ with the current $X_n$ such that the generative model $P_\theta(X_{n+1} \vert Z_n, X_n)$ is able to effectively guess the next state $X_{n+1}$ under $Q_*(X_{n+1}\vert X_n).$ In simpler words, the discriminative model update tends to _explore_ good beliefs $Z_n$ for predicting the next observation $X_{n+1}$. 

Note that the above two tendencies to _exploit_ and _explore_ could be in conflict with each other. For example, at the start of the training regime, the generative model $P_\theta$ is often a poor fit for the observations. In exploiting the bad generative model, the discriminative model update may end up with a belief $Z_n$ that poorly predicts the next observation $X_{n+1}$, where the prediction $P_\theta(X_{n+1}\vert Z_n, X_n)$ was made under this same generative model. However, by exploring beliefs $Z_n$ that well predict the next observation under the generative model, the discriminative model update is giving feedback which the generative model update can use for strengthening the useful parts of the generative model. More precisely, the generative model update will make these useful beliefs more likely under $P_\theta(Z_{n+1}\vert Z_{n}, X_{n})$ so that they can be exploited at the next discriminative model update.

In the long run, when the generative model is a good fit for the observations, the tendencies to exploit and to explore will be more in tune with each other. This is because beliefs generated by the model $P_\theta(Z_n\vert Z_{n-1}, X_{n-1})$ will also be useful for predicting the next state $P_\theta(X_{n+1}\vert Z_n,X_n).$

Explicitly, the exploitative part of the discriminative model update is estimated by

$$ \displaystyle  
\left( \log \frac{Q_\lambda(Z_{T+1}\vert Z_{T},X_{T})}{P_\theta(Z_{T+1}\vert Z_{T},X_{T})} \right) \sum_{t=0}^{T} \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) $$

while the explorative part is estimated by

$$ \displaystyle  
\left( \log \frac{Q_*(X_{T+1}\vert X_{T})}{P_\theta(X_{T+1}\vert Z_{T},X_{T})} \right) \sum_{t=0}^{T} \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) . $$

The explorative update is large when $Q_*(X_{T+1}\vert X_{T})$ and $P_\theta(X_{T+1}\vert Z_{T},X_{T})$ are far apart.

In the stochastic approximation, the explorative part is controlled by

$$ \displaystyle
\alpha_{n+1} \,(\xi(X_{n+1} \vert X_n)- \log P_{\theta_{n}}(X_{n+1}\vert Z_{n},X_{n})) $$

where $\xi(X_{n+1} \vert X_n)$ is an estimate of the true log-likelihood $\log Q_*(X_{n+1}\vert X_{n})$. When $X_{n+1}$ is too likely or too unlikely given $(Z_{n}, X_{n})$, there will be a big difference between the log-likelihood $\log P_{\theta_{n}}(X_{n+1}\vert Z_{n},X_{n})$ and the threshold $\xi(X_{n+1} \vert X_n).$ This will generate a strong signal response in the learning system to correct the discrepancy. 

In {cite}`rezende2014stochastic`, this strong signal was called _novelty_ or _surprise_. The authors hypothesized that biological neural networks could implement this signal using neuromodulation.

In {cite}`pozzi2020attention`, a reinforcement learning scheme for training multilayer neural networks was derived. To implement the weight updates, besides computing the usual feedforward signals, the scheme also computes feedback signals using feedback connections, a global modulating signal representing the reward prediction error, and a local gating signal representing top-down attention. The resulting weight updates are Hebbian. 

While there are many interesting similarities between their scheme and our algorithm, one major difference is that we do not require the feedback weights to be the same as the feedforward weights. In our algorithm, the feedback weights are represented by the parameter $\lambda$ and the feedforward weights by $\theta$. At the end of training,  the feedback weights will tend towards the feedforward weights because of the tendency to exploit. However, tying the weights together at the start of training could be detrimental to learning due to the need of the neural network to explore.

## Appendix: Discriminative model update

In this appendix, we derive the gradient

$$ \displaystyle
\frac{d}{d\lambda} I_{\bar{Q}_\lambda \Vert P_\theta}(Z_1 , X_1 \vert Z_0, X_0)$$

used in the discriminative model update. The methods used are similar to those employed in the policy gradient theorem {cite}`baxter2001infinite`.

We start with the following formula from {cite}`baxter2001infinite` and {cite}`karimi2019non` for the integral of a function $r(W)$ with respect to the derivative of the stationary distribution $\bar{\pi}_\lambda(W)$.

$$ \begin{array}{rl} &
\displaystyle \int r(W) \frac{d}{d\lambda} \bar{\pi}_\lambda(dW) 
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \sum_{t=0}^T \int \bar{\pi}_\lambda(dW_0) \int  \prod_{i=0}^{t} Q_\lambda(dW_{i+1}\vert W_i)    
\\ & \\ &
\quad \quad \displaystyle \times\,\, r(W_{t+1}) \frac{d}{d\lambda} \log Q_\lambda(W_1 \vert W_0)  
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \sum_{t=0}^T \int \bar{\pi}_\lambda(dW_{T-t}) \int  \prod_{i=T-t}^{T} Q_\lambda(dW_{i+1}\vert W_i)   
\\ & \\ &
\quad \quad \displaystyle  \times \,\, r(W_{T+1}) \frac{d}{d\lambda} \log Q_\lambda(W_{T-t+1} \vert W_{T-t})
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \sum_{t=0}^T \mathbb{E}_{Q_\lambda(W_{0..(T+1)})} \left[ r(W_{T+1})  \frac{d}{d\lambda} \log Q_\lambda(W_{T-t+1} \vert  W_{T-t}) \right]
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(W_{0..(T+1)})} \left[ r(W_{T+1}) \sum_{t=0}^T \frac{d}{d\lambda} \log Q_\lambda(W_{t+1} \vert W_{t}) \right]
. \end{array}$$

We now derive the discriminative model update. Let $\{W_n\}$ denote the Markov chain $\{(Z_{n+1},X_{n+1},Z_{n},X_{n})\}.$ Abusing notation, we write the distribution of $W_n$ as 

$$ \begin{array}{rl} & \displaystyle 
Q_\lambda(W_{n} \vert W_{n-1}) 
\\ & \\ & = \displaystyle
Q_\lambda\left((\,Z_{n+1},X_{n+1},Z_n,X_n)\, \vert\, (Z_n,X_n,Z_{n-1},X_{n-1}) \, \right)
\\ & \\ & = \displaystyle
Q_\lambda(Z_{n+1},X_{n+1}\vert Z_n, X_n)
\end{array} $$

and its stationary distribution as

$$ \begin{array}{rl} & \displaystyle 
\bar{\pi}_\lambda(W_0) 
\\ & \\ & = \displaystyle
\bar{\pi}_\lambda(Z_{1},X_{1},Z_0,X_0)
\\ & \\ & = \displaystyle
\bar{\pi}_\lambda(Z_0,X_0) Q_\lambda(Z_1,X_1\vert Z_0, X_0)
. \end{array} $$

By the product rule,

$$ \begin{array}{rl} & 
\displaystyle \frac{d}{d\lambda} I_{\bar{Q}_\lambda \Vert P_\theta}(Z_1,X_1 \vert Z_0,X_0) 
\\ & \\ &= 
\displaystyle \frac{d}{d\lambda} \int \left(\log \frac{Q_\lambda(Z_1,X_1 \vert Z_0,X_0)}{P_\theta(Z_1,X_1 \vert Z_0,X_0)} \right) \bar{\pi}_\lambda(dZ_1,dX_1,dZ_0,dX_0) 
\\ & \\ & = 
\displaystyle  \int \left( \frac{d}{d\lambda} \log \frac{Q_\lambda(Z_1,X_1 \vert Z_0,X_0)}{P_\theta(Z_1,X_1 \vert Z_0,X_0)} \right)  \bar{\pi}_\lambda(dZ_1,dX_1,dZ_0,dX_0)
\\ & \\ & 
\quad + \displaystyle  \int\left( \log \frac{Q_\lambda(Z_1,X_1 \vert Z_0,X_0)}{P_\theta(Z_1,X_1 \vert Z_0,X_0)} \right) \frac{d}{d\lambda} \bar{\pi}_\lambda(dZ_1,dX_1,dZ_0,dX_0) 
. \end{array}$$

The first term equals 

$$ \begin{array}{rl} &
\displaystyle \int \frac{\frac{d}{d\lambda} Q_\lambda(Z_1,X_1\vert Z_0,X_0)}{Q_\lambda(Z_1,X_1\vert Z_0,X_0)} Q_\lambda(dZ_1,dX_1 \vert Z_0,X_0)  \bar{\pi}_\lambda(dZ_0,dX_0) 
\\ & \\ & = 
\displaystyle \int \left( \int \frac{d}{d\lambda} Q_\lambda(dZ_1,dX_1\vert Z_0,X_0) \right) \bar{\pi}_\lambda(dZ_0,dX_0)
\\ & \\ & = 
\displaystyle \int \left( \frac{d}{d\lambda}\int Q_\lambda(dZ_1,dX_1\vert Z_0,X_0) \right) \bar{\pi}_\lambda(dZ_0,dX_0)
\\ & \\ & = \int
\displaystyle \left( \frac{d}{d\lambda} 1 \right)  \bar{\pi}_\lambda(dZ_0,dX_0)
\\ & \\ & = 
0. \end{array}$$

Taking derivatives of the stationary distribution, the second term becomes

$$ \displaystyle 
\lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+2)},X_{0..(T+2)})} \Bigg[ \left( \log \frac{Q_\lambda(Z_{T+2},X_{T+2}\vert Z_{T+1},X_{T+1})}{P_\theta(Z_{T+2},X_{T+2}\vert Z_{T+1},X_{T+1})} \right) \sum_{t=1}^T \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1},X_{t+1} \vert  Z_t,X_t) \Bigg]. $$

Lastly, because 

$$ \displaystyle
\begin{array}{rl} &
\displaystyle \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1},X_{t+1} \vert  Z_{t},X_{t}) 
\\ & \\ &
= \displaystyle \frac{d}{d\lambda} \Big( \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) + \log Q_*(X_{t+1} \vert X_t) \Big)
\\ & \\ &
= \displaystyle \frac{d}{d\lambda}  \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t})
, \end{array} $$

the gradient simplifies (after a change of indices) to

$$ \begin{array}{rl} &
\displaystyle \frac{d}{d\lambda} I_{\bar{Q}_\lambda \Vert P_\theta}(Z_1,X_1 \vert Z_0,X_0) 
\\ & \\ & = 
\displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+1)},X_{0..(T+1)})} \Bigg[ \left( \log \frac{Q_\lambda(Z_{T+1},X_{T+1}\vert Z_T,X_T)}{P_\theta(Z_{T+1},X_{T+1}\vert Z_T,X_T)} \right)  \sum_{t=1}^T \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \Bigg]
\\ & \\ & = 
\displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+1)},X_{0..(T+1)})} \Bigg[ \left( \log \frac{Q_\lambda(Z_{T+1},X_{T+1}\vert Z_T,X_T)}{P_\theta(Z_{T+1},X_{T+1}\vert Z_T,X_T)} \right)  \sum_{t=0}^T \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \Bigg]
\end{array}$$

where the last equality follows because the limit does not depend on the initial distribution of $(Z_0, X_0).$

## References

```{bibliography}
:filter: docname in docnames
```