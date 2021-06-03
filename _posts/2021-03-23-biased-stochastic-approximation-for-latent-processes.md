---
layout: post
title: Biased stochastic approximation for latent processes
---

We apply biased stochastic approximation and variational inference to optimize a relative entropy objective for latent Markov processes. We will be using [biased](https://shaoweilin.github.io/biased-stochastic-approximation/) stochastic approximation [KMMW19] where the stochastic updates are dependent on the past but the conditional expectation of the stochastic updates given the past is not equal to the mean field. These biased stochastic approximation schemes generalize the classical expectation maximization algorithm [KMMW19].

This post is a continuation from our [series](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

## What do we assume about the true distribution, the model and the learning objective?

As [before](https://shaoweilin.github.io/variational-inference-for-latent-processes/), we assume that the universe is a Markov process $$\{X_t\},$$ and let its true distribution be the path measure $$Q_*.$$

Suppose that we have a parametric discriminative model $$\{Q_\lambda : \lambda \in \Lambda\}$$ and a parametric generative model $$\{P_\theta : \theta \in \Theta\}$$ where the distributions $$Q_\lambda$$ and $$P_\theta$$ are path measures on some joint process $$\{(Z_t, X_t)\}.$$ The random variables $$Z_t$$ represent computational states in this discriminative-generative model. We can also interpret the $$Z_t$$ as sample beliefs from belief distributions $$Q_\lambda(Z_t\vert Z_{t-1},X_{t-1}).$$

We assume that in both models, the distributions are Markov and each $$Z_t$$ and $$X_t$$ are conditionally independent given their past.  We also assume that marginals $$Q(X_{0\ldots T})$$ of the discriminative model distributions $$Q_\lambda(Z_{0 \ldots T}, X_{0\ldots T})$$ are all equal to the true distribution $$Q_*(X_{0\ldots T}).$$
 
Some parts of universe $$\{X_t\}$$ are observed and other parts are unobserved. We will impose these conditions by putting constraints on the structure of the models $$\{Q_\lambda\}$$ and $$\{P_\theta\}$$, as described in this [article](https://shaoweilin.github.io/variational-inference-for-latent-processes/). 

Our goal is to train the models by minimizing the asymptotic relative entropy rate (continuous time) 

$$\lim_{T\rightarrow \infty} \frac{d}{dT}H_{Q \Vert P}(Z_{0\ldots T}, X_{0\ldots T})$$

or asymptotic conditional relative entropy (discrete time)

$$\lim_{n \rightarrow \infty} H_{Q \Vert P}(Z_{n+1}, X_{n+1} \vert Z_{n}, X_{n}).$$

over $$\{Q_\lambda\}$$ and $$\{P_\theta\}$$. We first explore the problem in discrete time, before discussing the analogous results in continuous time.

We assume that $$Q_\lambda$$ has a stationary distribution $$\bar{\pi}_\lambda,$$ and let $$\bar{Q}_\lambda$$ be the distribution of a Markov chain that has the same transition probabilities as $$Q_\lambda$$ but has the initial distribution $$\bar{\pi}_\lambda.$$ Then,

$$\lim_{n \rightarrow \infty} H_{Q_\lambda \Vert P_\theta}(Z_{n+1}, X_{n+1} \vert Z_{n}, X_{n}) = H_{\bar{Q}_\lambda \Vert P_\theta}(Z_1, X_1 \vert Z_0, X_0).$$

## What is the general intuition behind online learning for latent processes?

To minimize the conditional relative entropy objective, we adopt an approach similar to the expectation-maximization (EM) or exponential-mixture (em) [algorithm](https://shaoweilin.github.io/machine-learning-with-relative-entropy/). More precisely, we iteratively optimize for the discriminative model distribution $$Q_\lambda$$ and for the generative model distribution $$P_\theta$$ while holding the other constant. 

First, we pick some initial generative model distribution $$P_{\theta_0}$$ and discriminative model distribution $$Q_{\lambda_0}.$$ Then, for $$n = 0, 1, \ldots,$$ we repeat the next two steps.

----

**Step 1 (generative model update).** Fixing the discriminative model distribution $$Q_{\lambda_{n}}(Z_1 \vert Z_0, X_0),$$ minimize $$H_{\bar{Q}_{\lambda_{n}}\Vert P_{\theta}}(Z_1, X_1 \vert Z_0, X_0)$$ over generative model distributions $$P_{\theta}$$.

By definition,

$$\begin{array}{rl} & 
H_{\bar{Q}_{\lambda_{n}}\Vert P_{\theta}}(Z_1, X_1 \vert Z_0, X_0)
\\ & \\ & = 
\mathbb{E}_{\bar{Q}_{\lambda_{n}}} [\log Q_{\lambda_{n}}(Z_1, X_1 \vert Z_0,X_0)] 
\\ & \\ & 
\quad - \mathbb{E}_{\bar{Q}_{\lambda_{n}}} [\log P_\theta(Z_1, X_1 \vert Z_0,X_0)], 
\end{array}$$

where we note that the first term is independent of $$\theta$$.

We update the parameter $$\theta$$ using the gradient

$$\displaystyle \theta_{n+1} = \theta_n + \eta_{n+1} \mathbb{E}_{\bar{Q}_{\lambda_{n}}} \left[\left.\frac{d}{d\theta} \log P_\theta(Z_1, X_1 \vert Z_0,X_0)\right\vert _{\theta = \theta_n}\right].$$

where we can also write

$$ \begin{array}{rl} &
\displaystyle \mathbb{E}_{\bar{Q}_{\lambda}} \left[\left.\frac{d}{d\theta} \log P_\theta(Z_1, X_1 \vert Z_0,X_0)\right\vert _{\theta = \theta_n}\right]
\\ & \\ & =
\displaystyle
\lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+1)},X_{0..(T+1)})} \left[ \left.\frac{d}{d\theta} \log P_\theta(Z_{T+1}, X_{T+1} \vert Z_{T},X_{T})\right\vert _{\theta = \theta_n} \right]
. \end{array}
$$

**Step 2 (discriminative model update).** Fixing the generative model distribution $$P_{\theta_{n+1}},$$ minimize $$H_{\bar{Q}_\lambda \Vert P_{\theta_{n+1}}}(Z_1, X_1 \vert Z_0, X_0)$$ over discriminative model distributions $$Q_\lambda.$$

We update the parameter $$\lambda$$ using the gradient

$$\lambda_{n+1} = \displaystyle \lambda_n - \eta_{n+1} \left.\frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert P_{\theta_{n+1}}}(Z_1 , X_1 \vert Z_0, X_0)\right\vert _{\lambda = \lambda_n}$$

where, as shown in the [appendix](https://shaoweilin.github.io/biased-stochastic-approximation-for-latent-processes/#appendix-discriminative-model-update), we have

$$\begin{array}{rl} &
\displaystyle \frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert P_\theta}(Z_1, X_1 \vert Z_0, X_0) 
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+1)},X_{0..(T+1)})} \Bigg[ \left( \log \frac{Q_\lambda(Z_{T+1}, X_{T+1}\vert Z_{T},X_{T})}{P_\theta(Z_{T+1},X_{T+1}\vert Z_{T},X_{T})} \right) \,\,\times 
\\ & \\ & \quad\quad \displaystyle \sum_{t=0}^{T} \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \Bigg]
. \end{array}$$

----

## Is there a stochastic approximation of the above procedure?

In the above twp-step procedure, the term

$$\begin{array}{rl} &
\displaystyle \log \frac{Q_\lambda(Z_{T+1},X_{T+1}\vert Z_{T},X_{T})}{P_\theta(Z_{T+1},X_{T+1}\vert Z_{T},X_{T})} 
\\ & \\ & =
\displaystyle \log \frac{Q_\lambda(Z_{T+1}\vert Z_{T},X_{T})}{P_\theta(Z_{T+1}, X_{T+1}\vert Z_{T},X_{T})} + \log Q_*(X_{T+1}\vert X_{T})
. \end{array}$$

cannot be evaluated because it depends on the true distribution $$Q_*.$$ Fortunately, this term only scales the discriminative model update; it does not change the direction of the update. We will then replace the unknown $$\log Q_*(X_{T+1}\vert X_{T})$$ with an estimate.

Suppose we study the asymptotic time-average 

$$
\xi = -\lim_{T\rightarrow \infty} \frac{1}{T} \sum_{t=0}^T \log Q_*(X_{t+1}\vert X_{t})
$$

of the negative log-transition of the true distribution. Under mild regularity conditions, we have the ergodic relationship

$$
\xi = 
-\int \bar{\pi}_*(dX_0)Q_*(dX_1|X_0) \log Q_*(X_1|X_0)
$$

where $$\bar{\pi}_*$$ is the stationary distribution of $$Q_*.$$ Let $$\bar{Q}_*$$ be the distribution of the _true stationary process_ with initial distribution $$\bar{\pi}_*$$ and transition probabilies $$Q_*.$$ The asymptotic time-average $$\xi$$ is therefore the _true conditional entropy_ of $$X_1$$ given $$X_0$$ under the true stationary process. 

More [precisely](https://shaoweilin.github.io/building-foundations-of-information-theory-on-relative-entropy/#how-do-we-derive-entropy-from-relative-entropy), given random variables $$X_0, X_1, X_1',$$ we construct two distributions, namely

$$\bar{Q}_* \!\times\!\bar{Q}_* (X_1, X_1',X_0) = \bar{\pi}_*(X_0) Q_*(X_1 \vert X_0) Q_*(X_1'\vert X_0),$$

$$\bar{Q}_{**} (X_1, X_1', X_0) = \bar{\pi}_*(X_0) Q_*(X_1 \vert X_0) \,\mathbf{1}(X_1 = X_1').$$

where $$\mathbf{1}(X_1 = X_1')$$ is the indicator function that ensures that $$X_1$$ and $$X_1'$$ are copies of each other. Then, the true conditional entropy is

$$
\xi = H_{\bar{Q}_{**} \Vert \bar{Q}_* \!\times\! \bar{Q}_*} (X_1 \vert X_0).
$$

Let $$\hat{\xi}$$ be an estimate of this true conditional entropy. 

Now, the above two-step procedure has the following stochastic approximation. 

$$\displaystyle \theta_{n+1} = \theta_n + \eta_{n+1} \left.\frac{d}{d\theta} \log P_\theta(Z_{n}, X_{n} \vert Z_{n-1},X_{n-1}) \right\vert _{\theta = \theta_n}$$

$$\displaystyle \lambda_{n+1} = \lambda_n - \eta_{n+1} \alpha_{n} \left( \log \frac{Q_{\lambda_n}(Z_{n}\vert Z_{n-1},X_{n-1})}{P_{\theta_{n+1}}(Z_{n},X_{n}\vert Z_{n-1},X_{n-1})} -\hat{\xi} \right)$$ 

$$\displaystyle X_{n+1} \sim Q_*(X_{n+1} \vert X_{n})$$

$$\displaystyle Z_{n+1} \sim Q_{\lambda_{n+1}}(Z_{n+1} \vert Z_{n}, X_{n})$$

$$\displaystyle \alpha_{n+1} = \alpha_n + \left.\frac{d}{d\lambda} \log Q_{\lambda}(Z_{n+1} \vert  Z_{n},X_{n})\right\vert _{\lambda=\lambda_{n+1}}$$

In continuous time, the above updates will become differential equations. The samples $$Z_t$$ would be driven by a Poisson process, and the transition probabilities appearing in the updates for $$\theta_t$$, $$\alpha_t$$ and $$\beta_t$$ would be replaced by transition rates. 

Before we make some preliminary observations about this stochastic approximation, let us introduce some terminology. Given $$(Z_{n-1}, X_{n-1}),$$ suppose we sample $$(Z_n, X_n)$$ from $$Q_\lambda(Z_n,X_n \vert Z_{n-1}, X_{n-1}).$$ The _conditional expectation_ of a function $$r(Z_n, X_n, Z_{n-1}, X_{n-1})$$ is the expectation of $$r$$ conditioned on some given values of $$(Z_{n-1}, X_{n-1}).$$ The _mean field_ or _total expectation_ of $$r$$ is the expectation of its conditional expectation over the stationary distribution $$\bar{\pi}_\lambda$$ on $$(Z_{n-1}, X_{n-1}).$$

If the conditional expectations of the updates are independent of $$(Z_{n-1}, X_{n-1})$$, then they will be equal to their mean fields. In this case, we say that the stochastic approximation is _unbiased_. On the other hand, if the conditional expectations depend on $$(Z_{n-1}, X_{n-1})$$, we say that the stochastic approximation is _biased_.

In continuous time, the mean fields will be derivatives of relative entropy rates. The conditional expectations which depend on the current states $$(Z_t,X_t)$$ will be biased estimates of the mean fields.

## How can we interpret the discriminative model update?

For a fixed generative model $$P_\theta,$$ the discriminative model update looks for a distribution $$Q_\lambda(Z_n\vert Z_{n-1},X_{n-1})$$ that minimizes the learning objective $$H_{\bar{Q}_\lambda \Vert P_\theta}(Z_n, X_n \vert Z_{n-1}, X_{n-1}).$$ Intuitively, we can think of the update as looking for good belief $$Z_n$$ given the previous belief $$Z_{n-1}$$ and observation $$X_{n-1}.$$

Because $$Z_n$$ and $$X_n$$ are conditionally independent given the past, the learning objective decomposes as a sum of two terms.

$$\begin{array}{rl} & H_{\bar{Q}_\lambda \Vert P_\theta}(Z_n, X_n \vert Z_{n-1}, X_{n-1}) \\ & \\ &= H_{\bar{Q}_\lambda \Vert P_\theta}(Z_n \vert Z_{n-1}, X_{n-1}) + H_{\bar{Q}_\lambda \Vert P_\theta}(X_n \vert Z_{n-1}, X_{n-1}) \end{array}$$

The first term vanishes when 

$$ Q_\lambda(Z_n \vert Z_{n-1}, X_{n-1}) = P_\theta(Z_n \vert Z_{n-1}, X_{n-1}).$$

This term shows that the discriminative model update tends to _exploit_ the generative model $$P_\theta(Z_n\vert Z_{n-1}, X_{n-1})$$ in generating a belief $$Z_n.$$ 

The second term vanishes when $$Q_\lambda(X_n\vert Z_{n-1},X_{n-1}) = Q_*(X_n\vert X_{n-1})$$ equals $$P_\theta(X_n\vert Z_{n-1},X_{n-1}),$$ but this is clearly impossible because the true distribution is fixed. 

Instead, note that (after a change of indices)

$$\begin{array}{rl} & 
H_{\bar{Q}_\lambda \Vert P_\theta}(X_{n+1} \vert Z_n, X_n) 
\\ & \\ & = 
\displaystyle \int \bar{\pi}_*(dX_n) \bar{\pi}_\lambda(dZ_n\vert dX_n) H_{Q_*(X_{n+1} \vert X_n) \Vert \mathcal{P}_\theta(X_{n+1} \vert Z_n, X_n)} (X_{n+1})
\end{array}$$

so the parameter $$\lambda$$ has an effect only on the stationary transition $$\bar{\pi}_\lambda(dZ_n\vert dX_n).$$ Thus, in the long run, the discriminative model update tends to pair beliefs $$Z_n$$ with the current $$X_n$$ such that the generative model $$P_\theta(X_{n+1} \vert Z_n, X_n)$$ is able to effectively guess the next state $$X_{n+1}$$ under $$Q_*(X_{n+1}\vert X_n).$$ In simpler words, the discriminative model update tends to _explore_ good beliefs $$Z_n$$ for predicting the next observation $$X_{n+1}$$. 

Note that the above two tendencies to _exploit_ and _explore_ could be in conflict with each other. For example, at the start of the training regime, the generative model $$P_\theta$$ is often a poor fit for the observations. In exploiting the bad generative model, the discriminative model update may end up with a belief $$Z_n$$ that poorly predicts the next observation $$X_{n+1}$$, where the prediction $$P_\theta(X_{n+1}\vert Z_n, X_n)$$ was made under this same generative model. However, by exploring beliefs $$Z_n$$ that well predict the next observation under the generative model, the discriminative model update is giving feedback which the generative model update can use for strengthening the useful parts of the generative model. More precisely, the generative model update will make these useful beliefs more likely under $$P_\theta(Z_n\vert Z_{n-1}, X_{n-1})$$ so that they can be exploited at the next discriminative model update.

In the long run, when the generative model is a good fit for the observations, the tendencies to exploit and to explore will be more in tune with each other. This is because beliefs generated by the model $$P_\theta(Z_n\vert Z_{n-1}, X_{n-1})$$ will also be useful for predicting the next state $$P_\theta(X_{n+1}\vert Z_n,X_n).$$

Explicitly, the exploitative part of the discriminative model update is estimated by

$$
\displaystyle  \left( \log \frac{Q_\lambda(Z_{T+1}\vert Z_{T},X_{T})}{P_\theta(Z_{T+1}\vert Z_{T},X_{T})} \right) \sum_{t=0}^{T} \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t})
$$

while the explorative part is estimated by

$$
\displaystyle  \left( \log \frac{Q_*(X_{T+1}\vert X_{T})}{P_\theta(X_{T+1}\vert Z_{T},X_{T})} \right) \sum_{t=0}^{T} \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) .
$$

The explorative update is large when $$Q_*(X_{T+1}\vert X_{T})$$ and $$P_\theta(X_{T+1}\vert Z_{T},X_{T})$$ are far apart.


In the stochastic approximation, the explorative part is controlled by

$$
\alpha_{n+2} \,(- \log P_{\theta_{n+2}}(X_{n+1}\vert Z_{n},X_{n}) -\hat{\xi})
$$

where $$\hat{\xi}$$ is a fixed estimate of the true conditional entropy. When $$X_{n+1}$$ is too likely or too unlikely given $$(Z_{n}, X_{n})$$, there will be a big difference between the negative log-likelihood $$- \log P_{\theta_{n+2}}(X_{n+1}\vert Z_{n},X_{n})$$ and the threshold $$\hat{\xi}.$$ This will generate a strong signal response in the learning system to correct the discrepancy. 

In [JG14], this strong signal was called _novelty_ or _surprise_. The authors hypothesized that biological neural networks could implement this signal using neuromodulation.

In [PBR20], a reinforcement learning scheme for training multilayer neural networks was derived. To implement the weight updates, besides computing the usual feedforward signals, the scheme also computes feedback signals using feedback connections, a global modulating signal representing the reward prediction error, and a local gating signal representing top-down attention. The resulting weight updates are Hebbian. 

While there are many interesting similarities between their scheme and our algorithm, one major difference is that we do not require the feedback weights to be the same as the feedforward weights. In our algorithm, the feedback weights are represented by the parameter $$\lambda$$ and the feedforward weights by $$\theta$$. At the end of training,  the feedback weights will tend towards the feedforward weights because of the tendency to exploit. However, tying the weights together at the start of training could be detrimental to learning due to the need of the neural network to explore.

## Appendix: Discriminative model update

In this appendix, we derive the gradient

$$\frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert P_\theta}(Z_1 , X_1 \vert Z_0, X_0)$$

used in the discriminative model update. The methods used are similar to those employed in the policy gradient theorem [BB01].

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

We now derive the discriminative model update. Let $$Y_n$$ denote $$(Z_n,X_n).$$ By the product rule,

$$\begin{array}{rl} & 
\displaystyle \frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert P_\theta}(Y_1 \vert Y_0) 
\\ & \\ &= 
\displaystyle \frac{d}{d\lambda} \int \bar{\pi}_\lambda(dY_0) \int Q_\lambda(dY_1\vert Y_0) \log \frac{Q_\lambda(Y_1 \vert Y_0)}{P_\theta(Y_1 \vert Y_0)} 
\\ & \\ & = 
\displaystyle  \int  \frac{d}{d\lambda} \bar{\pi}_\lambda(dY_0) \int Q_\lambda(dY_1\vert Y_0) \log \frac{Q_\lambda(Y_1\vert Y_0)}{P_\theta(Y_1\vert Y_0)} 
\\ & \\ & 
\quad +  \displaystyle \int \bar{\pi}_\lambda(dY_0) \int \frac{d}{d\lambda} Q_\lambda(dY_1\vert Y_0) \log \frac{Q_\lambda(Y_1\vert Y_0)}{P_\theta(Y_1\vert Y_0)} 
\\ & \\ & 
\quad + \displaystyle  \int \bar{\pi}_\lambda(dY_0) \int Q_\lambda(dY_1\vert Y_0) \frac{d}{d\lambda} \log \frac{Q_\lambda(Y_1\vert Y_0)}{P_\theta(Y_1\vert Y_0)}
. \end{array}$$

The third term equals 

$$\begin{array}{rl} &
\displaystyle \int \bar{\pi}_\lambda(dY_0)  \int Q_\lambda(dY_1 \vert Y_0) \frac{\frac{d}{d\lambda} Q_\lambda(Y_1\vert Y_0)}{Q_\lambda(Y_1\vert Y_0)} 
\\ & \\ & = 
\displaystyle \int \bar{\pi}_\lambda(dY_0) \frac{d}{d\lambda}\int Q_\lambda(dY_1\vert Y_0) 
\\ & \\ & = 
\displaystyle \int \bar{\pi}_\lambda(dY_0) \frac{d}{d\lambda} 1 
\\ & \\ & = 
0. \end{array}$$

The second terms equals

$$\begin{array}{rl} &
\displaystyle \int \bar{\pi}_\lambda(dY_0) \int  Q_\lambda(dY_1 \vert Y_0)  \left(\log \frac{Q_\lambda(Y_1\vert Y_0)}{P_\theta(Y_1\vert Y_0)}\right) \frac{\frac{d}{d\lambda} \log Q_\lambda(Y_1 \vert Y_0)}{ Q_\lambda(Y_1 \vert Y_0)}
\\ & \\ &
= \displaystyle \int \bar{\pi}_\lambda(dY_0) \int  Q_\lambda(dY_1\vert Y_0)  \left(\log \frac{Q_\lambda(Y_1\vert Y_0)}{P_\theta(Y_1\vert Y_0)}\right) \frac{d}{d\lambda} \log Q_\lambda(Y_1 \vert Y_0)
\\ & \\ &
= \displaystyle \lim_{T \rightarrow \infty} \mathbb{E}_{Q_\lambda(Y_{0..(T+1)})} \Bigg[ \left(\log \frac{Q_\lambda(Y_{T+1}\vert Y_T)}{P_\theta(Y_{T+1}\vert Y_T)}\right) \frac{d}{d\lambda} \log Q_\lambda(dY_{T+1} \vert Y_T) \Bigg]
. \end{array}$$

Taking derivatives of the stationary distribution, the first term becomes

$$\begin{array}{rl} &
\displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Y_{0..(T+1)})} \Bigg[ \int Q_\lambda(dY_{T+2}\vert Y_{T+1}) \left(\log \frac{Q_\lambda(Y_{T+2} \vert Y_{T+1})}{P_\theta(Y_{T+2} \vert Y_{T+1})} \right)\sum_{t=0}^T \frac{d}{d\lambda} \log Q_\lambda(Y_{t+1}\vert  Y_{t}) \Bigg]
\\ & \\ & =
\displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Y_{0..(T+2)})} \Bigg[ \left( \log \frac{Q_\lambda(Y_{T+2}\vert Y_{T+1})}{P_\theta(Y_{T+2}\vert Y_{T+1})} \right) \sum_{t=0}^T \frac{d}{d\lambda} \log Q_\lambda(Y_{t+1} \vert  Y_{t}) \Bigg]
\end{array}$$

Combining this with the second term, we get 

$$\begin{array}{rl} &
\displaystyle \frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert P_\theta}(Y_1 \vert Y_0) 
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Y_{0..(T+2)})} \Bigg[ \left( \log \frac{Q_\lambda(Y_{T+2}\vert Y_{T+1})}{P_\theta(Y_{T+2}\vert Y_{T+1})} \right)  \sum_{t=0}^{T+1} \frac{d}{d\lambda} \log Q_\lambda(Y_{t+1} \vert  Y_{t}) \Bigg]
. \end{array}$$

Lastly, because 

$$
\begin{array}{rl} &
\displaystyle \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1},X_{t+1} \vert  Z_{t},X_{t}) 
\\ & \\ &
= \displaystyle \frac{d}{d\lambda} \Big( \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) + \log Q_*(X_{t+1} \vert X_t) \Big)
\\ & \\ &
= \displaystyle \frac{d}{d\lambda}  \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t})
, \end{array}
$$

the gradient simplifies (after a change of indices) to

$$\begin{array}{rl} &
\displaystyle \frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert P_\theta}(Z_1,X_1 \vert Z_0,X_0) 
\\ & \\ & = 
\displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+1)},X_{0..(T+1)})} \Bigg[ \left( \log \frac{Q_\lambda(Z_{T+1},X_{T+1}\vert Z_T,X_T)}{P_\theta(Z_{T+1},X_{T+1}\vert Z_T,X_T)} \right)  \sum_{t=0}^T \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \Bigg]
. \end{array}$$

## References

[BB01] Baxter, Jonathan, and Peter L. Bartlett. "Infinite-horizon policy-gradient estimation." _Journal of Artificial Intelligence Research_ 15 (2001): 319-350.

[JG14] Jimenez Rezende, Danilo, and Wulfram Gerstner. "Stochastic variational learning in recurrent spiking networks." _Frontiers in computational neuroscience_ 8 (2014): 38.

[KMMW19] Karimi, Belhal, Blazej Miasojedow, Éric Moulines, and Hoi-To Wai. "Non-asymptotic analysis of biased stochastic approximation scheme." _arXiv preprint arXiv:1902.00629_ (2019).

[L92] Leroux, Brian G. "Maximum-likelihood estimation for hidden Markov models." _Stochastic processes and their applications_ 40, no. 1 (1992): 127-143.

[PBR20] Pozzi, Isabella, Sander Bohte, and Pieter Roelfsema. "Attention-Gated Brain Propagation: How the brain can implement reward-based error backpropagation." _Advances in Neural Information Processing Systems_ 33 (2020).

[S01] Sato, Masa-Aki. "Online model selection based on the variational Bayes." _Neural computation_ 13, no. 7 (2001): 1649-1681.

