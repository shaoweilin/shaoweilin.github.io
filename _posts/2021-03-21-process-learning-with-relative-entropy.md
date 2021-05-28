---
layout: post
title: Process learning with relative entropy
---

Over the next few articles, we will derive a distributed learning algorithm for spiking neural networks with latent variables that minimizes some natural notion of relative entropy and provably converges over time. We will model these spiking neural networks with stochastic processes: both discrete-time and continuous-time processes, with or without latent variables.

In this article, we give a general overview of information-theoretic approaches to training stochastic processes, while postponing discussions about issues that arise from latent variables.

The actual process that generates the data could possibly be different from the model that we are training on the data. This problem is called model misspecification, which we are not addressing in this series. Our goal is to find a distribution in the model that fits the data well according to some sensible metric.

This post is a continuation from our series on [spiking networks, path integrals and motivic information](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/).

## How do we train a stochastic process using relative entropy?

Suppose that we have a stochastic process $$ \{X_t\}$$ where $$ X: I \rightarrow \mathcal{X}$$ is a map from some time interval $$ I \subset \mathbb{R}$$ to some countable state space $$ \mathcal{X}$$. If $$ I$$ is the continuous interval $$ [0, T],$$ then $$ X$$ is a continuous-time stochastic process. If $$ I$$ is the discrete interval $$ \{0, 1, \ldots, T\},$$ then $$ X$$ is a discrete-time stochastic process.

Let the true distribution of $$ \{X_t\}$$ be given by some path measure $$ Q_*(\gamma)$$ over paths $$ \gamma : I \rightarrow R$$. Suppose that we have a model or a family of path measures $$ P_\theta(\gamma)$$ parametrized by $$ \theta \in \Theta.$$ We also assume absolute continuity for our model, i.e. $$ Q_* \ll P_\theta$$ for all $$ \theta$$. This assumption is trivially satisfied if each model distribution assigns a non-zero measure to every cylinder in the sigma-algebra of our path space. Let $$ X_{a\ldots b}$$ denote the stochastic process over the interval $$ [a,b]$$, i.e. the family $$ \{X_t\}_{t \in [a,b]}.$$

Because we have absolute continuity, the Radon-Nikodym derivative $$ dQ_*/dP_\theta$$ exists, and we may define the time-averaged relative entropy

$$\displaystyle \frac{1}{T} H_{Q_*\Vert P_\theta}(X_{0\ldots T}) = \frac{1}{T} \int \log \frac{dQ_*}{dP_\theta} dQ_*. $$

To find the best model distribution $$ P_\theta$$ for approximation the true distribution $$ Q_*$$, we will minimize this relative entropy with respect to $$ \theta$$. As the time horizon $$ T$$ tends to infinity, we may also consider minimizing the limit of the time-averaged relative entropy

$$\displaystyle \lim_{T\rightarrow \infty} \frac{1}{T} H_{Q_*\Vert P_\theta}(X_{0\ldots T}).$$

## How do we train a stochastic process using relative entropy rate?

In discrete time, by the chain rule ([CR](https://shaoweilin.github.io/conditional-relative-entropy-and-its-axiomatizations/#mjx-eqn-CR)), the relative entropy decomposes into the sum

$$ \begin{array}{rl} H_{Q \Vert P}(X_{0\ldots n}) &= H_{Q \Vert P}(X_n \vert X_{0\ldots (n-1)}) \\ & \\ & \quad + H_{Q \Vert P}(X_{n-1} \vert X_{0\ldots (n-2)}) \\ & \\ & \quad + \cdots + H_{Q \Vert P}(X_1 \vert X_0) +H_{Q \Vert P}( X_0). \end{array}$$

In continuous time, we have the analogous result

$$ \displaystyle H_{Q \Vert P}(X_{0\ldots T}) = \int_0^T \frac{d}{ds}H_{Q \Vert P}(X_{0\ldots s}) \,ds $$

where

$$ \begin{array}{rl} &\displaystyle \frac{d}{ds}H_{Q \Vert P} (X_{0\ldots s}) \\ & \\ &= \lim_{\delta \rightarrow 0} \frac{1}{\delta} \left[ H_{Q \Vert P} (X_{0\ldots s+\delta}) - H_{Q \Vert P} (X_{0\ldots s})\right] \\ & \\ &= \lim_{\delta \rightarrow 0} \frac{1}{\delta} H_{Q \Vert P} (X_{s\ldots s+\delta}\vert X_{0\ldots s}) \end{array}$$

is also known as the _relative entropy rate_ of $$ X_t$$. The differential operator acting on the function $$ H_{Q \Vert P}$$ of the path $$ X_{0\ldots s}$$ may be seen as a generalization of the infinitesimal generator for Markov processes.

In fact, using Kingman's subadditive ergodic theory [L92], we can show that under strong stationarity conditions,

$$ \displaystyle \lim_{n\rightarrow \infty} \frac{1}{n} H_{Q \Vert P}(X_{0\ldots n}) = \lim_{n\rightarrow \infty} H_{Q \Vert P}(X_n \vert X_{0\ldots (n-1)}).$$

In continuous time, we have the corresponding relation

$$ \displaystyle \lim_{T\rightarrow \infty} \frac{1}{T} H_{Q \Vert P}(X_{0\ldots T}) = \lim_{T\rightarrow \infty} \frac{d}{dT}H_{Q \Vert P}(X_{0\ldots T}).$$

This gives us an asymptotic relationship between relative entropy and relative entropy rate.

Hence, to learn a strongly-stationary stochastic process, instead of minimizing the relative entropy to $$ Q_*$$ from $$ P_\theta$$, we may minimize their relative entropy rate. We will assume strong stationarity for the rest of this series.

In the case where we do not have strong stationarity, it will be hard to guarantee convergence of the learning algorithm. Greedily minimizing the terms $$H_{Q \Vert P}(X_t \vert X_{0\ldots (t-1)})$$ of the decomposition

$$ \begin{array}{rl} H_{Q \Vert P}(X_{0\ldots n}) &= H_{Q \Vert P}(X_n \vert X_{0\ldots (n-1)}) \\ & \\ & \quad + H_{Q \Vert P}(X_{n-1} \vert X_{0\ldots (n-2)}) \\ & \\ & \quad + \cdots + H_{Q \Vert P}(X_1 \vert X_0) +H_{Q \Vert P}( X_0). \end{array}$$

may not help in minimizing the asymptotic time-averaged relative entropy, since a parameter update to reduce $$H_{Q \Vert P}(X_t \vert X_{0\ldots (t-1)})$$ may cause earlier terms $$H_{Q \Vert P}(X_s \vert X_{0\ldots (s-1)}), s < t,$$ to increase more than the amount reduced. Getting this balance to work with stochastic updates is tricky and will require some kind of stationarity as well as new mathematical techniques from the field of biased stochastic approximation.

Note that the relative entropy rate does not depend on the initial distributions $$ Q(X_0)$$ and $$ P(X_0).$$ Therefore, by using the time-averaged relative entropy (or equivalently the relative entropy rate) as the learning objective, we are declaring that our main focus is learning the transition kernel of the true distribution.

I prefer to write the relative entropy rate as

$$ \displaystyle \frac{d}{ds}H_{Q \Vert P} (X_{0\ldots s}) = \frac{d X_{0\ldots s}}{ds} * \frac{\partial}{\partial X_{0\ldots s}} H_{Q \Vert P} (X_{0\ldots s})$$

where $$ *$$ is the inner product in some appropriate Hilbert space and

$$ \displaystyle \frac{\partial}{\partial X_{0\ldots s}} F (X_{0\ldots s})$$

is the functional derivative of a path function $$ F$$ with respect to the path $$ X_{0\ldots s}.$$ We will formalize the Hilbert space and functional derivative in another article.

## References

[L92] Leroux, Brian G. "Maximum-likelihood estimation for hidden Markov models." _Stochastic processes and their applications_ 40, no. 1 (1992): 127-143.

[S01] Sato, Masa-Aki. "Online model selection based on the variational Bayes." _Neural computation_ 13, no. 7 (2001): 1649-1681.

[KMMW19] Karimi, Belhal, Blazej Miasojedow, Éric Moulines, and Hoi-To Wai. "Non-asymptotic analysis of biased stochastic approximation scheme." _arXiv preprint arXiv:1902.00629_ (2019).