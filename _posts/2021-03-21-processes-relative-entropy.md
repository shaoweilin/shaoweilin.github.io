---
layout: post
title: Processes I - Relative Entropy
---

We derive a distributed learning algorithm for spiking neural networks with latent variables that minimizes some natural notion of relative entropy and provably converges over time. We will model these spiking neural networks with stochastic processes: both discrete-time and continuous-time processes, with or without latent variables.

The actual process that generates the data could possibly be different from the model that we are training on the data. This problem is called model misspecification, which we are not addressing in this article. Our goal is to find a distribution in the model that fits the data well according to some sensible metric.

The outline of this article is as follows.

1.  Training stochastic processes using relative entropy
2.  Training stochastic processes using relative entropy rate
3.  Training latent processes without knowing the future
4.  Training latent processes with biased stochastic approximation
5.  Training Markov processes with biased stochastic approximation

In (1) and (2), we give a general overview of information-theoretic approaches to training stochastic processes. In (3), we zoom in on stochastic processes with latent variables, and measure how the lack of knowledge of the future affects online learning. In (4), we explore online algorithms involving biased stochastic approximation which may be seen as generalizations of classical expectation maximization. In (5), we zoom in on latent processes which are Markovian, and derive conditions that lead to convergence of the biased stochastic approximation.

This post is a continuation from our series on [spiking networks, path integrals and motivic information](https://shaoweilin.wordpress.com/2020/08/28/spiking-networks-path-integrals-and-motivic-information/).

### How do we train a stochastic process using relative entropy?

Suppose that we have a stochastic process $$ \{X_t\}$$ where $$ X: I \rightarrow \mathcal{X}$$ is a map from some time interval $$ I \subset \mathbb{R}$$ to some countable state space $$ \mathcal{X}$$. If $$ I$$ is the continuous interval $$ [0, T],$$ then $$ X$$ is a continuous-time stochastic process. If $$ I$$ is the discrete interval $$ \{0, 1, \ldots, T\},$$ then $$ X$$ is a discrete-time stochastic process.

Let the true distribution of $$ \{X_t\}$$ be given by some path measure $$ Q_*(\gamma)$$ over paths $$ \gamma : I \rightarrow R$$. Suppose that we have a model or a family of path measures $$ P_\theta(\gamma)$$ parametrized by $$ \theta \in \Theta.$$ We also assume absolute continuity for our model, i.e. $$ Q_* \ll P_\theta$$ for all $$ \theta$$. This assumption is trivially satisfied if each model distribution assigns a non-zero measure to every cylinder in the sigma-algebra of our path space. Let $$ X_{a\ldots b}$$ denote the stochastic process over the interval $$ [a,b]$$, i.e. the family $$ \{X_t\}_{t \in [a,b]}.$$

Because we have absolute continuity, the Radon-Nikodym derivative $$ dQ_*/dP_\theta$$ exists, and we may define the time-averaged relative entropy

$$ \tag{RN}
\displaystyle \frac{1}{T} H_{Q_*\Vert P_\theta}(X_{0\ldots T}) = \frac{1}{T} \int \log \frac{dQ_*}{dP_\theta} dQ_*. $$

To find the best model distribution $$ P_\theta$$ for approximation the true distribution $$ Q_*$$, we will minimize this relative entropy with respect to $$ \theta$$. As the time horizon $$ T$$ tends to infinity, we may also consider minimizing the limit of the time-averaged relative entropy

$$ \tag{TA} \displaystyle \lim_{T\rightarrow \infty} \frac{1}{T} H_{Q_*\Vert P_\theta}(X_{0\ldots T}).$$

In equation [(TA)](https://shaoweilin.github.io/processes-relative-entropy/#mjx-eqn-TA), we see this.

