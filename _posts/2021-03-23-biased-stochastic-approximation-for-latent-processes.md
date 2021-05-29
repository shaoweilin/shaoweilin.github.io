---
layout: post
title: Biased stochastic approximation for latent processes
---

We apply biased stochastic approximation and variational inference to optimize a relative entropy objective for latent Markov processes. Using this technique, we prove under some regularity conditions that the learning algorithm converges to a local minima.

We will be using biased stochastic approximation [KMMW19] where the stochastic updates are dependent on the past but the conditional expectation of the stochastic updates given the past is not equal to the mean field. These biased stochastic approximation schemes generalize the classical expectation maximization algorithm [KMMW19].

This post is a continuation from our series on [spiking networks, path integrals and motivic information](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/).

## What is the general intuition behind online learning for latent processes?

As [before](https://shaoweilin.github.io/variational-inference-for-latent-processes/), we assume that the universe is a Markov process $$\{X_t\},$$ and let its true distribution be the path measure $$Q_*.$$

Suppose that we have a parametric discriminative model $$\{Q_\lambda : \lambda \in \Lambda\}$$ and a parametric generative model $$\{P_\theta : \theta \in \Theta\}$$ where the distributions $$Q_\lambda$$ and $$P_\theta$$ are path measures on some joint process $$\{(Z_t, X_t)\}.$$ The random variables $$Z_t$$ represent computational states in this discriminative-generative model.

We assume that in both models, the distributions are Markov and each $$Z_t$$ and $$X_t$$ are conditionally independent given their past.  We also assume that marginals $$Q(X_{0\ldots T})$$ of the discriminative model distributions $$Q_\lambda(Z_{0 \ldots T}, X_{0\ldots T})$$ are all equal to the true distribution $$Q_*(X_{0\ldots T}).$$
 
For our model, we consider a Markov process $$\{ (Z_t, X_t) \}$$ with a family of joint path measures $$P_\theta = \mathcal{M}(\mathcal{P}_\theta, \pi_\theta)$$ on $$\{ (Z_t, X_t) \},$$ where $$\mathcal{P}_\theta$$ and $$\pi_\theta$$ denote the Markov kernel and initial distribution. Since $$X_t = (Y_t, U_t)$$ with $$Y_t$$ hidden and $$U_t$$ observed, we require

$$\begin{array}{rl} & P_\theta(Z_{t+1}, Y_{t+1}, U_{t+1} \vert Z_t, Y_t, U_t) \\ & \\ & = Q_*(Y_{t+1} \vert Y_t, U_t) P_\theta(Z_{t+1} \vert Z_t, U_t) P_\theta(X_{t+1} \vert Z_t, U_t) \end{array}$$

so the parameter $$\theta$$ controls only the distribution on $$\{(Z_t, U_t)\}.$$

As before, let $$\Delta_{\mathcal{C}}$$ be the set of all path measures $$Q$$ on $$\{ (Z_t, X_t) \}$$ where $$Z_t$$ and $$X_t$$ are conditionally independent given their past and where each $$X_t$$ is conditionally independent of $$Z_{0\ldots (t-1)}$$ given its own past $$X_{0\ldots (t-1)}.$$ We will further consider a subspace $$\Delta_\mathcal{M} \subset \Delta_\mathcal{C}$$ of distributions which are Markov. Note that our model $$\{ P_\theta \}$$ need not fill the subspace $$\Delta_\mathcal{M}.$$

Our goal is to train the model by minimizing the limit of the time-averaged relative entropy

$$V(Q,\theta) = \displaystyle \lim_{T \rightarrow \infty} \frac{1}{T} H_{Q \Vert P_\theta}(Z_{0\ldots T},X_{0\ldots T})$$

over $$Q \in \Delta_\mathcal{M}$$ and $$\theta \in \Theta.$$ Compared to minimizing this objective for $$Q$$ over the larger space $$\Delta_\mathcal{C},$$ we will incur an additional cost due to the Markov constraint. We can think of this cost as the cost of _limited memory_ since the Markov property only allows us to remember the most recent state $$(Z_t, X_t)$$ as opposed to the full history $$Z_{0\ldots t}, X_{0\ldots t}.$$

With this Markov constraint, the optimal distribution $$Q$$ in Step 1 of the algorithm in the previous section will no longer satisfy

$$Q(Z_{n+1} \vert Z_{0\ldots n}, X_{0\ldots n}) = P_{\theta_n}(Z_{n+1} \vert Z_{0\ldots n}, X_{0\ldots n}).$$

Instead, we assume some discriminative model updates $$Q_{n+1} = F(Q_n, \theta_{n+1})$$ in addition to the generative model updates to $$P_{\theta_n}$$ to tackle the optimization problem in Step 1\.





Suppose the generative model $$\{P_\theta : \theta \in \Theta\}$$ and the discriminative model $$\{Q_\lambda :\lambda \in \Lambda\}$$ are parametric. 

We now focus on minimizing the relative entropy rate over $$Q_\lambda \in \Delta_\mathcal{M}$$ and $$P_\theta \in \Delta_\mathcal{M}.$$  We first explore the problem in discrete time, before discussing the analogous results in continuous time.

The discrete time analogue of the relative entropy rate under our Markov setting is the asymptotic conditional relative entropy $$\lim_{n \rightarrow \infty} H_{Q \Vert P}(Z_{n+1}, X_{n+1} \vert Z_{n}, X_{n}).$$

We assume that $$Q_\lambda$$ has a stationary distribution $$\bar{\pi}_\lambda,$$ and let $$\bar{Q}_\lambda$$ be the distribution of a Markov chain that has the same transition probabilities as $$Q_\lambda$$ but has the initial distribution $$\bar{\pi}_\lambda.$$ Then,

$$\lim_{n \rightarrow \infty} H_{Q_\lambda \Vert P_\theta}(Z_{n+1}, X_{n+1} \vert Z_{n}, X_{n}) = H_{\bar{Q}_\lambda \Vert \bar{P}_\theta}(Z_{1}, X_{1} \vert Z_{0}, X_{0})$$

where $$\bar{P}_\theta$$ is the distribution of any Markov chain that has the same transition probabilities as $$P_\theta.$$ The initial distribution of $$\bar{P}_\theta$$ does not matter in the above conditional relative entropy.

To minimize the conditional relative entropy objective, we adopt an approach similar to the expectation-maximization (EM) or exponential-mixture (em) [algorithm](https://shaoweilin.github.io/machine-learning-with-relative-entropy/). More precisely, we iteratively optimize for the discriminative model distribution $$Q_\lambda$$ and for the generative model distribution $$P_\theta$$ while holding the other constant. 

First, we pick some initial generative model distribution $$P_{\theta_0}$$ and discriminative model distribution $$Q_{\lambda_0}.$$ Then, for $$n = 0, 1, \ldots,$$ we repeat the next two steps.

----

**Step 1 (discriminative model update).** Fixing the generative model distribution $$P_{\theta_n},$$ minimize $$H_{\bar{Q}_\lambda \Vert \bar{P}_{\theta_n}}(Z_{1}, X_{1} \vert Z_{0}, X_{0})$$ over discriminative model distributions $$Q_\lambda.$$

Because $$Z_{1}$$ and $$X_{1}$$ are conditionally independent given the past,

$$\begin{array}{rl} & H_{\bar{Q}_\lambda \Vert \bar{P}_\theta}(Z_{1}, X_{1} \vert Z_{0}, X_{0}) \\ & \\ &= H_{\bar{Q}_\lambda \Vert \bar{P}_\theta}(Z_{1} \vert Z_{0}, X_{0}) + H_{\bar{Q}_\lambda \Vert \bar{P}_\theta}(X_{1} \vert Z_{0}, X_{0}). \end{array}$$

The second term is independent of $$\lambda$$ because it depends only on the true distribution $$Q_*$$ of the observables. 

We update the parameter $$\lambda$$ using the gradient

$$\lambda_{n+1} = \displaystyle \lambda_n - \eta_{n+1} \left.\frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert \bar{P}_{\theta_n}}(Z_{1} \vert Z_{0}, X_{0})\right\vert _{\lambda = \lambda_n}$$

where, as shown in the [appendix](https://shaoweilin.github.io/biased-stochastic-approximation-for-latent-processes/#appendix-discriminative-model-update), we have

$$\begin{array}{rl} &
\displaystyle \frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert \bar{P}_\theta}(Z_{1} \vert Z_{0}, X_{0}) 
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+1)},X_{0..(T+1)})} \Bigg[ \left( \log \frac{Q_\lambda(Z_{T+1}\vert Z_{T},X_{T})}{P_\theta(Z_{T+1}\vert Z_{T},X_{T})} \right) \,\,\times 
\\ & \\ & \quad\quad \displaystyle \sum_{t=0}^{T} \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \Bigg]
. \end{array}$$

Following [BB1] and [KMMW19], we approximate this gradient with the following numerically stable estimator

$$\begin{array}{rl} &
\displaystyle \frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert \bar{P}_\theta}(Z_{1} \vert Z_{0}, X_{0}) 
\\ & \\ & \approx 
\displaystyle \left( \log \frac{Q_\lambda(Z_{T+1}\vert Z_{T},X_{T})}{P_\theta(Z_{T+1}\vert Z_{T},X_{T})} \right) \sum_{t=0}^{T} \gamma^{T-t} \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t})
. \end{array}$$

where $$0 < \gamma < 1$$ is a discount factor, $$T$$ is sufficiently large and $$Z_{0\ldots (T+1)}, X_{0\ldots (T+1)}$$ is drawn from $$Q_{\lambda_n}.$$

**Step 2 (generative model update).** Fixing the discriminative model distribution $$Q_{\lambda_{n+1}}(Z_{1} \vert Z_{0}, X_{0}),$$ minimize $$H_{\bar{Q}_{\lambda_{n+1}}\Vert \bar{P}_{\theta}}(Z_{1}, X_{1} \vert Z_{0}, X_{0})$$ over generative model distributions $$P_{\theta}$$.

By definition,

$$\begin{array}{rl} & 
H_{\bar{Q}_{\lambda_{n+1}}\Vert \bar{P}_{\theta}}(Z_{1}, X_{1} \vert Z_{0}, X_{0})
\\ & \\ & = 
\mathbb{E}_{\bar{Q}_{\lambda_{n+1}}} [\log Q_{\lambda_{n+1}}(Z_{1}, X_{1} \vert Z_{0},X_{0})] 
\\ & \\ & 
\quad - \mathbb{E}_{\bar{Q}_{\lambda_{n+1}}} [\log P_\theta(Z_{1}, X_{1} \vert Z_{0},X_{0})], 
\end{array}$$

where we note that the first term is independent of $$\theta$$.

We update the parameter $$\theta$$ using the gradient

$$\displaystyle \theta_{n+1} = \theta_n + \eta_{n+1} \mathbb{E}_{\bar{Q}_{\lambda_{n+1}}} \left[\left.\frac{d}{d\theta} \log P_\theta(Z_{1}, X_{1} \vert Z_{0},X_{0})\right\vert _{\theta = \theta_n}\right].$$

----

Moreover, we could derive a stochastic approximation of the above two-step procedure by sampling $$Z_{n+1}$$ and $$X_{n+1}$$ from $$Q_{\lambda_{n+1}}(Z_{n+1}, X_{n+1} \vert Z_{n},X_{n}).$$

Specifically, we have

$$X_{n+1} \sim Q_*(X_{n+1} \vert X_{0\ldots n})$$

$$Z_{n+1} \sim Q_n(Z_{n+1} \vert Z_{0\ldots n}, X_{0\ldots n})$$

$$\displaystyle \theta_{n+1} = \theta_n + \eta_{n+1} \left.\frac{d}{d\theta} \log P_\theta(Z_{n+1}, X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}) \right\vert _{\theta = \theta_n}$$

$$\begin{array}{rl} & Q_{n+1}(Z_{n+2} \vert Z_{0\ldots (n+1)}, X_{0\ldots (n+1)}) \\ & \\ & = P_{\theta_n}(Z_{n+2} \vert Z_{0\ldots (n+1)}, X_{0\ldots (n+1)}). \end{array}$$


The derivative of this relative entropy rate will be the mean field of the stochastic approximations, while the stochastic updates will be biased estimates of this mean field where the bias depends on the past $$Z_{0\ldots n}, X_{0\ldots n}.$$ Unfortunately, we cannot apply the standard stochastic approximation theory of Robbins and Monro with relative entropy as the optimization objective because of this bias.


## How do we prove convergence using biased stochastic approximation?

In [KMMW19], the authors studied biased stochastic approximation in the case where the stochastic updates are Markovian.

In discrete time, to find the optimal model distribution $$P_\theta,$$ we apply the biased stochastic approximation scheme from the previous section. First, we initialize $$\theta_0$$ and $$Q_0$$ randomly, and sample

$$(Y_0, U_0) \sim Q_*(Y_0, U_0),$$

$$Z_0 \sim P_{\theta_0}(Z_0).$$

Then, for all $$n \geq 0,$$

$$(Y_{n+1}, U_{n+1}) \sim Q_*(Y_{n+1}, U_{n+1} \vert Y_n, U_n),$$

$$Z_{n+1} \sim Q_n(Z_{n+1} \vert Z_n, U_n),$$

$$\begin{array}{rl} \displaystyle \theta_{n+1} &= \displaystyle \theta_n + \eta_{n+1} \left. \frac{d}{d\theta} \log P_\theta(Z_{n+1}, U_{n+1} \vert Z_n, U_n) \right|_{\theta = \theta_n}, \end{array}$$

$$Q_{n+1} = F(Q_n, \theta_{n+1}).$$

Here, each $$Y_n$$ is unobserved and is not used in the $$\theta_n$$ updates, but we write it out explicitly above for use in our convergence analysis.

Note that in the previous section, the proposed update $$F(Q,\theta) \in \Delta_\mathcal{M}$$ was the distribution defined by

$$\begin{array}{rl} & F(Q,\theta)(Z_{0\ldots T}, X_{0\ldots T}) \\ & \\ & = Q_*(X_0) P_\theta(Z_0) \\ & \\ & \quad Q_*(X_1 \vert X_0) P_\theta(Z_1 \vert Z_0, U_0) \cdots \\ & \\ & \quad Q_*(X_T \vert X_{T-1}) P_\theta(Z_T \vert Z_{T-1}, U_{T-1}). \end{array}$$

This update depends only on $$Q_*$$ but not on $$Q$$. In general, we could consider more elaborate updates which do depend on $$Q.$$

In continuous time, the above biased stochastic approximation scheme becomes a Markov process where the true distribution is time-homogeneous but the model distribution is time-inhomogeneous. The Markov kernel $$\mathcal{P}_\theta$$ of the model distribution changes with continuous-time parameter updates

$$\displaystyle \frac{d\theta}{dt} = - \eta_t \frac{d}{d\theta} \frac{d}{ds}H_{Q \Vert P_\theta} (Z_{0\ldots s}, X_{0\ldots s})$$

that are driven by the derivative of the relative entropy rate.

We will now derive sufficient conditions on the model and true distributions for applying the convergence [theorem](https://shaoweilin.github.io/biased-stochastic-approximation/) for biased stochastic approximation.

Let $$\{W_n\}$$ be the $$Q_n$$-controlled Markov process with

$$W_0 = (Z_0, X_0, -, -),$$

$$\begin{array}{rl} W_{n+1} & = (Z_{n+1},X_{n+1}, Z_n, X_n) \\ & \\ & \in \mathcal{W} := \mathcal{Z} \times \mathcal{X} \times \mathcal{Z} \times \mathcal{X} \end{array}$$

for all $$n \geq 0,$$ whose distribution $$P_{Q_n,\theta_n}$$ is given by

$$P_{Q_n,\theta_n}(W_{n+1} \vert W_n) = Q_n (Z_{n+1}, X_{n+1} \vert Z_n, X_n). $$

We write the parameter updates as

$$\displaystyle \theta_{n+1} = \theta_n - \eta_{n+1} G_{Q_n, \theta_n}(W_{n+1}),$$

$$G_{Q_n, \theta_n}(W_{n+1}) = \displaystyle -\left. \frac{d}{d\theta} \log P_\theta(Z_{n+1}, X_{n+1} \vert Z_n, X_n) \right|_{\theta = \theta_n},$$

$$Q_{n+1} = F(Q_n, \theta_{n+1}).$$

Suppose that the Markov kernel $$\mathcal{P}_{Q_n}$$ of $$Q_n$$ has a unique stationary distribution $$\bar{\pi}_{Q_n}.$$ The mean field of $$G_{Q_n, \theta_n}(W_{n+1})$$ is given by

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

$$\frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert \bar{P}_\theta}(Z_{1} \vert Z_{0}, X_{0})$$

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
\displaystyle \frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert \bar{P}_\theta}(Z_{1} \vert Z_{0}, X_{0}) 
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
\displaystyle \frac{d}{d\lambda} H_{\bar{Q}_\lambda \Vert \bar{P}_\theta}(Z_{1} \vert Z_{0}, X_{0}) 
\\ & \\ &
= \displaystyle \lim_{T\rightarrow \infty} \mathbb{E}_{Q_\lambda(Z_{0..(T+2)},X_{0..(T+2)})} \Bigg[ \left( \log \frac{Q_\lambda(Z_{T+2}\vert Z_{T+1},X_{T+1})}{P_\theta(Z_{T+2}\vert Z_{T+1},X_{T+1})} \right) \,\,\times 
\\ & \\ & \quad\quad \displaystyle \sum_{t=0}^{T+1} \frac{d}{d\lambda} \log Q_\lambda(Z_{t+1} \vert  Z_{t},X_{t}) \Bigg]
. \end{array}$$

## References

[BB01] Baxter, Jonathan, and Peter L. Bartlett. "Infinite-horizon policy-gradient estimation." _Journal of Artificial Intelligence Research_ 15 (2001): 319-350.

[L92] Leroux, Brian G. "Maximum-likelihood estimation for hidden Markov models." _Stochastic processes and their applications_ 40, no. 1 (1992): 127-143.

[S01] Sato, Masa-Aki. "Online model selection based on the variational Bayes." _Neural computation_ 13, no. 7 (2001): 1649-1681.

[KMMW19] Karimi, Belhal, Blazej Miasojedow, Éric Moulines, and Hoi-To Wai. "Non-asymptotic analysis of biased stochastic approximation scheme." _arXiv preprint arXiv:1902.00629_ (2019).