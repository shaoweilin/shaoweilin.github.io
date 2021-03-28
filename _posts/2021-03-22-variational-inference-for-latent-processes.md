---
layout: post
title: Variational inference for latent processes
---

We introduce a variational objective, which is a form of relative entropy, for learning processes with latent variables. We discuss natural constraints on the variational parameter and the consequences of these constraints on learning.

### How do we train a latent process without prescience or probing?

In the training of hidden Markov models, the Baum-Welch algorithm or forward-backward algorithm is often used. It requires knowledge of the visible process from the start to the end of the time interval. An algorithm that only uses the visible process from the start to the present is called an online learning algorithm.

Even with _prescience_ or knowledge of the future, the Baum-Welch algorithm is only an expectation-maximization (EM) algorithm that computes an _approximate_ solution to the maximum likelihood problem. It approximates the evidence $$ P_\theta(X_{0\ldots T})$$ or marginal likelihood of the model distribution with a lower bound. The approximation can be made to go away if we could somehow _probe_ the latent variables, changing their states to measure their effects on the observed variables. We will define the concepts of _prescience_ and _probing_ more precisely below.

Suppose that we have two processes -- an observed $$ X_t \in \mathcal{X},$$ and a latent or hidden process $$ Z_t \in \mathcal{Z}$$. We may assume that $$ Z_t$$ and $$ X_t$$ are conditionally independent given their past. In discrete-time, this means that the conditional entropy satisfies

$$ H_{Q\Vert P}(Z_{t+1}| X_{t+1}, Z_{0\ldots t}, X_{0\ldots t}) = H_{Q\Vert P}(Z_{t+1} |Z_{0\ldots t}, X_{0\ldots t})$$

and vice versa. In continuous time, the conditional entropy satisfies

$$ H_{Q\Vert P}(Z_{t\ldots t+\delta}| X_{t\ldots t+\delta}, Z_{0\ldots t}, X_{0\ldots t}) \rightarrow H_{Q\Vert P}(Z_{t\ldots t+\delta} |Z_{0\ldots t}, X_{0\ldots t})$$

as $$ \delta \rightarrow 0$$ from above. In other words, $$ Z_t$$ does not lie in the causal cone of $$ X_t$$ and vice versa.

Under these conditions, the chain rule for relative entropy

$$ \begin{array}{rl} &H_{Q\Vert P}(Z_{n+1}, X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}) \\ & \\ &= H_{Q\Vert P}(Z_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}) + H_{Q\Vert P}(X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}). \end{array}$$

in discrete time has the continuous-time analogue

$$ \begin{array}{rl} & \displaystyle \frac{d}{ds}H_{Q \Vert P} (Z_{0\ldots s}, X_{0\ldots s}) \\ & \\ &= \displaystyle\frac{d Z_{0\ldots s}}{ds} * \frac{\partial}{\partial Z_{0\ldots s}} H_{Q \Vert P} (Z_{0\ldots s}, X_{0\ldots s}) \\ & \\ &\quad +\displaystyle \frac{d X_{0\ldots s}}{ds} * \frac{\partial}{\partial X_{0\ldots s}} H_{Q \Vert P} (Z_{0\ldots s}, X_{0\ldots s})\end{array}$$

which mirrors the multivariate chain rule in calculus

$$ \displaystyle \frac{d}{ds}f(z(s),x(s)) =\frac{dz}{ds} \frac{\partial}{\partial z} f(z,x) +\displaystyle \frac{dx}{ds} \frac{\partial}{\partial x} f(z,x).$$

Let $$ Q_*(X_{0\ldots T})$$ be the true distribution of the visible process $$ X_{0\ldots T}$$. Recall that our goal is to find a model $$ P_\theta$$ that minimizes the relative entropy

$$ \displaystyle H_{Q_*\Vert P_\theta}(X_{0\ldots T}).$$

Following the expectation-maximization strategy outlined [previously](https://shaoweilin.github.io/machine-learning-with-relative-entropy/), we consider some space of joint distributions on $$ Z_t, X_t$$ which extend the true distribution $$ Q_*$$, and attempt to minimize the joint relative entropy

$$ H_{Q\Vert P_\theta}(Z_{0\ldots T},X_{0\ldots T})$$

as $$ Q$$ varies over this space of extended distributions.

Let $$ \Delta$$ be the space of all joint path distributions $$ Q$$ on $$ Z_{0\ldots T}$$ and $$ X_{0\ldots T}$$ such that the marginal distribution $$ Q(X_{0\ldots T})$$ of the visible process equals the true distribution $$ Q_*(X_{0\ldots T})$$. In discrete time, we may decompose $$ Q$$ as

$$ \begin{array}{rl} & Q(Z_{0\ldots T}, X_{0\ldots T}) \\ & \\ & = Q(X_0) Q(Z_0 \vert X_0) \\ & \\ & \quad Q(X_1 \vert Z_0,X_0) Q(Z_1 \vert X_1, Z_0,X_0) \cdots \\ & \\ & \quad Q(X_T \vert Z_{0\ldots (T-1)},X_{0\ldots (T-1)}) Q(Z_T \vert X_T, Z_{0\ldots (T-1)}, X_{0\ldots (T-1)}). \end{array}$$

Note that in order for the marginal distribution $$ Q(X_{0\ldots T})$$ to match the true distribution $$ Q_*(X_{0\ldots T}),$$ the factors of the decomposition must satisfy

$$ \int Q(dZ_{0\ldots (t-1)}) Q(X_t \vert Z_{0\ldots (t-1)},X_{0\ldots (t-1)}) = Q_*(X_t \vert X_{0\ldots (t-1)}).$$

Now, let $$ \Delta_\mathcal{C} \subset \Delta$$ be the subspace of distributions where the variables $$ Z_t$$ and $$ X_t$$ are conditionally independent given their past and where each $$ X_t$$ is conditionally independent of $$ Z_{0\ldots (t-1)}$$ given its own past $$ X_{0\ldots (t-1)}.$$ Here, the subscript $$ \mathcal{C}$$ denotes the causal constraints on the distributions. Under these two requirements, for $$ Q \in \Delta_\mathcal{C}$$ we have the decomposition

$$ \begin{array}{rl} & Q(Z_{0\ldots T}, X_{0\ldots T}) \\ & \\ & = Q_*(X_0) Q(Z_0) \\ & \\ & \quad Q_*(X_1 \vert X_0) Q(Z_1 \vert Z_0,X_0) \cdots \\ & \\ & \quad Q_*(X_T \vert X_{0\ldots (T-1)}) Q(Z_T \vert Z_{0\ldots (T-1)}, X_{0\ldots (T-1)}). \end{array}$$

We may think of these factors as functionals (because they are functions of paths) parametrizing the space $$ \Delta_\mathcal{C}$$ of distributions. Note that under this decomposition, the marginal distribution $$ Q(X_{0\ldots T})$$ necessarily matches the true distribution $$ Q_*(X_{0\ldots T}).$$

By the chain rule of relative entropy,

$$ H_{Q\Vert P_\theta}(Z_{0\ldots T},X_{0\ldots T}) = H_{Q\Vert P_\theta}(Z_{0\ldots T}| X_{0\ldots T}) + H_{Q_*\Vert P_\theta}(X_{0\ldots T})$$

so a minimum value for $$ H_{Q\Vert P_\theta}(Z_{0\ldots T},X_{0\ldots T})$$ will be a lower bound for the minimum value for $$ H_{Q_*\Vert P_\theta}(X_{0\ldots T}).$$ As shown [previously](https://shaoweilin.github.io/machine-learning-with-relative-entropy/), when minimizing over the space $$ \Delta$$ of all possible joint distributions, the gap $$ H_{Q\Vert P_\theta}(Z_{0\ldots T}\vert X_{0\ldots T})$$ in the bound vanishes.

When minimizing over the subspace $$ \Delta_\mathcal{C}$$ however, this gap may not vanish. Loosely, the gap represents _the cost of not knowing the future and not probing the hidden_. Indeed, if $$ Z_t$$ and $$ X_t$$ are not conditionally independent given their past, then any dependence must come from the future; and if we are able to probe some latent variable, say $$ Y_{t-1}$$, which $$ X_t$$ depends on despite conditioning its past $$ X_{0\ldots (t-1)},$$ then we would gain additional information on the distribution of $$ X_{0\ldots T}$$. Since the gap vanishes when we optimize over distributions that allow such kinds of dependence, prescience and probing must be furnishing information for closing this gap.

Biological spiking networks need to work with the constraint of not knowing the future and not probing the hidden (in the absence of agency). For the remainder of this article, we will also work with this constraint by minimizing the joint relative entropy $$ H_{Q\Vert P_\theta}(Z_{0\ldots T},X_{0\ldots T})$$ over extensions $$ Q \in \Delta_\mathcal{C}$$ and parameters $$ \theta \in \Theta$$, instead of the visible relative entropy $$ H_{Q_*\Vert P_\theta}(X_{0\ldots T})$$.

In continuous time, the above decomposition of $$ Q \in \Delta_\mathcal{C}$$ can be written in terms of exponentials of integrals of transition rates, giving us some system of Kolmogorov equations.

### How do we train a latent process with biased stochastic approximation?

We now focus on minimizing $$ H_{Q\Vert P_\theta}(Z_{0\ldots T},X_{0\ldots T})$$ over $$ Q \in \Delta_\mathcal{C}$$ and $$ \theta \in \Theta.$$ We first explore the problem in discrete time, before discussing the analogous results in continuous time. We will be using biased stochastic approximation [KMMW19] where the stochastic updates are dependent on the past but the conditional expectation of the stochastic updates given the past is not equal to the mean field. These biased stochastic approximation schemes generalize the classical expectation maximization algorithm [KMMW19].

To gain some intuition for biased stochastic approximation, consider the following decomposition of the relative entropy objective by chain rule

$$ \begin{array}{rl} & H_{Q\Vert P_\theta}(Z_{0\ldots n},X_{0\ldots n}) \\ & \\ &= H_{Q\Vert P_\theta}(Z_{0},X_{0}) \\ & \\ & \quad + H_{Q\Vert P_\theta}(Z_1, X_1 \vert Z_{0},X_{0}) + \cdots \\ & \\ & \quad + H_{Q\Vert P_\theta}(Z_n, X_n \vert Z_{0\ldots (n-1)},X_{0\ldots (n-1)}). \end{array}$$

To minimize the relative entropy objective, we adopt the following heuristic that iteratively solves for the functional parameters $$ Q(Z_{i+1} \vert Z_{0\ldots i}, X_{0\ldots i})$$ and model distribution $$ P_\theta.$$ First, we pick some initial distribution $$ Q(Z_0)$$ and initial model distribution $$ P_{\theta_0}.$$ Then, for $$ n = 0, 1, \ldots,$$ we repeat the next two steps.

<span style="text-decoration:underline;">Step 1</span>. Fixing the functional parameters $$ Q(Z_{i+1} \vert Z_{0\ldots i}, X_{0\ldots i})$$ for $$ i = 0, \ldots, n-1$$ and model distribution $$ P_{\theta_n},$$ find a functional parameter $$ Q(Z_{n+1} \vert Z_{0\ldots n}, X_{0\ldots n})$$ to minimize $$ H_{Q\Vert P_{\theta_n}}(Z_{n+1}, X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}).$$

Now, because $$ Z_{n+1}$$ and $$ X_{n+1}$$ are conditionally independent given the past,

$$ \begin{array}{rl} &H_{Q\Vert P_{\theta_n}}(Z_{n+1}, X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}) \\ & \\ &= H_{Q\Vert P_{\theta_n}}(Z_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}) + H_{Q\Vert P_{\theta_n}}(X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}). \end{array}$$

The second term is independent of $$ Q(Z_{n+1} \vert Z_{0\ldots n}, X_{0\ldots n}),$$ and the first term vanishes when

$$ Q(Z_{n+1} \vert Z_{0\ldots n}, X_{0\ldots n}) = P_{\theta_n}(Z_{n+1} \vert Z_{0\ldots n}, X_{0\ldots n}).$$

<span style="text-decoration:underline;">Step</span> 2\. Fixing the functional parameters $$ Q(Z_{i+1} \vert Z_{0\ldots i}, X_{0\ldots i})$$ for $$ i = 0, \ldots, n,$$ find a model distribution $$ P_{\theta}$$ to minimize $$ H_{Q\Vert P_{\theta}}(Z_{n+1}, X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n})$$.

We pursue this objective using the gradient update

$$ \displaystyle \theta_{n+1} = \theta_n - \eta_{n+1} \frac{d}{d\theta} H_{Q\Vert P_\theta}(Z_{n+1}, X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}).$$

where $$ \eta_{n+1}$$ is the learning rate. Because

$$ \begin{array}{rl} & H_{Q\Vert P_\theta}(Z_{n+1}, X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}) \\ & \\ & = \mathbb{E}_{Q(Z_{0\ldots (n+1)},X_{0\ldots (n+1)})} [\log Q(Z_{n+1}, X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n})] \\ & \\ & \quad - \mathbb{E}_{Q(Z_{0\ldots (n+1)},X_{0\ldots (n+1)})} [\log P_\theta(Z_{n+1}, X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n})], \end{array}$$

and $$ Q$$ is fixed, the first term is a constant so the gradient update becomes

$$ \displaystyle - \mathbb{E}_{Q(Z_{0\ldots (n+1)},X_{0\ldots (n+1)})} \left[\left.\frac{d}{d\theta} \log P_\theta(Z_{n+1}, X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n})\right|_{\theta = \theta_n}\right].$$

Moreover, we could derive a stochastic approximation of the above procedure by sampling $$ Z_{n+1}$$ and $$ X_{n+1}$$ in Step 1 and replacing the gradient update in Step 2 with a one-sample approximation. Specifically, we have

$$ X_{n+1} \sim Q_*(X_{n+1} \vert X_{0\ldots n})$$

$$ Z_{n+1} \sim Q_n(Z_{n+1} \vert Z_{0\ldots n}, X_{0\ldots n})$$

$$ \displaystyle \theta_{n+1} = \theta_n + \eta_{n+1} \left.\frac{d}{d\theta} \log P_\theta(Z_{n+1}, X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}) \right|_{\theta = \theta_n}$$

$$ \begin{array}{rl} & Q_{n+1}(Z_{n+2} \vert Z_{0\ldots (n+1)}, X_{0\ldots (n+1)}) \\ & \\ & = P_{\theta_n}(Z_{n+2} \vert Z_{0\ldots (n+1)}, X_{0\ldots (n+1)}). \end{array}$$

Unfortunately, this online stochastic approximation only makes one pass through the decomposition of the relative entropy $$ H_{Q\Vert P_\theta}(Z_{0\ldots n},X_{0\ldots n}),$$ so we cannot apply the standard stochastic approximation theory of Robbins and Monro with relative entropy as the optimization objective.

However, ergodic theory tells that

$$ \displaystyle \lim_{n\rightarrow \infty} \frac{1}{n} H_{Q \Vert P}(Z_{0 \ldots n}, X_{0 \ldots n}) = \lim_{n\rightarrow \infty} H_{Q \Vert P}(Z_{n+1}, X_{n+1} \vert Z_{0 \ldots n}, X_{0 \ldots n})$$

so the online updates may be thought of as multiple passes at optimizing the relative entropy rate. The derivative of this relative entropy rate will be the mean field of the stochastic approximations, while the stochastic updates will be biased estimates of this mean field where the bias depends on the past $$ Z_{0\ldots n}, X_{0\ldots n}.$$

### References

[Leroux92] Leroux, Brian G. "Maximum-likelihood estimation for hidden Markov models." _Stochastic processes and their applications_ 40, no. 1 (1992): 127-143.

[Sato01] Sato, Masa-Aki. "Online model selection based on the variational Bayes." _Neural computation_ 13, no. 7 (2001): 1649-1681.

[KMMW19] Karimi, Belhal, Blazej Miasojedow, Éric Moulines, and Hoi-To Wai. "Non-asymptotic analysis of biased stochastic approximation scheme." _arXiv preprint arXiv:1902.00629_ (2019).