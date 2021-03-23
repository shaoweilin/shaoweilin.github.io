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

$$ \tag{LM} \displaystyle \lim_{T\rightarrow \infty} \frac{1}{T} H_{Q_*\Vert P_\theta}(X_{0\ldots T}).$$

### How do we train a stochastic process using relative entropy rate?

In discrete time, by the chain rule, the relative entropy decomposes into the sum

$$ \begin{array}{rl} H_{Q \Vert P}(X_{0\ldots n}) &= H_{Q \Vert P}(X_n \vert X_{0\ldots (n-1)}) \\ & \\ & \quad + H_{Q \Vert P}(X_{n-1} \vert X_{0\ldots (n-2)}) \\ & \\ & \quad + \cdots + H_{Q \Vert P}(X_1 \vert X_0) +H_{Q \Vert P}( X_0). \end{array}$$

In continuous time, we have the analogous result

$$ \displaystyle H_{Q \Vert P}(X_{0\ldots T}) = \int_0^T \frac{d}{ds}H_{Q \Vert P}(X_{0\ldots s}) \,ds $$

where

$$ \begin{array}{rl} &\displaystyle \frac{d}{ds}H_{Q \Vert P} (X_{0\ldots s}) \\ & \\ &= \lim_{\delta \rightarrow 0} \frac{1}{\delta} \left[ H_{Q \Vert P} (X_{0\ldots s+\delta}) - H_{Q \Vert P} (X_{0\ldots s})\right] \\ & \\ &= \lim_{\delta \rightarrow 0} \frac{1}{\delta} H_{Q \Vert P} (X_{s\ldots s+\delta}\vert X_{0\ldots s}) \end{array}$$

is also known as the _relative entropy rate_ of $$ X_t$$. The differential operator acting on the function $$ H_{Q \Vert P}$$ of the path $$ X_{0\ldots s}$$ may be seen as a generalization of the infinitesimal generator for Markov processes.

In fact, using Kingman's subadditive ergodic theory [Leroux92], we can show that under strong stationarity conditions,

$$ \displaystyle \lim_{n\rightarrow \infty} \frac{1}{n} H_{Q \Vert P}(X_{0\ldots n}) = \lim_{n\rightarrow \infty} H_{Q \Vert P}(X_n \vert X_{0\ldots (n-1)}).$$

In continuous time, we have the corresponding relation

$$ \displaystyle \lim_{T\rightarrow \infty} \frac{1}{T} H_{Q \Vert P}(X_{0\ldots T}) = \lim_{T\rightarrow \infty} \frac{d}{dT}H_{Q \Vert P}(X_{0\ldots T}).$$

This gives us an asymptotic relationship between relative entropy and relative entropy rate.

Hence, to learn a strongly-stationary stochastic process, instead of minimizing the relative entropy to $$ Q_*$$ from $$ P_\theta$$, we may minimize their relative entropy rate. We will make this idea rigorous later in the article.

Note that the relative entropy rate does not depend on the initial distributions $$ Q(X_0)$$ and $$ P(X_0).$$ Therefore, by using the time-averaged relative entropy (or equivalently the relative entropy rate) as the learning objective, we are declaring that our main focus is learning the transition kernel of the true distribution.

I prefer to write the relative entropy rate as

$$ \displaystyle \frac{d}{ds}H_{Q \Vert P} (X_{0\ldots s}) = \frac{d X_{0\ldots s}}{ds} * \frac{\partial}{\partial X_{0\ldots s}} H_{Q \Vert P} (X_{0\ldots s})$$

where $$ *$$ is the inner product in some appropriate Hilbert space and

$$ \displaystyle \frac{\partial}{\partial X_{0\ldots s}} F (X_{0\ldots s})$$

is the functional derivative of a path function $$ F$$ with respect to the path $$ X_{0\ldots s}.$$ We will formalize the Hilbert space and functional derivative in another article.

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

Following the expectation-maximization strategy outlined [previously](https://shaoweilin.wordpress.com/2020/10/23/machine-learning-with-relative-entropy/), we consider some space of joint distributions on $$ Z_t, X_t$$ which extend the true distribution $$ Q_*$$, and attempt to minimize the joint relative entropy

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

so a minimum value for $$ H_{Q\Vert P_\theta}(Z_{0\ldots T},X_{0\ldots T})$$ will be a lower bound for the minimum value for $$ H_{Q_*\Vert P_\theta}(X_{0\ldots T}).$$ As shown [previously](https://shaoweilin.wordpress.com/2020/10/23/machine-learning-with-relative-entropy/), when minimizing over the space $$ \Delta$$ of all possible joint distributions, the gap $$ H_{Q\Vert P_\theta}(Z_{0\ldots T}\vert X_{0\ldots T})$$ in the bound vanishes.

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

### How do we train a Markov process using biased stochastic approximation?

In [KMMW19], the authors studied biased stochastic approximation in the case where the stochastic updates are Markovian.

Suppose that the environment is a Markov process $$ \{ X_t \} = \{ (Y_t, U_t) \}$$ where only $$ U_t$$ is observed, and let the true distribution be the joint path measure $$ Q_*$$ on $$ (Y_t, U_t)$$. We write $$ Q_* = \mathcal{M}(\mathcal{P}_{Q_*}, \pi_{Q_*})$$ where $$ \mathcal{P}_{Q_*}$$ is the Markov kernel and $$ \pi_{Q_*}$$ is the initial distribution.

For our model, we consider a Markov process $$ \{ (Z_t, X_t) \}$$ with a family of joint path measures $$ P_\theta = \mathcal{M}(\mathcal{P}_\theta, \pi_\theta)$$ on $$ \{ (Z_t, X_t) \},$$ where $$ \mathcal{P}_\theta$$ and $$ \pi_\theta$$ denote the Markov kernel and initial distribution. Since $$ X_t = (Y_t, U_t)$$ with $$ Y_t$$ hidden and $$ U_t$$ observed, we require

$$ \begin{array}{rl} & P_\theta(Z_{t+1}, Y_{t+1}, U_{t+1} \vert Z_t, Y_t, U_t) \\ & \\ & = Q_*(Y_{t+1} \vert Y_t, U_t) P_\theta(Z_{t+1} \vert Z_t, U_t) P_\theta(X_{t+1} \vert Z_t, U_t) \end{array}$$

so the parameter $$ \theta$$ controls only the distribution on $$ \{(Z_t, U_t)\}.$$

As before, let $$ \Delta_{\mathcal{C}}$$ be the set of all path measures $$ Q$$ on $$ \{ (Z_t, X_t) \}$$ where $$ Z_t$$ and $$ X_t$$ are conditionally independent given their past and where each $$ X_t$$ is conditionally independent of $$ Z_{0\ldots (t-1)}$$ given its own past $$ X_{0\ldots (t-1)}.$$ We will further consider a subspace $$ \Delta_\mathcal{M} \subset \Delta_\mathcal{C}$$ of distributions which are Markov. Note that our model $$ \{ P_\theta \}$$ need not fill the subspace $$ \Delta_\mathcal{M}.$$

Our goal is to train the model by minimizing the limit of the time-averaged relative entropy

$$ V(Q,\theta) = \displaystyle \lim_{T \rightarrow \infty} \frac{1}{T} H_{Q \Vert P_\theta}(Z_{0\ldots T},X_{0\ldots T})$$

over $$ Q \in \Delta_\mathcal{M}$$ and $$ \theta \in \Theta.$$ Compared to minimizing this objective for $$ Q$$ over the larger space $$ \Delta_\mathcal{C},$$ we will incur an additional cost due to the Markov constraint. We can think of this cost as the cost of _limited memory_ since the Markov property only allows us to remember the most recent state $$ (Z_t, X_t)$$ as opposed to the full history $$ Z_{0\ldots t}, X_{0\ldots t}.$$

With this Markov constraint, the optimal distribution $$ Q$$ in Step 1 of the algorithm in the previous section will no longer satisfy

$$ Q(Z_{n+1} \vert Z_{0\ldots n}, X_{0\ldots n}) = P_{\theta_n}(Z_{n+1} \vert Z_{0\ldots n}, X_{0\ldots n}).$$

Instead, we assume some functional updates $$ Q_{n+1} = F(Q_n, \theta_{n+1})$$ in addition to the $$ \theta_n$$ updates to tackle the optimization problem in Step 1\.

In discrete time, to find the optimal model distribution $$ P_\theta,$$ we apply the biased stochastic approximation scheme from the previous section. First, we initialize $$ \theta_0$$ and $$ Q_0$$ randomly, and sample

$$ (Y_0, U_0) \sim Q_*(Y_0, U_0),$$

$$ Z_0 \sim P_{\theta_0}(Z_0).$$

Then, for all $$ n \geq 0,$$

$$ (Y_{n+1}, U_{n+1}) \sim Q_*(Y_{n+1}, U_{n+1} \vert Y_n, U_n),$$

$$ Z_{n+1} \sim Q_n(Z_{n+1} \vert Z_n, U_n),$$

$$ \begin{array}{rl} \displaystyle \theta_{n+1} &= \displaystyle \theta_n + \eta_{n+1} \left. \frac{d}{d\theta} \log P_\theta(Z_{n+1}, U_{n+1} \vert Z_n, U_n) \right|_{\theta = \theta_n}, \end{array}$$

$$ Q_{n+1} = F(Q_n, \theta_{n+1}).$$

Here, each $$ Y_n$$ is unobserved and is not used in the $$ \theta_n$$ updates, but we write it out explicitly above for use in our convergence analysis.

Note that in the previous section, the proposed update $$ F(Q,\theta) \in \Delta_\mathcal{M}$$ was the distribution defined by

$$ \begin{array}{rl} & F(Q,\theta)(Z_{0\ldots T}, X_{0\ldots T}) \\ & \\ & = Q_*(X_0) P_\theta(Z_0) \\ & \\ & \quad Q_*(X_1 \vert X_0) P_\theta(Z_1 \vert Z_0, U_0) \cdots \\ & \\ & \quad Q_*(X_T \vert X_{T-1}) P_\theta(Z_T \vert Z_{T-1}, U_{T-1}). \end{array}$$

This update depends only on $$ Q_*$$ but not on $$ Q$$. In general, we could consider more elaborate updates which do depend on $$ Q.$$

In continuous time, the above biased stochastic approximation scheme becomes a Markov process where the true distribution is time-homogeneous but the model distribution is time-inhomogeneous. The Markov kernel $$ \mathcal{P}_\theta$$ of the model distribution changes with continuous-time parameter updates

$$ \displaystyle \frac{d\theta}{dt} = - \eta_t \frac{d}{d\theta} \frac{d}{ds}H_{Q \Vert P_\theta} (Z_{0\ldots s}, X_{0\ldots s})$$

that are driven by the derivative of the relative entropy rate.

We will now derive sufficient conditions on the model and true distributions for applying the convergence [theorem](https://shaoweilin.wordpress.com/2020/12/01/biased-stochastic-approximation/) for biased stochastic approximation.

Let $$ \{W_n\}$$ be the $$ Q_n$$-controlled Markov process with

$$ W_0 = (Z_0, X_0, -, -),$$

$$ \begin{array}{rl} W_{n+1} & = (Z_{n+1},X_{n+1}, Z_n, X_n) \\ & \\ & \in \mathcal{W} := \mathcal{Z} \times \mathcal{X} \times \mathcal{Z} \times \mathcal{X} \end{array}$$

for all $$ n \geq 0,$$ whose distribution $$ P_{Q_n,\theta_n}$$ is given by

$$ P_{Q_n,\theta_n}(W_{n+1} \vert W_n) = Q_n (Z_{n+1}, X_{n+1} \vert Z_n, X_n). $$

We write the parameter updates as

$$ \displaystyle \theta_{n+1} = \theta_n - \eta_{n+1} G_{Q_n, \theta_n}(W_{n+1}),$$

$$ G_{Q_n, \theta_n}(W_{n+1}) = \displaystyle -\left. \frac{d}{d\theta} \log P_\theta(Z_{n+1}, X_{n+1} \vert Z_n, X_n) \right|_{\theta = \theta_n},$$

$$ Q_{n+1} = F(Q_n, \theta_{n+1}).$$

Suppose that the Markov kernel $$ \mathcal{P}_{Q_n}$$ of $$ Q_n$$ has a unique stationary distribution $$ \bar{\pi}_{Q_n}.$$ The mean field of $$ G_{Q_n, \theta_n}(W_{n+1})$$ is given by

$$ \begin{array}{rl} g(Q_n,\theta_n) & =- \displaystyle \int \bar{\pi}_{Q_n} (dZ_0, dX_0) Q_n(dZ_1, dX_1 \vert Z_0, X_0) \\ & \\ & \quad \displaystyle \left. \frac{d}{d\theta}\log P_\theta(Z_{1}, X_{1} \vert Z_0, X_0) \right|_{\theta = \theta_n} \\ & \\ & = \displaystyle \frac{d}{d\theta} \int \bar{\pi}_{Q_n} (dZ_0, dX_0) Q_n(dZ_1, dX_1 \vert Z_0, X_0) \\ & \\ & \quad \displaystyle \left. \log \frac{Q_n(Z_{1}, X_{1} \vert Z_0, X_0)}{\mathcal{P}_\theta(Z_{1}, X_{1} \vert Z_0, X_0)}\right|_{\theta = \theta_n} \\ & \\ & = \displaystyle \left. \frac{d}{d\theta} H_{\mathcal{M}(\mathcal{P}_{Q_n},\bar{\pi}_{Q_n}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1}, X_{1} \vert Z_0, X_0) \right|_{\theta = \theta_n} \end{array}$$

where $$ \mathcal{M}(\mathcal{P},\bar{\pi})$$ denotes the Markov distribution with transition kernel $$ \mathcal{P}$$ and initial distribution $$ \bar{\pi}$$; and where $$ \mathcal{M}(\mathcal{P}, -)$$ is any Markov distribution with transition kernel $$ \mathcal{P}$$ if the initial distribution is irrelevant.

We now define the Lyapunov function

$$ \begin{array}{rl} \displaystyle V(Q,\theta) & := \displaystyle \lim_{T \rightarrow \infty} \frac{1}{T} H_{Q \Vert P_\theta}(Z_{0\ldots T},X_{0\ldots T}) \\ & \\ & = \displaystyle \lim_{T \rightarrow \infty} H_{Q \Vert P_\theta}(Z_T, X_T \vert Z_{0\ldots (T-1)},X_{0\ldots (T-1)}) \\ & \\ & = \displaystyle \lim_{T \rightarrow \infty} H_{Q \Vert P_\theta}(Z_T, X_T \vert Z_{T-1},X_{T-1}) \\ & \\ & = \displaystyle H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1}, X_{1} \vert Z_0, X_0) \end{array}$$

where $$ \mathcal{P}_{Q}$$ and $$ \bar{\pi}_{Q}$$ are respectively the transition kernel and the unique stationary distribution of $$ Q.$$ Here, the second equality is the asymptotic relationship between relative entropy and relative entropy rate; the third equality follows from the Markov property on $$ Q$$ and $$ P_\theta$$.

Therefore, the mean field satisfies

$$ g(Q_n, \theta_n) = \displaystyle \frac{\partial V}{\partial \theta}(Q_n, \theta_n),$$

so assumptions A1 and A2 of the [convergence theorem](https://shaoweilin.wordpress.com/2020/12/15/training-spiking-neural-networks-with-latent-variables/) are automatically satisfied.

Now, keeping $$ \theta$$ fixed, let us minimize $$ V(Q,\theta)$$ over $$ Q \in \Delta_\mathcal{M}$$ so that we may update $$ Q_n$$. In the training algorithm, we can think of $$ Q_n(Z_{t+1}\vert Z_t, X_t)$$ as the functional parameter or distribution controlling a sampler that spits out possible explanations $$ Z_{t+1}$$ of the observations $$ X_t$$ based on prior knowledge $$ Z_t.$$

Since

$$ \begin{array}{rl} V(Q,\theta) &= H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1}, X_{1} \vert Z_0, X_0) \\ & \\ & = H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1} \vert Z_0, X_0) \\ & \\ & \quad + H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (X_{1} \vert Z_0, X_0) \\ & \\ & = H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1} \vert Z_0, X_0) \\ & \\ & \quad + \displaystyle \int \bar{\pi}_{Q} (dZ_0, dX_0) H_{Q_*(X_1 \vert X_0) \Vert \mathcal{P}_\theta(X_1 \vert Z_0, X_0)} (X_{1}) \\ & \\ & = H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1} \vert Z_0, X_0) \\ & \\ & \quad + \displaystyle \int \bar{\pi}_{Q} (dX_0) \bar{\pi}_{Q} (dZ_0 \vert X_0) H_{Q_*(X_1 \vert X_0) \Vert \mathcal{P}_\theta(X_1 \vert Z_0, X_0)} (X_{1}) , \end{array}$$

where $$ \bar{\pi}_Q$$ is the stationary distribution for the true distribution $$ Q$$ on $$ \{X_t\}$$, we may extinguish the first term by setting

$$ Q(Z_1 \vert Z_0, X_0) = P_\theta(Z_1 \vert Z_0, X_0)$$

which was proposed in the previous section. This first term represents the sampler's incentive to _exploit_ the current model distribution $$ P_\theta(Z_1 \vert Z_0, X_0)$$ for most likely explanations of the observations.

As for the second term, it decreases if for each $$ X_0$$, we assign a larger weight $$ \bar{\pi}_{Q}(dZ_0 \vert X_0)$$ to $$ Z_0$$ where the relative entropy $$ H_{Q_*(X_1 \vert X_0) \Vert \mathcal{P}_\theta(X_1 \vert Z_0, X_0)} (X_{1})$$ is smaller. In other words, the second term shrinks if for each observation $$ X_0,$$ the stationary distribution assigns a larger probability to hidden states $$ Z_0$$ which can explain the next observation $$ X_1$$ under the given model distribution $$ P_\theta.$$ This second term represents the sampler's incentive to _explore_ new explanations which fit the observations better as evaluated by $$ P_\theta(X_1 \vert Z_0, X_0)$$, even if those new explanations are less probable under $$ P_\theta(Z_1 \vert Z_0, X_0)$$.

In the previous section, we chose a sampler $$ F(-,\theta)$$ which has a strong tendency to exploit. In practical applications, the design of a good sampler $$ F(Q,\theta)$$ may depend on the application, or on some additional mechanism to compare predictions against explanations, or on some kind of curriculum to guide the learning machine so that it may arrive quickly at the most useful explanations.

In this section, we will just assume that the Lyapunov function does not increase under the update $$ Q_{n+1} = F(Q_n, \theta_{n+1})$$. The update could for instance compare $$ V(Q_n,\theta_{n+1})$$ against $$ V(Q_{n+1}, \theta_{n+1})$$ for some proposal $$ Q_{n+1},$$ and only accept the proposal if there is an improvement.

We now study the Poisson equation

$$ L_{Q,\theta} \hat{E}_{Q, \theta} (w_0) = E_{Q,\theta}(w_0)$$

where $$ w_0 = (z_1, x_1, z_0, x_0),$$

$$ \begin{array}{rl} E_{Q,\theta}(w_0) & = \displaystyle - g(Q,\theta) - \frac{d}{d\theta}\log P_\theta(z_{1}, x_{1} \vert z_0, x_0) \\ & \\ & = - \displaystyle  \frac{d}{d\theta} \Big( V(Q,\theta)+ \log P_\theta(z_{1}, x_{1} \vert z_0, x_0) \Big) , \end{array} $$

$$ L_{Q,\theta} \hat{E}_{Q, \theta} (w_0) = \mathcal{P}_{Q,\theta} \hat{E}_{Q, \theta} (w_0)- \hat{E}_{Q, \theta} (w_0).$$

The solution of the Poisson equation is given by

$$ \hat{E}_{Q, \theta} (w_0) = - \displaystyle \lim_{n \rightarrow \infty} \sum_{k=0}^n \mathcal{P}_{Q,\theta}^k E_{Q,\theta}(w_0).$$

We observe that

$$ \begin{array}{rl} & \mathcal{P}_{Q,\theta} E_{Q,\theta}(w_0) \\ & \\ & = \displaystyle -\frac{d}{d\theta} \Bigg( V(Q,\theta) - \int Q(dZ_2, dX_2 \vert z_1, x_1) \log \frac{Q(Z_{2}, X_{2} \vert z_1, x_1)}{P_\theta(Z_{2}, X_{2} \vert z_1, x_1)} \Bigg) \\ & \\ & = \displaystyle - \frac{d}{d\theta} \Big( H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1}, X_{1} \vert Z_0, X_0) \\ & \\ & \qquad \qquad - H_{\mathcal{M}(\mathcal{P}_{Q},\pi_{z_1, x_1}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_1, X_1 \vert Z_0, X_0) \Big) \end{array}$$

where $$ \pi_{z_1, x_1}$$ is the initial distribution with $$ Z_1 = z_1, X_1 = x_1$$ almost surely. Then,

$$ \begin{array}{rl} & \mathcal{P}_{Q,\theta} \hat{E}_{Q, \theta} (w_0) \\ & \\ & = E_{Q,\theta}(w_0) + \hat{E}_{Q, \theta} (w_0) \\ & \\ & = - \displaystyle \lim_{n \rightarrow \infty} \sum_{k=0}^{n-1} \mathcal{P}_{Q,\theta}^{k+1} E_{Q,\theta}(w_0) \\ & \\ & = \displaystyle \lim_{n \rightarrow \infty} \frac{d}{d\theta} \Big( \sum_{k=0}^{n-1} H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{k+1}, X_{k+1} \vert Z_k, X_k) \\ & \\ & \quad \displaystyle - \sum_{k=0}^{n-1} H_{\mathcal{M}(\mathcal{P}_{Q},\pi_{z_1, x_1}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{k+1}, X_{k+1} \vert Z_k, X_k) \Big) \\ & \\ & = \displaystyle \lim_{n \rightarrow \infty} \frac{d}{d\theta} \Big( H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{n}, X_{n} \vert Z_0, X_0) \\ & \\ & \quad \displaystyle - H_{\mathcal{M}(\mathcal{P}_{Q},\pi_{z_1, x_1}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{n}, X_{n} \vert Z_0, X_0) \Big) \end{array}$$

Bringing them all together,

$$ \begin{array}{rl} V(Q,\theta) &= H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{1}, X_{1} \vert Z_0, X_0) \\ & \\ g(Q, \theta) & = \displaystyle \frac{\partial V}{\partial \theta}(Q, \theta) \\ & \\ E_{Q,\theta}(w_0) & = - \displaystyle  \frac{d}{d\theta} \Big( V(Q,\theta)+ \log P_\theta(z_{1}, x_{1} \vert z_0, x_0) \Big) \\ & \\ \mathcal{P}_{Q,\theta} \hat{E}_{Q, \theta} (w_0) & = \displaystyle \lim_{n \rightarrow \infty} \frac{d}{d\theta} \Big( H_{\mathcal{M}(\mathcal{P}_{Q},\bar{\pi}_{Q}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{n}, X_{n} \vert Z_0, X_0) \\ & \\ & \quad \displaystyle - H_{\mathcal{M}(\mathcal{P}_{Q},\pi_{z_1, x_1}) \Vert \mathcal{M}(\mathcal{P}_\theta, -)} (Z_{n}, X_{n} \vert Z_0, X_0) \Big) \\ & \\ \hat{E}_{Q, \theta} (w_0) & = \mathcal{P}_{Q,\theta} \hat{E}_{Q, \theta} (w_0) - E_{Q,\theta}(w_0) \end{array}$$

We impose the following regularity conditions.

<span style="text-decoration:underline;">C1 (Stationarity</span>). For all $$ Q \in \Delta_\mathcal{M}, \theta \in \Theta,$$ the Markov kernel $$ \mathcal{P}_{F(Q,\theta)}$$ has a unique stationary distribution $$ \bar{\pi}_{F(Q,\theta)}$$

<span style="text-decoration:underline;">C2 (Exploitation and Exploration</span>). For all $$ Q \in \Delta_\mathcal{M}, \theta \in \Theta,$$

$$ V(F(Q, \theta),\theta) \leq V(Q,\theta).$$

<span style="text-decoration:underline;">C3 ($$ \ell$$-smoothness</span>). There exists $$ \ell < \infty$$ such that for all $$ Q \in \Delta_\mathcal{M}, \theta, \theta' \in \Theta,$$

$$ \displaystyle \left\Vert \frac{\partial V}{\partial \theta}(Q,\theta) - \frac{\partial V}{\partial \theta}(Q,\theta') \right\Vert \leq \ell \Vert \eta - \eta' \Vert.$$

<span style="text-decoration:underline;">C4 (Regularity of solution of Poisson equation</span>). There exists $$ \ell_0, \ell_1 < \infty$$ such that for all $$ Q \in \Delta_\mathcal{M}, \theta, \theta' \in \Theta, w \in \mathcal{W},$$

$$ \Vert \hat{E}_{Q, \theta} (w) \Vert \leq \ell_0, \quad \Vert \mathcal{P}_{Q,\eta} \hat{E}_{Q,\theta}(w) \Vert \leq \ell_0,$$

$$ \Vert \mathcal{P}_{Q,\theta} \hat{E}_{Q,\theta}(w) - \mathcal{P}_{Q,\theta'} \hat{E}_{Q,\theta'} (w) \Vert \leq \ell_1 \Vert \theta - \theta' \Vert.$$

<span style="text-decoration:underline;">C5 (Boundedness of correction term</span>). There exists $$ \sigma < \infty$$ such that for all $$ Q \in \Delta_\mathcal{M}, \theta \in \Theta, w \in \mathcal{W},$$

$$ \Vert E_{Q,\theta} (w) \Vert \leq \sigma.$$

<span style="text-decoration:underline;">Theorem (Convergence of Biased Stochastic Approximation</span>). Suppose that we have state updates

$$ (Y_{k+1}, U_{k+1}) \sim Q_*(Y_{k+1}, U_{k+1} \vert Y_k, U_k),$$

$$ Z_{k+1} \sim Q_k(Z_{k+1} \vert Z_k, U_k),$$

and parameter updates

$$ \begin{array}{rl} \displaystyle \theta_{k+1} &= \displaystyle \theta_k + \eta_{k+1} \left. \frac{d}{d\theta} \log P_\theta(Z_{k+1}, U_{k+1} \vert Z_k, U_k) \right|_{\theta = \theta_k}, \end{array}$$

$$ Q_{k+1} = F(Q_k, \theta_{k+1}).$$

for $$ 0 \leq k \leq n,$$ using step sizes $$ \eta_k = \eta_0 k^{-1/2}$$ for sufficiently small $$ \eta_0 \geq 0,$$ and using a random stop time $$ 0 \leq N \leq n$$ with $$ \mathbb{P}(N = l) := (\sum_{k=0}^n \eta_{k+1})^{-1} \eta_{l+1}.$$ Then assuming C1-C5, we have

$$ \mathbb{E}(\Vert g(Q_N, \theta_N) \Vert^2) = O(\log n / \sqrt{n} ).$$

### References

[Leroux92] Leroux, Brian G. "Maximum-likelihood estimation for hidden Markov models." _Stochastic processes and their applications_ 40, no. 1 (1992): 127-143.

[Sato01] Sato, Masa-Aki. "Online model selection based on the variational Bayes." _Neural computation_ 13, no. 7 (2001): 1649-1681.

[KMMW19] Karimi, Belhal, Blazej Miasojedow, Éric Moulines, and Hoi-To Wai. "Non-asymptotic analysis of biased stochastic approximation scheme." _arXiv preprint arXiv:1902.00629_ (2019).

Testing a link to equation ([RN](https://shaoweilin.github.io/processes-relative-entropy/#mjx-eqn-RN)).
