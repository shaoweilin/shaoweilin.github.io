---
layout: post
title: Variational inference for latent processes
---

We introduce a variational objective, which is a form of relative entropy, for learning processes with latent variables. We discuss natural constraints on the variational parameter and the consequences of these constraints on passive/active learning and online/offline learning with limited/unlimited memory.

This post is a continuation from our series on [spiking networks, path integrals and motivic information](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/).

## How do we apply variational inference to latent processes?

Suppose that we have two processes -- an observed $$X_t \in \mathcal{X},$$ and a latent or hidden process $$Z_t \in \mathcal{Z}$$. Let $$Q_*(X_{0\ldots T})$$ be the true distribution of the visible process $$X_{0\ldots T}$$. Recall that our goal is to find a generative model $$P(Z,X)$$ that minimizes the relative entropy

$$\displaystyle H_{Q_*\Vert P}(X_{0\ldots T})$$

(or equivalently the relative entropy rate if we assume strong stationarity.)

[Previously](https://shaoweilin.github.io/machine-learning-with-relative-entropy/), we discussed how this optimization problem over the space of distributions on $$X$$ may be lifted to an optimization problem over the space of distributions on $$(Z,X)$$ using variational inference. In particular, we will be considering different spaces of joint distributions $$Q(Z,X)$$ which extend the true distribution $$Q_*(X)$$, and attempt to minimize the joint relative entropy

$$H_{Q\Vert P}(Z_{0\ldots T},X_{0\ldots T})$$

as $$Q$$ varies over some space of extended distributions.

## Is variational inference necessarily passive?

The form of variational inference we are considering here is necessarily passive. This means that the true distribution $$Q_*(X_{0\ldots T})$$ is unaffected by the learning agent's estimates of the states of the latent variables $$Z_{0\ldots T}.$$ 

However, it is possible to design variational methods for models where the estimates of the latent variables have an effect on the observed data, such as in active learning, reinforcement learning or partially-ordered Markov decision processes. The estimates could lead a learning agent to effect some change on the environment which then affects the observed data. Unfortunately, we will not be considering this active form of variational inference here. A complete understanding of biological intelligence must however analyze the active case.

In our passive form of variational inference, the most general case for the space of joint path distributions $$Q(Z_{0\ldots T}, X_{0\ldots T})$$ is the unconstrained space $$\Delta$$ where $$Q$$ factors as 

$$Q(Z_{0\ldots T}, X_{0\ldots T}) = Q_*(X_{0\ldots T}) Q(Z_{0\ldots T}\vert X_{0\ldots T}).$$ 

The factor $$Q(Z_{0\ldots T}\vert X_{0\ldots T})$$ is a variational parameter in the optimization of $$H_{Q\Vert P}(Z_{0\ldots T},X_{0\ldots T}).$$ It is a discriminative model for approximately inferring $$Z$$ from $$X,$$ as opposed to the generative model $$P(Z,X)$$ for approximately sampling from $$Q_*(X).$$ Variational inference involves optimization over the space of discriminative models $$Q(Z,X)$$ and generative models $$P(Z,X).$$

By the chain rule of relative entropy,

$$H_{Q\Vert P}(Z_{0\ldots T},X_{0\ldots T}) = H_{Q\Vert P}(Z_{0\ldots T}\vert  X_{0\ldots T}) + H_{Q_*\Vert P}(X_{0\ldots T})$$

so a minimum value for $$H_{Q\Vert P}(Z_{0\ldots T},X_{0\ldots T})$$ will be a lower bound for the minimum value for $$H_{Q_*\Vert P}(X_{0\ldots T}).$$ As shown [previously](https://shaoweilin.github.io/machine-learning-with-relative-entropy/), when minimizing with $$Q$$ varying over the unconstrained space $$\Delta$$, the gap $$H_{Q\Vert P}(Z_{0\ldots T}\vert X_{0\ldots T})$$ in the bound vanishes.

## How do we perform online learning on latent processes?

In the training of hidden Markov models, the Baum-Welch algorithm or forward-backward algorithm is often used. It requires knowledge of the visible process from the start to the end of the time interval. An algorithm that only uses the visible process from the start to the present is called an _online_ learning algorithm. Otherwise, the learning algorithm is said to be _offline_.

To understand the difference between online learning and offline learning in variational inference, let us consider a joint distribution 

$$Q(Z_{0\ldots T}, X_{0\ldots T}) = Q_*(X_{0\ldots T}) Q(Z_{0\ldots T}\vert X_{0\ldots T})$$ 

in the unconstrained space $$\Delta.$$ In the discrete time case, we may decompose the discriminative model $$Q(Z_{0\ldots T}\vert X_{0\ldots T})$$ as

$$\begin{array}{rl} Q(Z_{0\ldots T}\vert X_{0\ldots T}) & =  Q(Z_0 \vert X_{0\ldots T}) \\ & \\ & \quad  Q(Z_1 \vert Z_0,X_{0\ldots T}) \cdots \\ & \\ & \quad Q(Z_T \vert Z_{0\ldots (T-1)}, X_{0\ldots T}). \end{array}$$

Each of the factors in the decomposition are unconstrained, so inference for each variable $$Z_t$$ could depend on observations from the entire visible process $$X_{0\ldots T}.$$ Therefore, optimizing over the unconstrained space $$\Delta$$ corresponds to a form of offline learning where the entropy gap $$H_{Q\Vert P}(Z_{0\ldots T}\vert X_{0\ldots T})$$ vanishes.

For computational reasons, we may require that the discriminative model performs inference on $$Z_t$$ using only observations $$X_{0\ldots (t-1)}$$ from the past. We shall assume that $$Q(Z_{0\ldots T}\vert X_{0\ldots T})$$ has a factorization

$$\begin{array}{rl} Q(Z_{0\ldots T}\vert X_{0\ldots T}) & = Q(Z_0) \\ & \\ & \quad Q(Z_1 \vert Z_0,X_0) \cdots \\ & \\ & \quad Q(Z_T \vert Z_{0\ldots (T-1)}, X_{0\ldots (T-1)}) \end{array}$$

and let $$\Delta_\mathcal{C} \subset \Delta$$ be the subspace of distributions $$Q(Z_{0\ldots T}, X_{0\ldots T})$$ with this property.  Here, the subscript $$\mathcal{C}$$ denotes that we have causal constraints on the distributions. 

We may think of these factors as functionals (because they are functions of paths) parametrizing the space $$\Delta_\mathcal{C}$$ of distributions. 

In continuous time, the above decomposition of $$Q \in \Delta_\mathcal{C}$$ can be written in terms of exponentials of integrals of transition rates, giving us some system of Kolmogorov equations.

From this factorization, we can deduce by marginalization that for all $$t$$,

$$\begin{array}{rl} Q(Z_{0\ldots t}, X_{0\ldots t}) & = Q(X_{0\ldots t}) Q(Z_0) \\ & \\ & \quad Q(Z_1 \vert Z_0,X_0) \cdots \\ & \\ & \quad Q(Z_t \vert Z_{0\ldots (t-1)}, X_{0\ldots (t-1)}) \end{array}$$

and consequently, 

$$\begin{array}{rl}  Q(Z_{t+1}, X_{t+1} \vert Z_{0\ldots t}, X_{0\ldots t}) & = \displaystyle \frac{ Q(Z_{0\ldots (t+1)}, X_{0\ldots (t+1)}) }{ Q(Z_{0\ldots t}, X_{0\ldots t}) } \\ & \\ & = Q( X_{t+1} \vert X_{0\ldots t}) Q(Z_{t+1} \vert Z_{0\ldots t}, X_{0\ldots t}).\end{array}$$

This implies that $$Z_t$$ and $$X_t$$ are conditionally independent given their past. In discrete-time, this means that 

$$Q(Z_{t+1}, X_{t+1} \vert Z_{0\ldots t}, X_{0\ldots t}) =  Q(Z_{t+1} \vert Z_{0\ldots t}, X_{0\ldots t}) Q(X_{t+1} \vert Z_{0\ldots t}, X_{0\ldots t})$$

and vice versa. In continuous time, the analogous property is

$$Q(Z_{t\ldots t+\delta}, X_{t\ldots t+\delta} \vert Z_{0\ldots t}, X_{0\ldots t}) \rightarrow Q(Z_{t\ldots t+\delta} \vert Z_{0\ldots t}, X_{0\ldots t}) Q(X_{t\ldots t+\delta} \vert Z_{0\ldots t}, X_{0\ldots t})$$

as $$\delta \rightarrow 0$$ from above. 

Another way to state this requirement is to say that $$Z_t$$ does not lie in the causal cone of $$X_t$$ and vice versa. There are interesting connections to [causal conditions](https://en.wikipedia.org/wiki/Causality_conditions) (classification of space-time manifolds), [causal structures](https://en.wikipedia.org/wiki/Causal_structure) (relationships between points on a continuous space) and [causal sets](https://en.wikipedia.org/wiki/Causal_sets) (relationships between points on a discrete space) but we will not be exploring them in this article.


If $$Z_t$$ and $$X_t$$ are conditionally independent given their past for both the discriminative model $$Q(Z,X)$$ and the generative model $$P(Z,X)$$, then in discrete time, the chain rule for relative entropy simplifies to

$$\begin{array}{rl} &H_{Q\Vert P}(Z_{n+1}, X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}) \\ & \\ &= H_{Q\Vert P}(Z_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}) + H_{Q\Vert P}(X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}). \end{array}$$

in discrete time has the continuous-time analogue

$$\begin{array}{rl} & \displaystyle \frac{d}{ds}H_{Q \Vert P} (Z_{0\ldots s}, X_{0\ldots s}) \\ & \\ &= \displaystyle\frac{d Z_{0\ldots s}}{ds} * \frac{\partial}{\partial Z_{0\ldots s}} H_{Q \Vert P} (Z_{0\ldots s}, X_{0\ldots s}) \\ & \\ &\quad +\displaystyle \frac{d X_{0\ldots s}}{ds} * \frac{\partial}{\partial X_{0\ldots s}} H_{Q \Vert P} (Z_{0\ldots s}, X_{0\ldots s})\end{array}$$

which mirrors the multivariate chain rule in calculus

$$\displaystyle \frac{d}{ds}f(z(s),x(s)) =\frac{dz}{ds} \frac{\partial}{\partial z} f(z,x) +\displaystyle \frac{dx}{ds} \frac{\partial}{\partial x} f(z,x).$$

Online learning involves minimizing over the subspace $$\Delta_\mathcal{C},$$ where the discriminative factors $$Q(Z_t \vert Z_{0\ldots (t-1)}, X_{0\ldots (t-1)})$$ depend only on information from the past. When optimizing over $$\Delta_\mathcal{C},$$ the entropy gap $$H_{Q\Vert P}(Z_{0\ldots T}\vert X_{0\ldots T})$$ might not vanish. Loosely, the gap represents _the cost of not knowing the future_. Indeed, if $$Z_t$$ and $$X_t$$ are not conditionally independent given their past, then any dependence must come from the future. Since the gap vanishes when we optimize over distributions that allow such kinds of dependence, the future must be furnishing information for closing this gap.

## How do we perform variational inference with limited memory?

Suppose now that computationally, both the discriminative model and the generative model have limited memory. More precisely, the models cannot store the full histories $$Z_{0\ldots t}, X_{0\ldots t}$$ but they can recall the most recent states $$Z_t, X_t$$ so the distributions satisfy the Markov property. Hence, $$Q(Z_{0\ldots T}\vert X_{0\ldots T})$$ has a factorization

$$\begin{array}{rl} Q(Z_{0\ldots T}\vert X_{0\ldots T}) & = Q(Z_0) \\ & \\ & \quad Q(Z_1 \vert Z_0,X_0) \cdots \\ & \\ & \quad Q(Z_T \vert Z_{T-1}, X_{T-1}). \end{array}$$

and let $$\Delta_\mathcal{M} \subset \Delta_\mathcal{C}$$ be the subspace of distributions $$Q(Z_{0\ldots T} \vert X_{0\ldots T})$$ with this property.  

Our goal is to train the model by minimizing the limit of the time-averaged relative entropy or relative entropy rate

$$\begin{array}{rl}  V(Q,\theta) &= \displaystyle \lim_{T \rightarrow \infty} \frac{1}{T} H_{Q \Vert P}(Z_{0\ldots T},X_{0\ldots T}) \\ & \\ & \displaystyle = \lim_{T\rightarrow \infty} \frac{d}{dT}H_{Q \Vert P}(Z_{0\ldots T}, X_{0\ldots T}) \end{array}$$

over $$ Q \in \Delta_\mathcal{M}$$ and $$ \theta \in \Theta.$$ Compared to minimizing this objective for $$ Q$$ over the larger space $$ \Delta_\mathcal{C},$$ we will incur an additional cost due to the Markov constraint. We can think of this cost as the cost of _limited memory_.

Of course, the recent state $$Z_t$$ can also be used to store copies of older states $$Z_{t-1}, Z_{t-2}, \ldots$$ but because the dimension of $$Z_t$$ is finite, only a limited number of those states can be stored. A smarter way is to store compressed versions of those states, or only store pertinent information about those states. 

Biological spiking networks need to work with the constraint of not knowing the future and having limited memory. For the remainder of this series, we will also work with this constraint by minimizing the time-averaged relative entropy or relative entropy rate over extensions $$Q \in \Delta_\mathcal{M}$$ and parameters $$\theta \in \Theta$$.

## References

[L92] Leroux, Brian G. "Maximum-likelihood estimation for hidden Markov models." _Stochastic processes and their applications_ 40, no. 1 (1992): 127-143.

[S01] Sato, Masa-Aki. "Online model selection based on the variational Bayes." _Neural computation_ 13, no. 7 (2001): 1649-1681.

[KMMW19] Karimi, Belhal, Blazej Miasojedow, Éric Moulines, and Hoi-To Wai. "Non-asymptotic analysis of biased stochastic approximation scheme." _arXiv preprint arXiv:1902.00629_ (2019).