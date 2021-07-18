---
layout: post
title: Relative inference with mutable processes
---

We introduce a information-theoretic objective, which is a form of relative information between a discriminative model and a generative model, for learning processes using models with [mutable](https://shaoweilin.github.io/machine-learning-with-relative-information/#why-should-we-consider-mutable-variables-rather-than-latent-variables) variables. This technique is known as [relative inference](https://shaoweilin.github.io/machine-learning-with-relative-information/#why-do-we-need-a-better-name-for-variational-inference) (also called approximate inference, variational inference or variational Bayes).

We discuss natural constraints on the discriminative and generative models, and the consequences of these constraints on:

- passive/active learning;
- online/offline learning; 
- learning with limited/unlimited memory;
- learning with limited/unlimited sensing.

This post is a continuation from our [series](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

## How do we apply relative inference to latent processes?

Suppose that we have two processes -- an observed $$X_t \in \mathcal{X},$$ and a latent or hidden process $$Z_t \in \mathcal{Z}$$. Let $$Q_*(X_{0\ldots T})$$ be the true distribution of the visible process $$X_{0\ldots T}$$. Recall that our goal is to find a generative model $$P(Z,X)$$ that minimizes the relative information

$$\displaystyle I_{Q_*\Vert P}(X_{0\ldots T})$$

(or equivalently the relative information rate if we assume strong stationarity.)

[Previously](https://shaoweilin.github.io/machine-learning-with-relative-information/), we discussed how this optimization problem over the space of distributions on $$X$$ may be lifted to an optimization problem over the space of distributions on $$(Z,X)$$ using [relative inference](https://shaoweilin.github.io/machine-learning-with-relative-information/#why-do-we-need-a-better-name-for-variational-inference). In particular, we will be considering different spaces of discriminative models $$Q(Z,X) = Q_*(X) Q(Z\vert X),$$ and attempt to minimize the joint relative information

$$I_{Q\Vert P}(Z_{0\ldots T},X_{0\ldots T})$$

as $$Q$$ varies over some space of joint distributions.

## Is relative inference necessarily passive?

The form of relative inference we are considering here is necessarily passive. This means that the true distribution $$Q_*(X_{0\ldots T})$$ is unaffected by the learning agent's estimates of the states of the latent variables $$Z_{0\ldots T}.$$ 

However, it is possible to design methods in relative inference for models where the estimates of the latent variables have an effect on the observed data, such as in active learning, reinforcement learning or partially-ordered Markov decision processes. The estimates could lead a learning agent to effect some change on the environment which then affects the observed data. Unfortunately, we will not be considering this active form of relative inference here. A complete understanding of biological intelligence must however analyze the active case.

In our passive form of relative inference, the most general case for the space of joint path distributions $$Q(Z_{0\ldots T}, X_{0\ldots T})$$ is the unconstrained space $$\Delta$$ where $$Q$$ factors as 

$$Q(Z_{0\ldots T}, X_{0\ldots T}) = Q_*(X_{0\ldots T}) Q(Z_{0\ldots T}\vert X_{0\ldots T}).$$ 

The factor $$Q(Z_{0\ldots T}\vert X_{0\ldots T})$$ is a parameter in the optimization of $$I_{Q\Vert P}(Z_{0\ldots T},X_{0\ldots T}).$$ It is a discriminative model for approximately inferring $$Z$$ from $$X,$$ as opposed to the generative model $$P(Z,X)$$ for approximately sampling from $$Q_*(X).$$ Relative inference involves optimization over the space of discriminative models $$Q(Z,X)$$ and generative models $$P(Z,X).$$

By the chain rule of relative information,

$$I_{Q\Vert P}(Z_{0\ldots T},X_{0\ldots T}) = I_{Q\Vert P}(Z_{0\ldots T}\vert  X_{0\ldots T}) + I_{Q_*\Vert P}(X_{0\ldots T})$$

so a minimum value for $$I_{Q\Vert P}(Z_{0\ldots T},X_{0\ldots T})$$ will be a lower bound for the minimum value for $$I_{Q_*\Vert P}(X_{0\ldots T}).$$ As shown [previously](https://shaoweilin.github.io/machine-learning-with-relative-information/), when minimizing with $$Q$$ varying over the unconstrained space $$\Delta$$, the gap $$I_{Q\Vert P}(Z_{0\ldots T}\vert X_{0\ldots T})$$ in the bound vanishes.

## How do we perform online learning on latent processes?

In the training of hidden Markov models, the Baum-Welch algorithm or forward-backward algorithm is often used. It requires knowledge of the visible process from the start to the end of the time interval. An algorithm that only uses the visible process from the start to the present is called an _online_ learning algorithm. Otherwise, the learning algorithm is said to be _offline_.

To understand the difference between online learning and offline learning in relative inference, let us consider a joint distribution 

$$Q(Z_{0\ldots T}, X_{0\ldots T}) = Q_*(X_{0\ldots T}) Q(Z_{0\ldots T}\vert X_{0\ldots T})$$ 

in the unconstrained space $$\Delta.$$ In the discrete time case, we may decompose the discriminative model $$Q(Z_{0\ldots T}\vert X_{0\ldots T})$$ as

$$\begin{array}{rl} Q(Z_{0\ldots T}\vert X_{0\ldots T}) & =  Q(Z_0 \vert X_{0\ldots T}) \\ & \\ & \quad  Q(Z_1 \vert Z_0,X_{0\ldots T}) \cdots \\ & \\ & \quad Q(Z_T \vert Z_{0\ldots (T-1)}, X_{0\ldots T}). \end{array}$$

Each of the factors in the decomposition are unconstrained, so inference for each variable $$Z_t$$ could depend on observations from the entire visible process $$X_{0\ldots T}.$$ Therefore, optimizing over the unconstrained space $$\Delta$$ corresponds to a form of offline learning where the information gap $$I_{Q\Vert P}(Z_{0\ldots T}\vert X_{0\ldots T})$$ vanishes.

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

If $$Z_t$$ and $$X_t$$ are conditionally independent given their past for both $$Q(Z,X)$$ and $$P(Z,X)$$, then in discrete time, the chain rule for relative information simplifies to

$$\begin{array}{rl} &I_{Q\Vert P}(Z_{n+1}, X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}) \\ & \\ &= I_{Q\Vert P}(Z_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}) + I_{Q\Vert P}(X_{n+1} \vert Z_{0\ldots n},X_{0\ldots n}). \end{array}$$

This chain rule has the continuous-time analogue

$$\begin{array}{rl} & \displaystyle \frac{d}{ds}I_{Q \Vert P} (Z_{0\ldots s}, X_{0\ldots s}) \\ & \\ &= \displaystyle\frac{d Z_{0\ldots s}}{ds} * \frac{\partial}{\partial Z_{0\ldots s}} I_{Q \Vert P} (Z_{0\ldots s}, X_{0\ldots s}) \\ & \\ &\quad +\displaystyle \frac{d X_{0\ldots s}}{ds} * \frac{\partial}{\partial X_{0\ldots s}} I_{Q \Vert P} (Z_{0\ldots s}, X_{0\ldots s})\end{array}$$

which mirrors the multivariate chain rule in calculus

$$\displaystyle \frac{d}{ds}f(z(s),x(s)) =\frac{dz}{ds} \frac{\partial}{\partial z} f(z,x) +\displaystyle \frac{dx}{ds} \frac{\partial}{\partial x} f(z,x).$$

Online learning involves minimizing over the subspace $$\Delta_\mathcal{C},$$ where the discriminative factors $$Q(Z_t \vert Z_{0\ldots (t-1)}, X_{0\ldots (t-1)})$$ depend only on information from the past. When optimizing over $$\Delta_\mathcal{C},$$ the information gap $$I_{Q\Vert P}(Z_{0\ldots T}\vert X_{0\ldots T})$$ might not vanish. Loosely, the gap represents _the cost of not knowing the future_. Indeed, if $$Z_t$$ and $$X_t$$ are not conditionally independent given their past, then any dependence must come from the future. Since the gap vanishes when we optimize over distributions that allow such kinds of dependence, the future must be furnishing information for closing this gap.

## How do we perform relative inference with limited memory?

Suppose now that computationally, both the discriminative model and the generative model have limited memory. More precisely, the models cannot store the full histories $$Z_{0\ldots t}, X_{0\ldots t}$$ but they can recall the most recent states $$Z_t, X_t$$ so conditioned on the observed data, the distributions satisfy the Markov property. Hence, $$Q(Z_{0\ldots T}\vert X_{0\ldots T})$$ has a factorization

$$\begin{array}{rl} Q(Z_{0\ldots T}\vert X_{0\ldots T}) & = Q(Z_0) \\ & \\ & \quad Q(Z_1 \vert Z_0,X_0) \cdots \\ & \\ & \quad Q(Z_T \vert Z_{T-1}, X_{T-1}). \end{array}$$

and let $$\Delta_\mathcal{M} \subset \Delta_\mathcal{C}$$ be the subspace of distributions that arise from $$Q(Z_{0\ldots T} \vert X_{0\ldots T})$$ with this property.  

Compared to minimizing the objective $$I_{Q\Vert P}(Z_{0\ldots T},X_{0\ldots T})$$ for $$Q$$ over the larger space $$\Delta_\mathcal{C},$$ we will incur an additional increase in the information gap $$I_{Q\Vert P}(Z_{0\ldots T}\vert  X_{0\ldots T})$$ due to the Markov constraint. We can think of this increase as the cost of _limited memory_.

Of course, the recent state $$Z_t$$ can also be used to store copies of older states $$Z_{t-1}, Z_{t-2}, \ldots$$ but because the dimension of $$Z_t$$ is finite, only a limited number of those states can be stored. A smarter way is to store compressed versions of those states, or only store pertinent information about those states. 


## How do we perform relative inference with limited sensing?

In the mathematical analysis, it will be semantically convenient to assume that the process $$X_{0\ldots T}$$ represents the history of _every particle in the universe_ and that it is a Markov process. 

The latent process $$Z_{0\ldots T}$$ will not represent the states of unobserved particles in the universe; rather, it will represent states of computation in the discriminative model $$Q(Z,X)$$ and generative model $$P(Z,X).$$ 

Of course, not every particle will be observed for inference by the discriminative model - we impose this constraint by writing down discriminative models $$Q(Z\vert X)$$ which do not compute with the states of those unobserved particles. Similarly, the unobserved particle will not be modeled by the generative model - we impose this constraint by fixing a trivial distribution (e.g. assigning probability one to some fixed state) to the states of the unobserved particles.

Such constraints on the discriminative model [could](https://shaoweilin.github.io/relative-inference-with-mutable-processes/#remark-cost-of-limited-sensing) cause the information gap $$I_{Q\Vert P}(Z_{0\ldots T}\vert  X_{0\ldots T})$$ to increase yet again. We can think of this increase as the cost of _limited sensing_.

The reason we say that this Markov assumption is _semantically convenient_ is as follows. We could alternatively model the universe as a joint process $$X_{0\ldots T}, U_{0\ldots T}$$ where $$X_{0\ldots T}$$ is observed and $$U_{0\ldots T}$$ is unobserved. However, the mathematical analysis becomes notationally complicated because of the need to write down both processes. It is easier to subsume the distinction between observed and unobserved within the discriminative model distribution $$Q$$ and generative model distribution $$P.$$

Under this Markov assumption and under mild regularity conditions, the time-averaged relative information will be equal to the relative information rate.

$$\begin{array}{rl}  V(Q,P) &= \displaystyle \lim_{T \rightarrow \infty} \frac{1}{T} I_{Q \Vert P}(Z_{0\ldots T},X_{0\ldots T}) \\ & \\ & \displaystyle = \lim_{T\rightarrow \infty} \frac{d}{dT}I_{Q \Vert P}(Z_{0\ldots T}, X_{0\ldots T}) \end{array}$$

The discrete time analogue of the relative information rate under this Markov setting is the asymptotic conditional relative information 

$$\lim_{n\rightarrow \infty} I_{Q \Vert P}(Z_{n+1},X_{n+1}\vert Z_n, X_n).$$

Our goal is therefore to train the model by minimizing the relative information rate or conditional relative information over Markov models $$\{Q \}$$ and $$\{ P_\theta \}$$ using methods from relative inference.

Biological spiking networks need to work with the constraints of not knowing the future and having limited memory and sensing. For the remainder of this series, we will also work with these constraints.

## Remark: Cost of limited sensing 

Let $$V_t$$ and $$U_t$$ be the observed and unobserved components of each $$X_t.$$ Suppose the generative model $$P(Z_{0\ldots T},X_{0\ldots T})$$ assigns probability one to some fixed path of $$\{U_t\}.$$ It follows that the variables $$U_t$$ cannot provide us with any information about the states $$Z_t$$ under $$P,$$ so 

$$P(Z_{0\ldots T} \vert V_{0\ldots T}, U_{0\ldots T}) = P(Z_{0\ldots T} \vert V_{0\ldots T}).$$

When $$P$$ is fixed, the information gap $$I_{Q\Vert P}(Z_{0\ldots T}\vert  X_{0\ldots T})$$ is minimized when $$Q(Z_{0\ldots T}\vert  X_{0\ldots T})$$ is as close to $$P(Z_{0\ldots T} \vert X_{0\ldots T}) = P(Z_{0\ldots T} \vert V_{0\ldots T})$$ as possible. Therefore, restricting $$Z_{0\ldots T}$$ to depend only on the observables $$V_{0\ldots T}$$ under $$Q$$ as opposed to all of $$X_{0\ldots T}$$ will not cause the information gap to increase. The gap will increase however if $$Q$$ is forced to infer $$Z_{0\ldots T}$$ from a strict subset of $$V_{0\ldots T}.$$