---
date: 2020-10-14
excerpts: 2
---

# Path integrals and continuous-time Markov chains

We give an introduction to continuous-time Markov chains, and define path measures for these objects.

This post is a continuation from our [series](2020-08-28-motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

## What is a stochastic process?

A stochastic process {cite}`albeverio2017probabilistic` on a measure space $(\Omega, \mathcal{A},P)$ indexed by a set $I$ is a family $\{X_t \}_{t \in I}$ of random variables $X_t : \Omega \rightarrow R$ on $\Omega,$ where $(R$ is some measure space. This allows us to compute _finite joint distributions_ of the form

$$P(\{\omega \in \Omega \vert X_{t_1}(\omega) \in \mathcal{A}_1, \ldots, X_{t_n}(\omega) \in \mathcal{A}_n\})$$

for some finite set $\{t_1, \ldots, t_n\}$ of indices and some measurable subsets $\mathcal{A}_1, \ldots, \mathcal{A}_n$ of $R.$

## What is a path measure and a path integral?

A priori, a stochastic process is just a collection of random variables for which we write down finite joint distributions. How do we define a path through these random variables? Is there a space of paths imbued with a measure $\mu$ that is consistent with the finite joint distributions?

A path in $R$ is a function $\gamma: I \rightarrow R$ over the index set $I$, and the set of all paths is denoted by $R^I.$ Note that paths need not be continuous, though they will later need to satisfy some kind of Holder continuity so that line integrals along a fixed path are well-defined.

We want to construct a $\sigma$-algebra of measurable subsets of $R^I$ and a measure $\mu$ over these measurable subsets such that _cylinders_ of the form

$$\{\gamma \in R^I \vert \gamma(t_1) \in \mathcal{A}_1, \ldots, \gamma(t_n) \in \mathcal{A}_n\}$$

are measurable and have measure equal to the finite joint distribution defined previously.

The _Kolmogorov extension theorem_ tells us that under mild topological assumptions about $R$, the answer is Yes. In fact, the desired $\sigma$-algebra will be generated by the cylinders defined above, and the measure of each set in the $\sigma$-algebra will be given by the Caratheodory extension theorem.

We now define _path integrals_ - integrals of functions of paths over this path measure $\mu$. The definition will not involve writing down a density at each path in the path space. The existence of such a path density requires the path measure that is absolutely continuous with respect to the Lebesgue measure on the index set $I$, which often does not hold. Instead, as commonly done for classical Lebesgue integration, we will define the path integral of a measurable function to be the limit of path integrals of simpler _cylinder_ functions.

Let $\mu$ be a path measure on some path space $R^I$. Given a finite subset $J = \{t_1, \ldots, t_n\} \subset I$, let $\pi_J : R^I \rightarrow R^J$ be the projection

$$\pi_J (\gamma) = (\gamma(t_1), \ldots, \gamma(t_1)).$$

Let $\mu_J = \mu \circ \pi_J^{-1}$ be the pushforward measure on the space $R^J.$

A _cylinder function_ $f: R^I \rightarrow \mathbb{C}$ is a function of the form

$$f(\gamma) = F(\gamma(t_1), \ldots, \gamma(t_n))$$

for some $t_1, \ldots, t_n \in I$ and some measurable function $F: R^n \rightarrow \mathbb{C}$. The path integral of the cylinder function is defined to be

$$\int_{R^I} f(\gamma) d\mu(\gamma) = \int_{R^n} F(x_1, \ldots, x_n) d\mu_J(x_1, \ldots, x_n).$$

The path integral of any measurable function $g : R^I \rightarrow \mathbb{C}$ is then defined as the limit of path integrals of cylinder functions that converge to $g$.

## What is a line integral along a path?

Given a path measure $\mu$, the path integral

$$\displaystyle \int_{R^I} f(\gamma) d\mu(\gamma)$$

may often be written (informally) in the form

$$\displaystyle \int_{R^I} e^{-S(\gamma)} D\gamma$$

or (in quantum mechanics)

$$\displaystyle \int_{R^I} e^{-i S(\gamma)} D\gamma$$

for some informal "Lebesgue path measure" $D\gamma$ and some path function $S(\gamma)$ called the _action_. The action is often an integral

$$\displaystyle S(\gamma) = \int_I L(\gamma(t),\dot{\gamma}(t)) dt$$

of some _Lagrangian_ $L(\gamma(t),\dot{\gamma}(t))$ over the path. We call $S(\gamma)$ a _line integral_ along $\gamma$ because it is an integral driven by a given path $\gamma$.

As an example, consider the Wiener process or Brownian motion, with its associated Wiener measure on the space of paths. The measure

$$\{\gamma \in R^I \vert \gamma(t_1) \in \mathcal{A}_1, \ldots, \gamma(t_n) \in \mathcal{A}_n\}$$

of a cylinder is given by the path integral

$$\displaystyle Z^{-1} \int_{\mathcal{A}_n} \cdots \int_{\mathcal{A}_1} e^{-S_n(\gamma)} dx_1 \cdots dx_n$$

where $t_0=0,$

$$\displaystyle \begin{array}{rl} S_n(\gamma) &=\displaystyle \frac{1}{2} \sum_{j=0}^{n-1} \frac{\vert x_{i+1}-x_i\vert ^2}{t_{i+1}-t_i} \\ & \\ &=\displaystyle \frac{1}{2} \sum_{j=0}^{n-1} \left\vert  \frac{x_{i+1}-x_i}{t_{i+1}-t_i}\right\vert ^2 (t_{i+1}-t_i)\end{array} $$

is the action, and

$$\displaystyle \begin{array}{rl}Z &=\displaystyle\left[(2\pi)^{n} (t_n-t_{n-1})\cdots(t_1-t_0) \right]^{d/2} \\ & \\ &=\displaystyle\int_{\mathcal{A}_n} \cdots \int_{\mathcal{A}_1} e^{-S_n(\gamma)} dx_1 \cdots dx_n.\end{array}$$

is the normalizing constant.

Informally, if we were allowed to take limits as $n \rightarrow \infty,$ we would have the path measure

$$\displaystyle \mu(\mathcal{A}) = Z^{-1}\int_\mathcal{A} e^{-S(\gamma)} D\gamma$$

defined in terms of the action

$$\displaystyle S(\gamma) = \frac{1}{2}\int_I \vert \dot{\gamma}(t)\vert ^2 dt$$

and the normalization constant

$$\displaystyle Z =\int_{R^I}e^{-S(\gamma)} D\gamma .$$

Path integrals over the path measure of a measurable function $f:R^I \rightarrow \mathbb{C}$ can then be written as

$$\displaystyle Z^{-1} \int_{R^I} f(\gamma) e^{-S(\gamma)} D\gamma.$$

Of course, these path integrals are not well-defined in the Lebesgue sense and should be interpreted as alternative notations for usual path integrals over the Wiener measure.

There has been a lot of work {cite}`albeverio2011path` {cite}`albeverio2017probabilistic` in the last few decades on formalizing these path measures and path integrals.

Most notable was the breakthrough on understanding line integrals such as $S(\gamma)$ driven by a fixed path $\gamma$, using Martin Hairer's theory of regularity structures which was inspired by rough path theory {cite}`hairer2014theory`. Path integrals of these line integrals over some path measure can then be defined formally using the Lyon-Ito map from rough path theory without a need to define any densities with respect to a "Lebesgue path measure" {cite}`inahama2019rough`. Thus, line integrals and path integrals are decoupled.

Absorbing the normalization constant $Z^{-1}$ into the action $S(\gamma)$ for convenience, the path integral

$$\displaystyle F(s) = \int_{R^I} e^{-s S(\gamma)} f(\gamma) D\gamma$$

can informally be thought of as the Laplace transform (or Fourier transform for the quantum case)

$$F(s) = \displaystyle \int_0^\infty e^{-st} J(t) dt $$

of the Gelfand-Leray function

$$J(t) = \displaystyle \frac{d}{dt} \int_{0 \leq S(\gamma) \leq t} f(\gamma) D\gamma.$$

This Gelfand-Leray function is similar to the [density of states](https://en.wikipedia.org/wiki/Density_of_states) studied in solid state physics and condensed matter physics. It integrates the path function $f(\gamma)$ over all paths having a fixed energy $S(\gamma)=t$.

## What is a continuous-time Markov chain?

A _continuous-time Markov chain_ (CTMC) is a stationary stochastic process $X(t)$ with finite or countable state space $\mathcal{X}$ in continuous time $t \in [0, \infty)$ that satisfies the Markov property, i.e.

$$\displaystyle\begin{array}{rl} & \mathbb{P}(X(t_{n+1}) = x_{n+1} \vert X(t_{n}) = x_{n}, \ldots, X(t_{0}) = x_{0} ) \\ & \\ & \quad = \mathbb{P}(X(t_{n+1}) = x_{n+1} \vert X(t_{n}) = x_{n} ) \end{array}$$

for all times $t_0 \leq \ldots \leq t_{n+1}$ and states $x_0, \ldots, x_{n+1}.$ Here, we say that a stochastic process is _stationary_ if the joint distribution of any collection $X_{t_1}, \ldots, X_{t_n}$ of random variables does not change when the indices $t_1, \ldots, t_n$ are shifted to any $t_1+\delta, \ldots, t_n+\delta$.

The finite joint distribution of a CTMC can be written as a product of conditional probabilities and an initial distribution:

$$\displaystyle \begin{array}{rl} & \mathbb{P}( X(t_{n}) = x_{n}, \ldots, X(t_{0}) = x_{0} ) = \\ & \\ & \quad \mathbb{P}(X(t_{n}) = x_{n} \vert X(t_{n-1}) = x_{n-1} ) \cdots \\ & \\ & \quad \mathbb{P}(X(t_{1}) = x_{1} \vert X(t_{0}) = x_{0} ) \mathbb{P}( X(t_{0}) = x_{0} ). \end{array}$$

The limit

$$\Gamma_{xy} = \displaystyle \lim_{\delta \rightarrow 0} \frac{\mathbb{P}(X(t+\delta) = y \vert X(t) = x ) +\mathbb{I}(x=y)}{\delta}$$

is the _transition rate_ from state $x$ to state $y$, where $\mathbb{I}$ is the indicator function. This limit does not depend on the time $t$ because the stochastic process is stationary. The (possibly infinite-dimensional) matrix $\Gamma = (\Gamma_{xy})_{x,y \in \mathcal{X}}$ is called the _transition rate matrix_. Note that $\Gamma_{xx} = -\sum_{y\neq x} \Gamma_{xy}$ because the probabilities $\mathbb{P}(X(t+\delta) = y \vert X(t) = x )$ sum to one as $y$ varies over all the states.

The dynamics of the CTMC is completely determined by the initial distribution and the transition rate matrix. Indeed, let $p(t) = (p(t)_x)_{x \in \mathcal{X}}$ with

$$p(t)_x = P(X(t) = x)$$

be the (possibly infinite-dimensional) vector of state probabilities. Then, $p(t)$ satisfies the _forward equation_

$$\displaystyle \frac{d}{dt} p(t) = \Gamma p(t)$$

whose solution is given in terms of the matrix exponential

$$\displaystyle p(t) = e^{t\Gamma} p(0)$$

and the transition probabilities for $t \geq s$ are given by

$$\mathbb{P}(X(t) = y \vert X(s) = x ) = (e^{(t-s)\Gamma})_{xy}.$$

The Kolmogorov extension theorem then allows us to define a path measure on the path space $\mathcal{X}^{[0,\infty)}$ as well as path integrals with respect to this path measure.

## How can we approximate a continuous-time Markov chain with a discrete-time Markov chain?

There are roughly two ways of approximating a continuous-time Markov chain with one that is discrete-time. The first is to let the discrete timings represent the timings of observations. The second is to let the discrete timings represent the timings of transitions.

For the first way involving observation timings, one naïve approach is to make observations at regular intervals, such as what we did for path integrals in the previous section. The problem with this approach is that limits of integrals often do not behave well when we let the time interval tend to zero.

Alternatively, within a time interval $[0,T],$ we may assume that the observations occur with Poisson rate $\lambda,$ i.e. we have an average of $\lambda$ observations in a unit time interval. Consequently, the probability that there are $N$ observations is

$$
\mathbb{P}[N=n] = \displaystyle e^{-\lambda T} \frac{(\lambda T)^n}{n!}.
$$

Given $N,$ we then choose $N$ observations uniformly within the interval $[0,T].$ The probabilities of particular observations at these $N$ timings are then given by the cylinder probabilities described earlier in this post.

As we let the observation rate $\lambda$ tend to infinity, we get better approximations of the continuous-time Markov chain and the limits of integrals should also behave well. This strategy also works well for path integrals in relativistic quantum theory, e.g. see §5 of {cite}`gill2002foundations`.

The second way involving transition timings works only when each state $X(t)$ of the stochastic process is unchanging for some nonzero time interval. This is true for continuous-time Markov chains with finite or countable state space $\mathcal{X}$. In our discussion, we will trivially allow transitions to be from a state $x$ back to itself for the sake of generality. 

Suppose that the Markov chain with state $X(t) = x \in \mathcal{X}$ at time $t$ has a Poisson rate $F_{xy}$ of transitioning to some state $y \in \mathcal{X},$ where $y$ is allowed to be equal to $x.$ Then, the rate transition matrix $(\Gamma_{xy})$ has entries $\Gamma_{xy} = F_{xy}$ for $y \neq x$, and $\Gamma_{xx} = -\sum_{y\neq x} F_{xy}$ otherwise. At state $x$, the Poisson rate to any transition event is

$$ \displaystyle \lambda_x = \sum_{y} F_{xy} $$

and the transition probabilities for the next state $y$ are

$$\displaystyle P_{xy} = \frac{F_{xy}}{\sum_y F{xy}} = \frac{F_{xy}}{\lambda_x}.$$

This second approach involving transition timings is suitable when we are simulating a continuous-time learning algorithm on machines with discrete processor clocks. The machine will keep track of transitions in the learning system, and approximate the continuous-time learning updates between the transitions with discrete-time updates.


## References

```{bibliography}
:filter: docname in docnames
```