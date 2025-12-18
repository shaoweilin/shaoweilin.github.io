---
date: 2020-09-08
excerpts: 3
---

# Building foundations of information theory on relative information

The relative information {cite}`baez2014bayesian` (also known as relative entropy or Kullback-Leibler divergence) is an important object in information theory for measuring how far a probability measure $Q$ is from another probability measure $P.$ Here, $Q$ is usually the true distribution of some real phenomenon, and $P$ is some model distribution.

In this post, we emphasize that the relative information is fundamental in the sense that all other interesting information-theoretic objects may be derived from it. We also outline how relative information can be defined without probability mass functions or probability density functions, or even in the absence of absolute continuity.

This is the first post in our [series](2020-08-28-motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

## Why build foundations of information theory on relative information?

Firstly, we want to show that relative information is the "right way" to think about machine learning problems. Many methods like max entropy or max likelihood can be framed in terms of min relative information. We can then use this reformulation to derive more robust learning algorithms (e.g. stochastic gradients) and we can prove asymptotic properties of these algorithms more easily. We will also be using (conditional) relative information to derive learning algorithms for statistical models with hidden variables. All of these will be explained in a later post.

The second reason is because we will be extending relative information to take values in a motivic ring so as to get motivic information theory. This then allows us to write down path integrals that don't run into convergence issues.

## What is relative information?

Given probability measures $P, Q$ on a finite state space $\{1, \ldots, n\},$ the relative information to $Q$ from $P$ is the sum

$$I_{Q\Vert P} = \displaystyle \sum_{i=1}^{n} Q(i) \log \frac{Q(i)}{P(i)}.$$

In the continuous case, given probability measures $P, Q$ on $\mathbb{R}^n$ with density functions $p(x), q(x),$ the relative information to $Q$ from $P$ is the integral

$$\displaystyle I_{Q\Vert P} = \int q(x) \log \frac{q(x)}{p(x)}\, dx.$$

It is not difficult to show that $I_{Q\Vert P} \geq 0$ for all probability measures $P, Q$ with equality if and only if $P = Q$ almost everywhere.

## How do we derive entropy from relative information?

Classical textbooks for information theory define the relative information in terms of entropy (and cross entropy). We could choose instead to think of the relative information as the fundamental object, and entropy as a special case {cite}`gray2011entropy`, {cite}`chodrow2017divergence`.

Let $X$ be a random variable with values in the measurable space $(\Omega, \mathcal{B})$, and let $P_X$ be its distribution. For discrete state spaces $\Omega$, the entropy of $X$ is defined as

$$H_{P_X}(X) = - \displaystyle \sum_{i=1}^{n} P_X(i) \log P_X(i).$$

In general, for any measurable space $(\Omega, \mathcal{B})$, we now construct two extreme measures on the product space $(\Omega \times \Omega, \mathcal{B} \otimes \mathcal{B})$.

The first is the product measure $P_X \times P_X$ where

$$(P_X \times P_X) (F \times G) = P_X(F) P_X(G)$$

for all $F,G \in \mathcal{B}$. We may think of it as the joint distribution on two independent random variables $X_1, X_2$ whose marginals are both equal to $P_X$.

The second is the _diagonal_ measure

$$P_{XX}(F\times G) = P_X(F \cap G)$$

for all $F,G \in \mathcal{B}$. When $\Omega$ is finite, $P_{XX}(x_1,x_2)$ equals $P(x_1)$ if $x_1 = x_2$, and zero otherwise. We may think of it as the joint distribution on two dependent random variables $X_1 = X_2$ whose marginals are also both equal to $P_X$.

The entropy of $X$ may be then defined as

$$\begin{array}{rl} H_{P_X}(X) &= I_{P_{XX} \Vert P_X\times P_X} \\ & \\ &= \displaystyle \sum_{i,j} P_{XX}(i,j) \log \frac{P_{XX}(i,j)}{P_X(i)P_X(j)} \\ & \\ &= \displaystyle \sum_{i} P_{X}(i) \log \frac{P_{X}(i)}{P_X(i)P_X(i)} \\ & \\ &= -\displaystyle \sum_{i} P_X(i) \log P_X(i), \end{array}$$

the relative information to $P_{XX}$ from $P_X\times P_X.$ Therefore, entropy measures the amount of information gained when we learn that two random variables $X_1, X_2$ previously believed to be completely independent are actually completely dependent.

A different view of the relationship between entropy and relative information starts with the observation that Shannon's formula for discrete entropy does not behave well when we take its limit to get a formula for continuous entropy. Jaynes proposed a correction called the limiting density of discrete points (LDDP) that is defined as the negative relative information to $P_X$ from the uniform distribution {cite}`jaynes1957information`. In the discrete case, the LDDP works out to be

$$\displaystyle \sum_{i=1}^{n} P_X(i) \log \frac{P_X(i)}{1/n} = \log n - H_{P_X}(X).$$

Using the relative information of the uniform distribution to $P_X$ as the "right way" to think about the entropy of $X$, we find that this relative information is always positive and it decreases to zero as a system loses its structure and becomes more uniform.

In many interesting problems, the uniform distribution is not well-defined, so this second definition of entropy in terms of relative information will not make sense. We will then revert to the first definition as the relative information to the dependent distribution from the independent distribution.

## How do we define relative information without densities?

In our definition of relative information, the densities $p(x) = dP/dx$ and $q(x) = dQ/dx$ are Radon-Nikodym derivatives which exist if and only if $P$ and $Q$ are absolutely continuous with respect to the Lebesgue measure on $\mathbb{R}^n.$ However, there are many important applications where this assumption is not true, such as in the case of path measures for stochastic processes.

To avoid such technical difficulties, we may define the relative information in terms of the Radon-Nikodym derivative $dQ/dP,$ which exists when $Q$ is absolutely continuous with respect to $P$ (i.e. $Q \ll P$).

$$\displaystyle I_{Q\Vert P} = \int \log \frac{dQ}{dP} \,dQ$$

Substituting the equations

$$dQ = \displaystyle \frac{dQ}{dx} dx$$

$$\displaystyle \frac{dQ}{dP} = \displaystyle \frac{dQ}{dx}/\frac{dP}{dx}$$

arising from the chain rule for Radon-Nikodym derivatives, we get the earlier definition of the relative information in terms of densities.

## How do we define relative information without absolute continuity?

When $Q$ is not absolutely continuous with respect to $P,$ the Radon-Nikodym derivative $dQ/dP$ does not exist and the above definition of $I_{Q\Vert P}$ does not make sense. To extend the definition, we consider the relative information of a partition of $\Omega$ {cite}`gray2011entropy`.

Let $(\Omega, \mathcal{B})$ be a measurable space and let $P, Q$ be two probability measures on this space. For finite partitions $\mathcal{W}=\left\{ \mathcal{W}_1, \ldots, \mathcal{W}_k \right\}$ of $\Omega$, we define

$$I_{Q\Vert P}(\mathcal{W}) = \displaystyle\sum_{\mathcal{W}_i \in \mathcal{W}} Q(\mathcal{W}_i) \log \frac{Q(\mathcal{W}_i)}{P(\mathcal{W}_i)},$$

and for infinite partitions $\mathcal{U}$, we define

$$I_{Q\Vert P}(\mathcal{U}) = \displaystyle \sup_{\mathcal{U}\leq \mathcal{W}} I_{Q\Vert P}(\mathcal{W})$$

where the supremum is taken over all finite partitions $\mathcal{W}$ such that $\mathcal{U}$ is a refinement of $\mathcal{W}$. Finally, we define $I_{Q\Vert P} = I_{Q\Vert P}(\Omega)$.

Given any random variable $X$ on $\Omega$, we also define $I_{Q\Vert P}(X)$ to be $I_{Q\Vert P}(\mathcal{U})$ where $\mathcal{U}$ is the partition of $\Omega$ induced by $X$. Because the relative information only depends on the induced distributions $P_X, Q_X$ for $X$, we have

$$I_{Q\Vert P}(X) = I_{Q_X \Vert P_X}.$$

When $Q$ is not absolutely continuous with respect to $P,$ we can show that this definition tells us that the relative information is infinite.

## How do we define relative information when the total measure is not one?

Let us assume that both $P, Q$ have the same total measure $T$ which is not necessarily equal to one. Let $\bar{P} = P/T$ and $\bar{Q}= Q/T$ be the respective probability measures. We define

$$I_{Q\Vert P} =\displaystyle \sum_{i=1}^{n} Q(i) \log \frac{Q(i)}{P(i)}.$$

Then, a simple calculation shows that

$$I_{Q\Vert P} = T I_{\bar{Q}\Vert \bar{P}}.$$

## How do we define relative information when the total measures of $P, Q$ are not the same?

The short answer is Don't. It does not make sense to compare two measures with different total measures.

The long answer is as follows. Suppose that we do want to go ahead and extend the definition of relative information to this situation. What property of relative information would we want to preserve?

A reasonable property would be the nonnegativity of relative information. If we study the proof of nonnegativity in the classical situation, it hinges on the fact that

$$x-1-\log x \geq 0$$

for all $x \geq 0$. Continuing the proof, we replace $x$ by $P(i)/Q(i)$, multiply the inequality by $Q(i)$ and sum up the inequalities over $i$. This gives us

$$\displaystyle \sum_{i=1}^{n} P(i)-\displaystyle\sum_{i=1}^{n} Q(i)+ \displaystyle\sum_{i=1}^{n} Q(i) \log \frac{Q(i)}{P(i)} \geq 0.$$

Let us define naively

$$I_{Q\Vert P} = \displaystyle\sum_{i=1}^{n} P(i)-\displaystyle\sum_{i=1}^{n} Q(i)+\displaystyle\sum_{i=1}^{n} Q(i) \log \frac{Q(i)}{P(i)}.$$

Let $T_P, T_Q$ be the total measures of $P, Q$ assuming that they are finite, and let $\bar{P} = P/T_P$ and $\bar{Q}= Q/T_Q$ be the corresponding probability measures. Then,

$$\begin{array}{rl} I_{Q\Vert P} &= T_P-T_Q+\displaystyle\sum_{i=1}^{n} T_Q \bar{Q}(i) \log \frac{T_Q \bar{Q}(i)}{T_P \bar{P}(i)} \\ & \\ &= \displaystyle T_P-T_Q+ T_Q \log \frac{T_Q}{T_P}+ T_Q I_{\bar{Q}\Vert \bar{P}}.\end{array}$$

We see that the value of $I_{Q\Vert P}$ depends on the measures $P, Q$ only through the classical relative information $I_{\bar{Q}\Vert \bar{P}}$, so we do not gain anything new. Relative information is only meaningful for understanding the distance between two measures that distribute the same total measure differently.

## References

```{bibliography}
:filter: docname in docnames
```