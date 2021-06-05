---
layout: post
title: Building foundations of information theory on relative entropy
---

The relative entropy (or Kullback-Leibler divergence) is an important object in information theory for measuring how far a probability measure $$Q$$ is from another probability measure $$P.$$ Here, $$Q$$ is usually the true distribution of some real phenomenon, and $$P$$ is some model distribution.

In this article, we emphasize that the relative entropy is fundamental in the sense that all other interesting information-theoretic objects may be derived from it. We also outline how relative entropy can be defined without probability mass functions or probability density functions, or even in the absence of absolute continuity.

This is the first post in our [series](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

## Why build foundations of information theory on relative entropy?

Firstly, we want to show that relative entropy is the "right way" to think about machine learning problems. Many methods like max entropy or max likelihood can be framed in terms of min relative entropy. We can then use this reformulation to derive more robust learning algorithms (e.g. stochastic gradients) and we can prove asymptotic properties of these algorithms more easily. We will also be using (conditional) relative entropy to derive variational learning algorithms for statistical models with hidden variables. All of these will be explained in a later post.

The second reason is because we will be extending relative entropy to take values in a motivic ring so as to get motivic information theory. This then allows us to write down path integrals that don't run into convergence issues.

## What is relative entropy?

Given probability measures $$P, Q$$ on a finite state space $$\{1, \ldots, n\},$$ the relative entropy to $$Q$$ from $$P$$ is the sum

$$H_{Q\Vert P} = \displaystyle \sum_{i=1}^{n} Q(i) \log \frac{Q(i)}{P(i)}.$$

In the continuous case, given probability measures $$P, Q$$ on $$\mathbb{R}^n$$ with density functions $$p(x), q(x),$$ the relative entropy to $$Q$$ from $$P$$ is the integral

$$\displaystyle H_{Q\Vert P} = \int q(x) \log \frac{q(x)}{p(x)}\, dx.$$

It is not difficult to show that $$H_{Q\Vert P} \geq 0$$ for all probability measures $$P, Q$$ with equality if and only if $$P = Q$$ almost everywhere.

## How do we derive entropy from relative entropy?

Classical textbooks for information theory define the relative entropy in terms of entropy (and cross entropy). We could choose instead to think of the relative entropy as the fundamental object, and entropy as a special case [[G11]](#ref-G11), [[C17]](#ref-C17).

Let $$X$$ be a random variable with values in the measurable space $$(\Omega, \mathcal{B})$$, and let $$P_X$$ be its distribution. For discrete state spaces $$\Omega$$, the entropy of $$X$$ is defined as

$$H_{P_X}(X) = - \displaystyle \sum_{i=1}^{n} P_X(i) \log P_X(i).$$

In general, for any measurable space $$(\Omega, \mathcal{B})$$, we now construct two extreme measures on the product space $$(\Omega \times \Omega, \mathcal{B} \otimes \mathcal{B})$$.

The first is the product measure $$P_X \times P_X$$ where

$$(P_X \times P_X) (F \times G) = P_X(F) P_X(G)$$

for all $$F,G \in \mathcal{B}$$. We may think of it as the joint distribution on two independent random variables $$X_1, X_2$$ whose marginals are both equal to $$P_X$$.

The second is the _diagonal_ measure

$$P_{XX}(F\times G) = P_X(F \cap G)$$

for all $$F,G \in \mathcal{B}$$. When $$\Omega$$ is finite, $$P_{XX}(x_1,x_2)$$ equals $$P(x_1)$$ if $$x_1 = x_2$$, and zero otherwise. We may think of it as the joint distribution on two dependent random variables $$X_1 = X_2$$ whose marginals are also both equal to $$P_X$$.

The entropy of $$X$$ may be then defined as

$$\begin{array}{rl} H_{P_X}(X) &= H_{P_{XX} \Vert P_X\times P_X} \\ & \\ &= \displaystyle \sum_{i,j} P_{XX}(i,j) \log \frac{P_{XX}(i,j)}{P_X(i)P_X(j)} \\ & \\ &= \displaystyle \sum_{i} P_{X}(i) \log \frac{P_{X}(i)}{P_X(i)P_X(i)} \\ & \\ &= -\displaystyle \sum_{i} P_X(i) \log P_X(i), \end{array}$$

the relative entropy to $$P_{XX}$$ from $$P_X\times P_X.$$ Therefore, entropy measures the amount of information gained when we learn that two random variables $$X_1, X_2$$ previously believed to be completely independent are actually completely dependent.

A different view of the relationship between entropy and relative entropy starts with the observation that Shannon's formula for discrete entropy does not behave well when we take its limit to get a formula for continuous entropy. Jaynes proposed a correction called the limiting density of discrete points (LDDP) that is defined as the negative relative entropy to $$P_X$$ from the uniform distribution [[J57]](#ref-J57). In the discrete case, the LDDP works out to be

$$\displaystyle \sum_{i=1}^{n} P_X(i) \log \frac{P_X(i)}{1/n} = \log n - H(X).$$

Using the relative entropy of the uniform distribution to $$P_X$$ as the "right way" to think about the entropy of $$X$$, we find that this relative entropy is always positive and it decreases to zero as a system loses its structure and becomes more uniform.

In many interesting problems, the uniform distribution is not well-defined, so this second definition of entropy in terms of relative entropy will not make sense. We will then revert to the first definition as the relative entropy to the dependent distribution from the independent distribution.

## How do we define relative entropy without densities?

In our definition of relative entropy, the densities $$p(x) = dP/dx$$ and $$q(x) = dQ/dx$$ are Radon-Nikodym derivatives which exist if and only if $$P$$ and $$Q$$ are absolutely continuous with respect to the Lebesgue measure on $$\mathbb{R}^n.$$ However, there are many important applications where this assumption is not true, such as in the case of path measures for stochastic processes.

To avoid such technical difficulties, we may define the relative entropy in terms of the Radon-Nikodym derivative $$dQ/dP,$$ which exists when $$Q$$ is absolutely continuous with respect to $$P$$ (i.e. $$Q \ll P$$).

$$\displaystyle H_{Q\Vert P} = \int \log \frac{dQ}{dP} \,dQ$$

Substituting the equations

$$dQ = \displaystyle \frac{dQ}{dx} dx$$

$$\displaystyle \frac{dQ}{dP} = \displaystyle \frac{dQ}{dx}/\frac{dP}{dx}$$

arising from the chain rule for Radon-Nikodym derivatives, we get the earlier definition of the relative entropy in terms of densities.

## How do we define relative entropy without absolute continuity?

When $$Q$$ is not absolutely continuous with respect to $$P,$$ the Radon-Nikodym derivative $$dQ/dP$$ does not exist and the above definition of $$H_{Q\Vert P}$$ does not make sense. To extend the definition, we consider the relative entropy of a partition of $$\Omega$$ [[G11]](#ref-G11).

Let $$(\Omega, \mathcal{B})$$ be a measurable space and let $$P, Q$$ be two probability measures on this space. For finite partitions $$\mathcal{W}=\left\{ \mathcal{W}_1, \ldots, \mathcal{W}_k \right\}$$ of $$\Omega$$, we define

$$H_{Q\Vert P}(\mathcal{W}) = \displaystyle\sum_{\mathcal{W}_i \in \mathcal{W}} Q(\mathcal{W}_i) \log \frac{Q(\mathcal{W}_i)}{P(\mathcal{W}_i)},$$

and for infinite partitions $$\mathcal{U}$$, we define

$$H_{Q\Vert P}(\mathcal{U}) = \displaystyle \sup_{\mathcal{U}\leq \mathcal{W}} H_{Q\Vert P}(\mathcal{W})$$

where the supremum is taken over all finite partitions $$\mathcal{W}$$ such that $$\mathcal{U}$$ is a refinement of $$\mathcal{W}$$. Finally, we define $$H_{Q\Vert P} = H_{Q\Vert P}(\Omega)$$.

Given any random variable $$X$$ on $$\Omega$$, we also define $$H_{Q\Vert P}(X)$$ to be $$H_{Q\Vert P}(\mathcal{U})$$ where $$\mathcal{U}$$ is the partition of $$\Omega$$ induced by $$X$$. Because the relative entropy only depends on the induced distributions $$P_X, Q_X$$ for $$X$$, we have

$$H_{Q\Vert P}(X) = H_{Q_X \Vert P_X}.$$

When $$Q$$ is not absolutely continuous with respect to $$P,$$ we can show that this definition tells us that the relative entropy is infinite.

## How do we define relative entropy when the total measure is not one?

Let us assume that both $$P, Q$$ have the same total measure $$T$$ which is not necessarily equal to one. Let $$\bar{P} = P/T$$ and $$\bar{Q}= Q/T$$ be the respective probability measures. We define

$$H_{Q\Vert P} =\displaystyle \sum_{i=1}^{n} Q(i) \log \frac{Q(i)}{P(i)}.$$

Then, a simple calculation shows that

$$H_{Q\Vert P} = T H_{\bar{Q}\Vert \bar{P}}.$$

## How do we define relative entropy when the total measures of $$P, Q$$ are not the same?

The short answer is Don't. It does not make sense to compare two measures with different total measures.

The long answer is as follows. Suppose that we do want to go ahead and extend the definition of relative entropy to this situation. What property of relative entropy would we want to preserve?

A reasonable property would be the nonnegativity of relative entropy. If we study the proof of nonnegativity in the classical situation, it hinges on the fact that

$$x-1-\log x \geq 0$$

for all $$x \geq 0$$. Continuing the proof, we replace $$x$$ by $$P(i)/Q(i)$$, multiply the inequality by $$Q(i)$$ and sum up the inequalities over $$i$$. This gives us

$$\displaystyle \sum_{i=1}^{n} P(i)-\displaystyle\sum_{i=1}^{n} Q(i)+ \displaystyle\sum_{i=1}^{n} Q(i) \log \frac{Q(i)}{P(i)} \geq 0.$$

Let us define naively

$$H_{Q\Vert P} = \displaystyle\sum_{i=1}^{n} P(i)-\displaystyle\sum_{i=1}^{n} Q(i)+\displaystyle\sum_{i=1}^{n} Q(i) \log \frac{Q(i)}{P(i)}.$$

Let $$T_P, T_Q$$ be the total measures of $$P, Q$$ assuming that they are finite, and let $$\bar{P} = P/T_P$$ and $$\bar{Q}= Q/T_Q$$ be the corresponding probability measures. Then,

$$\begin{array}{rl} H_{Q\Vert P} &= T_P-T_Q+\displaystyle\sum_{i=1}^{n} T_Q \bar{Q}(i) \log \frac{T_Q \bar{Q}(i)}{T_P \bar{P}(i)} \\ & \\ &= \displaystyle T_P-T_Q+ T_Q \log \frac{T_Q}{T_P}+ T_Q H_{\bar{Q}\Vert \bar{P}}.\end{array}$$

We see that the value of $$H_{Q\Vert P}$$ depends on the measures $$P, Q$$ only through the classical relative entropy $$H_{\bar{Q}\Vert \bar{P}}$$, so we do not gain anything new. Relative entropy is only meaningful for understanding the distance between two measures that distribute the same total measure differently.

## References

<a id="ref-G11"></a>[[G11]](#ref-G11) Gray, Robert M. _Entropy and information theory_. Springer Science & Business Media, 2011.

<a id="ref-C17"></a>[[C17]](#ref-C17) Chodrow, Philip. "Divergence, entropy, information: An opinionated introduction to information theory." _arXiv preprint arXiv:1708.07459_ (2017).

<a id="ref-J57"></a>[[J57]](#ref-J57) Jaynes, Edwin T. "Information theory and statistical mechanics." _Physical review_ 106, no. 4 (1957): 620.