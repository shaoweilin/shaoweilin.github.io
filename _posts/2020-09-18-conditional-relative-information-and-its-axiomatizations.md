---
layout: post
title: Conditional relative information and its axiomatizations
---

In this post, we will study the conditional form of relative information. We will also look at how conditional relative information can be axiomatized and extended to non-real-valued measures.

This post is a continuation from our [series](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

## What is conditional relative information?

Suppose we have two random variables $$X, Y$$ and probability measures $$P, Q$$ on $$\Omega$$. We are interested in how far the model conditional $$P_{Y\vert X}$$ is to the true conditional $$Q_{Y\vert X}$$ on average over $$Q_X$$, and we want to ignore the model marginal $$P_X$$.  

Let $$P_{XY}, Q_{XY}$$ be the induced joint distributions for $$X, Y$$. We first construct a distribution $$R_{XY}$$ which has the same conditional distribution $$R_{Y\vert X} = P_{Y\vert X}$$ as the model but has a marginal $$R_X = Q_X$$ equal to that of the true distribution [[G11]](#ref-G11). Namely,

$$R_{XY}(F \times G) = \int_F P_{Y\vert X}(G\vert x) dQ_X(x)$$

where $$F$$ and $$G$$ are measurable sets over the state spaces of $$X$$ and $$Y$$ respectively.

We then define the _conditional relative information_ to be

$$I_{Q\Vert P}(Y\vert X) = I_{Q_{XY} \Vert R_{XY}}.$$

In the case where the corresponding densities are well-defined, we have

$$\begin{array}{rl} I_{Q\Vert P}(Y\vert X) &= \int \int q(y\vert x) \log \frac{q(y\vert x)}{p(y\vert x)}\, dy \,q(x)dx \\ & \\ &= \int \int q(y, x) \log \frac{q(y\vert x)}{p(y\vert x)} \,dy\, dx \end{array}.$$

which is the relative information to $$q(y\vert x)$$ from $$p(y\vert x)$$ averaged over $$q(x)$$.

## Whats is the chain rule for conditional relative information?

In statistics and machine learning, we often think of $$Q$$ as a true distribution that we are trying to uncover and $$P$$ as a model distribution for approximating $$Q$$. The relative information $$I_{Q\Vert P}$$ measures how far the model is to the truth.

To uncover the truth, it makes strategic sense to study different facets $$X, Y$$ of reality, and to build up reality one facet at a time. For example, we may want to know how far our model is to reality in modeling $$X$$ and focus on modeling $$X$$, before moving on to what our model says about both $$X, Y$$. The chain rule of conditional relative information says that the divergence of our model to reality for $$X, Y$$ is simply the sum of the divergences for $$X$$ and for $$Y \vert X$$:

$$\tag{CR} I_{Q\Vert P}(Y, X) = I_{Q\Vert P}(Y\vert X) + I_{Q\Vert P}(X).$$

Therefore, to get a good model of $$X, Y$$, we could attempt to minimize the divergences for $$X$$ and for $$Y \vert X$$ in parallel.

## How do we derive conditional entropy from conditional relative information?

Just as the entropy of a random variable $$X$$ with distribution $$P_X$$ can be defined as the relative information to the dependent distribution $$P_{XX}$$ from the independent distribution $$P_X \times P_X,$$ we will do the same for conditional entropy.

Given random variables $$X, Y$$ with joint distribution $$P_{XY}$$, we define the conditional entropy of $$Y \vert X$$ as

$$H(Y\vert X) = I_{P_{XY,XY}\Vert P_{XY} \times P_{XY}}(Y\vert X)$$

the conditional relative information of $$Y \vert X$$ to the dependent distribution $$P_{XY} \times P_{XY}$$ from the independent distribution $$P_{XY,XY}.$$

According to the chain rule of conditional relative information,

$$I_{P_{XY,XY}\Vert P_{XY} \times P_{XY}}(Y, X) = I_{P_{XY,XY}\Vert P_{XY} \times P_{XY}}(Y\vert X) + I_{P_{XY,XY}\Vert P_{XY} \times P_{XY}}(X).$$

By the definition of entropy in our [introduction](https://shaoweilin.github.io/building-foundations-of-information-theory-on-relative-information/), the first and third terms are the entropies $$H(Y,X)$$ and $$H(X)$$ respectively, while the second term is the conditional entropy $$H(Y\vert X)$$. Thus, we recover the classical chain rule for conditional entropy

$$H(Y,X) = H(Y\vert X) +H(X).$$

## Is there an axiomatization of conditional entropy?

As described in our [previous post](https://shaoweilin.github.io/building-foundations-of-information-theory-on-relative-information/), we allow the total measures of $$P, Q$$ to be different from one, but we require their total measures to be the same.

We start with an axiomatizations of conditional entropy with the hope of deriving axiomatizations of conditional relative information. I like the following categorical view of conditional entropy [[B11]](#ref-B11). I've taken the liberty of rewriting it in our notations.

Given a measured space $$(\Omega, \mathcal{B}, P)$$, a finite measurable function $$Y : \Omega \rightarrow S_Y$$ and a morphism $$f :S_Y \rightarrow S_X$$ between finite sets, let $$X = f \circ Y$$ and let $$P_Y, P_X$$ be the induced measures on $$S_Y, S_X$$.

In this case, the conditional entropy of $$Y \vert X$$ is

$$\begin{array}{rl} H(Y\vert X) &= H(Y,X)-H(X)\\ & \\ &= H(Y) - H(X) \\ & \\ &= - T\displaystyle\sum_{y} \bar{P}_Y(y) \log \bar{P}_Y(y) + T\displaystyle \sum_{x} \bar{P}_X(x) \log \bar{P}_X(x) \\ & \\&= -\displaystyle \sum_{y} P_Y(y) \log P_Y(y) + \displaystyle\sum_{x} P_X(x) \log P_X(x) \end{array}$$

where $$T$$ is the total measure of $$P$$, and $$\bar{P}_X = P_X/T$$ and $$\bar{P}_Y = P_Y/T$$ are probability measures.

Given two measured spaces $$(\Omega_1, \mathcal{B}_1, P_1)$$ and $$(\Omega_2, \mathcal{B}_2, P_2)$$, let $$(\Omega_1 \sqcup \Omega_2, \mathcal{B}_1 \oplus \mathcal{B}_2, P_1 \oplus P_2)$$ be their direct sum. Here, $$\Omega_1 \sqcup \Omega_2$$ is the disjoint union, and $$S \in \mathcal{B}_1 \oplus \mathcal{B}_2$$ if and only if $$S \cap \Omega_1 \in \mathcal{B}_1$$ and $$S \cap \Omega_2 \in \mathcal{B}_2$$. The measure of $$S$$ is the sum of that of $$S \cap \Omega_1$$ and of $$S \cap \Omega_2$$.

Let $$F_P$$ be a family of maps indexed by measures $$P$$ such that each $$F_P$$ sends morphisms $$f :S_X \rightarrow S_Y$$ between finite sets to $$[0, \infty)$$. Suppose that the family $$F_P$$ satisfies the following three axioms.

1.  _Functoriality._ $$F_P(f\circ g) = F_P(f) + F_P(g)$$
2.  _Homogeneity_. $$F_{\lambda P}(f) = \lambda F_P(f)$$
3.  _Additivity._ $$F_{P_1\oplus P_2}(f_1 \oplus f_2) = F_{P_1}(f_1) + F_{P_2}(f_2)$$
4.  _Continuity._ $$F$$ is continuous

Then, $$F_P(f)$$ must be $$c H(Y\vert X)$$ for some constant $$c \geq 0$$.

Given a classical conditional entropy $$H(Y\vert X)$$, we can now write this as $$F_P(f)$$ where $$f$$ is the projection $$(Y,X) \mapsto X$$.

The nice thing about the above categorical axiomatization of conditional entropy is that it fits into the view where the objects of study are spaces $$E, B$$ and fibrations $$\pi: E \rightarrow B$$ equipped with measures. The conditional entropy is the sum of the entropies of the fibers $$\pi^{-1}(b)$$ weighted by $$P_B(b)$$.

## Is there an axiomatization of conditional relative information?

We prefer to work with conditional relative information rather than conditional entropy. Its axiomatization should tell us how it behaves with respect to products and coproducts of the measures being compared.

Our axioms. Note the addition of the product rule. I'm not sure if the product axiom can be derived from the others when the state spaces are not finite. Perhaps it will follow from continuity and the fact that the limit of coproducts is the product of limits.

1.  _Functoriality_. $$G_{Q \Vert P} (f \circ g) = G_{Q \Vert P} (f) + G_{Q \Vert P} (g)$$
2.  _Homogeneity_. $$G_{\lambda Q \Vert \lambda P} (f) = \lambda G_{Q \Vert P} (f)$$
3.  _Coproduct_. $$G_{Q_1 \oplus Q_2 \Vert P_1 \oplus P_2} (f_1 \oplus f_2) = G_{Q_1 \Vert P_1} (f_1) + G_{Q_2 \Vert P_2} (f_2)$$
4.  _Product_. $$G_{Q_1 \times Q_2 \Vert P_1 \times P_2} (f_1,f_2) = T_2 G_{Q_1 \Vert P_1} (f_1) + T_1 G_{Q_2 \Vert P_2} (f_2)$$
5.  _Continuity_. $$G$$ is continuous.

Here, $$T_1$$ and $$T_2$$ are the total measures of $$Q_1$$ and $$Q_2$$ respectively.

The axioms for conditional entropy follow immediately from these axioms for conditional relative information, because we can write

$$F_{P_X}(f) = G_{TP_{XX} \Vert P_X \times P_X} (f)$$

where $$T$$ is the total measure of $$P_X$$.

## References

<a id="ref-B11"></a>[[B11]](#ref-B11) Baez, John C., Tobias Fritz, and Tom Leinster. "A characterization of entropy in terms of information loss." _Entropy_ 13, no. 11 (2011): 1945-1957.

<a id="ref-G11"></a>[[G11]](#ref-G11) Gray, Robert M. _Entropy and information theory_. Springer Science & Business Media, 2011.