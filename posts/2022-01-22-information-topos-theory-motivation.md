---
date: 2022-01-22
excerpts: 2
---

# Information topos theory - motivation

Relative information (also known as the Kullback-Leibler divergence) is an important fundamental concept in statistical learning and information theory. 

The (conditional) relative information

$$ \displaystyle
I_{q \Vert p}(Z|X) = \int q(X) \int q(Z|X) \log \frac{q(Z|X)}{p(Z|X)} dZ\, dX$$

gives the weighted divergence from a *model* conditional distribution $p(Z \vert X)$ to the *true* conditional distribution $q(Z \vert X),$ where the weighting is given by the true distribution $q(X.)$ We call it a *divergence* rather than a *distance* because it is not symmetric with respect to $q$ and $p.$

Relative information has two key aspects: divergence and conditioning. As a divergence, it says that the amount of information is relative, not absolute. As a function of a conditional relationship, it says that we should study the amount of information in something unknown while factoring in what we do know.

An (elementary) topos is a category that has just enough basic properties for us to do both geometry and logic. When studying morphisms between logical spaces in topoi, it might be useful to have a measure of information that is relative (giving us distances or divergences between points) and conditional (giving us the change in information through a morphism). Over the next few posts, we will explore examples of such uses in statistics, algebra, number theory and quantum physics.

In the same way that a Riemannian metric on a smooth manifold allows us to study curvatures and ultimately dynamical systems in general relativity via Riemannian geometry, and in the same way that the Fisher information metric on a smooth manifold allows us to study generalization and ultimately learning algorithms in statistics via information geometry, we think of information topos theory as a systematic way of associating an appropriate metric to a geometric topos space. Ultimately, the hope is to use this metric topos as a space-time substitute for understanding quantum mechanics and gravity.

In this post, we will motivate the development of an information theory for topoi, without invoking any logarithms, integrals, probabilities or measure theory. The mathematical foundations will be postponed to a later post.

## Bayesian bundles

John Baez and Tobias Fritz {cite}`baez2014bayesian` have a nice characterization of *conditional* relative information as the unique functor (up to scaling) from the category of probability distributions on finite sets to the additive monoid $[0,\infty]$ satisfying some basic linearity, convexity and continuity conditions. 

Instead of defining relative information as a function of a conditional relationship, they provide a measure-preserving morphism $f: X \rightarrow Y$
paired with a stochastic right inverse $s: Y \rightarrow X.$ I prefer to think in terms of a bundle $\pi_f : \Delta_X \rightarrow \Delta_Y$ that projects from the space of distributions over $X$ to the space of distributions over $Y$ by marginalization, and a section $\sigma_s : \Delta_Y \rightarrow \Delta_X$ that lifts a distribution over $Y$ to a distribution over $X$ by multiplication with a conditional distribution. Perhaps this interpretation was an intent of the authors too. 

Suppose we have a bundle $\pi: \Delta \rightarrow \Gamma$ and three sections - a true $\sigma_q : \Gamma \rightarrow \Delta$, a model $\sigma_p : \Gamma \rightarrow \Delta$ and a base point $\omega_q : * \rightarrow \Gamma.$ Here, $*$ is the one-point space. We may then consider the relative information for this bundle $\pi,$ from the model $\sigma_p$ to the true $\sigma_q,$ weighted by the distribution $\omega_q(*)$ (which you may think of as a prior).

This generalization beyond conditional relationships to bundles lets us define relative information for non-statistical categories.

## From bundles to presheaves

If we have a bundle $\pi:\Delta \rightarrow \Gamma$ and some topology $C$ of open sets over the base space $\Gamma,$ we may construct a presheaf $U: C^\text{op}\rightarrow \text{Set}$ from $\pi$ by associating to each open set the set of sections over that open set. See the tutorial {cite}`baez2020topos2` by John Baez for an excellent description of this construction.

For the rest of this post, we will be using presheaves instead of bundles, but the two ideas are somewhat interchangeable if you prefer to think in terms of bundles.

So far, we have glossed over what it means to be an "open set" of the base space $\Gamma.$ In statistics, when $\Gamma$ is a space of distributions, there is a natural choice of topology inherited from the topology of the real numbers which are used in defining the distributions or measures. However, we _want to avoid_ this topology. 

Instead, we want to think of $\Gamma$ as a kind of _context_ and an open set of $\Gamma$ should be some kind of _subcontext_. This subcontext should lead us to other bundles and sections. In our example involving the bundles $\Delta \rightarrow \Gamma$ and $\Gamma \rightarrow *,$ we want to think of $\Delta, \Gamma, *$ as contexts in our category $C.$ We then have sections $\sigma_q, \sigma_p \in U(\Gamma)$ and $\omega_q \in U(*)$ where $U : C^\text{op} \rightarrow \text{Set}$ is our presheaf. We will make these ideas more rigorous later.

## Presheaf catagories

We now have relative information for a presheaf $U : C^\text{op} \rightarrow \text{Set},$ a base space or object $\Gamma$ in $C$, and sections $\sigma_q, \sigma_p \in U(\Gamma).$ By Yoneda's lemma, the sections $\sigma \in U(\Gamma)$ are in one-to-one correspondence with natural transformations 

$$\eta_\sigma : \text{Hom}(-, \Gamma) \rightarrow U$$

where $\text{Hom}(-, \Gamma) : C^\text{op} \rightarrow \text{Set}$ is also a presheaf.

Therefore, it makes sense to consider relative information within a larger _presheaf category_ where the objects are presheaves (which are functors $C^\text{op} \rightarrow \text{Set}$) and the morphisms are natural transformations between then presheaves. This presheaf category is an example of an _elementary topos_ {cite}`baez2020topos5`.

## Natural models

Recall that we had a presheaf $U : C^\text{op} \rightarrow \text{Set},$ a context $\Gamma$ in $C$ and a section $\sigma \in U(\Gamma).$ In the statistical setting, the section is a map $\sigma : \Gamma \rightarrow \Delta.$ We want to think of the image $\Gamma.\sigma := \sigma(\Gamma)$ also as a context, with the projection $\Gamma.\sigma \rightarrow \Gamma$ as a morphism in $C.$

In the non-statistical setting, how do we go about extending the context $\Gamma$ by a section $\sigma \in U(\Gamma)?$ How do we check if $\Gamma.\sigma$ is a context in the category $C$ without forcefully requiring $C$ to be expanded to include these extensions?

Awodey has a beautiful way of constructing context extensions by imposing a simple requirement on the presheaf category {cite}`awodey2018natural`. He begins with a fixed natural transformation $p: \dot{U} \rightarrow U$ of presheaves on a category $C.$ He then requires this natural transformation to be _representable_, i.e. its pullback along any natural transformation $\text{Hom}(-,\Gamma) \rightarrow U$ is representable.

Given a representable natural transformation $p: \dot{U} \rightarrow U,$ we can build a dependent type theory with $C$ as the category of contexts and substitutions in the theory. The map $p$ then takes each term in the universe $\dot{U}$ to its type in $U.$ A representable natural transformation (together with a choice of pullbacks in some presheaf category) is also called a _natural model of type theory_.

## Conclusion

In our next post, we will continue with motivations for information topos theory and discuss posetal subcategories of our presheaf category. Relative information will then be defined as the divergence between two sections of a poset with respect to a morphism in the poset.

## References

```{bibliography}
:filter: docname in docnames
```