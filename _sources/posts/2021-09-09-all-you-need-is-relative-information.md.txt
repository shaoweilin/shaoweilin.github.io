---
date: 2021-09-09
excerpts: 2
---

# All you need is relative information

### Abstract
Relative information (relative entropy, KL divergence) and variational inference are powerful tools for deriving learning algorithms and their asymptotic properties, for both static systems and dynamic systems. The goal of this talk is to motivate a general online stochastic learning algorithm for stochastic processes with latent variables or memory, that provably converges under some regularity conditions. Please visit [https://bit.ly/3kmovql](https://bit.ly/3kmovql) for details.

In the first half of the talk, we study static systems, viewing maximum likelihood and Bayesian inference through the lens of relative information. In particular, their generalization errors may be derived by resolving the singularities of relative information. We then frame the two learning algorithms as special cases of variational inference with different computational constraints.

In the second half of the talk, we study dynamic systems, extending this variational inference method and computational perspective to stochastic processes and online learning. In particular, the training objective function will be a form of relative information which can be optimized iteratively in a way similar to expectation-maximization. The relative information objective provides a precise way to discuss the trade-off between exploration and exploitation during training.

### Details
[Math Machine Learning Seminar MPI MIS + UCLA](https://www.mis.mpg.de/calendar/lectures/2021/abstract-32595.html)

[Slides](https://w3id.org/people/shaoweilin/public/20210909-mpi-mis-ucla.pdf)

[YouTube](https://youtu.be/U2HnLtwgiqQ)

### Questions

There was a question about resources on singular learning theory. I could not remember the link during the talk, but below is a page with some materials. 

[Resources on singular learning](../singular/)

There was a question about the analyticity of relative information. Often for singular models, the relative information is not real-analytic. In my work with Mathias Drton and Carlos Amendola, we showed that the asymptotic properties of the statistical model can be derived from an analytic function that was, in some sense, equivalent to the relative information. In fact, we can go further to show that in many cases, the relative information is equivalent to a polynomial. More details in the slides below.

[Polynomial equivalence of the Kullback information for mixture models](https://w3id.org/people/shaoweilin/public/20180705-vilnius.pdf)

There was a request for more information about the sheaf perspective in my talk. I am working on a new post about "information type theory." There I will describe a partially-ordered set of spaces with bundles and sections between them. Relative information becomes a measure of complexity on this poset. It is natural to impose a topos structure on this poset, by describing the products, exponentials and subobject classifier. The topos structure then allows us to do logic on this poset. By using relative information in learning, we hope to discover a model of the observed universe with good logical properties. I also want to explore the extent quantum mechanics and quantum field theory can be embedded in this informational topos framework, in the spirit of Kochen-Specker's partial boolean algebras and Abramsky's logic of contextuality