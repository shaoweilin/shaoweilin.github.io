---
layout: post
title: Spiking neural networks
excerpt_separator: <!--more-->
---

In this article, we study a class of spiking neural network models based on continuous-time Markov chains with latent variables. Using our [theory](https://shaoweilin.github.io/convergence-of-biased-stochastic-approximation/) of online learning through the optimization of conditional relative entropy and variational inference, we derive local Hebbian learning rules for the spiking network which are provably convergent to local minima of the relative entropy objective.

This post is a continuation from our [series](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

<!--more-->

## How do we model a spiking network?

Let $$\mathcal{V}_X$$ be a finite set representing the collection of observed or sensory neurons whose states are determined completely by the state of the universe. For [simplicity](https://shaoweilin.github.io/biased-stochastic-approximation-for-latent-processes/#what-do-we-assume-about-the-true-distribution-the-model-and-the-learning-objective), we assume the sensory neurons capture enough information about the universe such that its true distribution is Markov. We may throw in hypothetical neurons until this is satisfied.

Let $$\mathcal{V}_Z$$ be a finite set representing the collection of latent or cortical neurons which computationally assist the spiking network in learning an approximation of the true distribution. The states of these latent neurons represent samples of beliefs that the spiking network has about the current state of the universe.

At each time $$t \geq 0,$$ each neuron $$i \in \mathcal{V}_X$$ has a state $$X_{it} \in \mathbb{N}\times \mathbb{B} = \{0,1,\ldots\}\times \{0,1\}$$ where the first natural number component represents the number of times the neuron has spiked since $$t=0$$ and the second binary component denotes if the spiking neuron is resting (state 0) or refractory (state 1). Similarly, each neuron $$i \in \mathcal{V}_Z$$ has a state $$Z_{it} \in \mathbb{N}\times\mathbb{B}.$$ Let $$X_t$$ denote the vector $$(X_{it}:i \in \mathcal{S}_X)$$ and $$Z_t$$ denote $$(Z_{it}: i \in \mathcal{S}_Z).$$

A neuron is in resting state if it is ready to fire when its membrane potential is sufficiently high. A neuron is in refractory state if it has just fired and is unable to fire again until its ion channels recover.

Let $$\mathcal{E}_Q$$ be a finite set representing the collection of edges or connections from the union $$\mathcal{S}_Z \cup \mathcal{S}_X$$ to $$\mathcal{S}_Z$$ that participate in the discriminative model of the spiking network. This discriminative model computes the distribution or belief of $$Z_{t+1}$$ given $$(Z_t, X_t).$$ The model plays an important role in learning. Note that there are no edges to $$\mathcal{S}_X$$ in this set of connections.

Let $$\mathcal{E}_P$$ be a finite set representing the collection of edges or connections from $$\mathcal{S}_Z \cup \mathcal{S}_X$$ to $$\mathcal{S}_Z \cup \mathcal{S}_X$$ that participate in the generative model of the spiking network. In the absense of sensory information $$X_{t+1}$$, this generative model estimates from $$(Z_t, X_t)$$ the distribution of $$(Z_{t+1}, X_{t+1})$$, $$(Z_{t+2}, X_{t+2})$$, etc. The model plays an important role in prediction.  

For each edge $$ij \in \mathcal{E}_Q$$ for some $$i \in \mathcal{S}_Z \cup \mathcal{S}_X$$ and $$j \in \mathcal{S}_Z,$$ we have a weight $$\lambda_{ij} \in \mathbb{R}$$ representing the synaptic strength of the corresponding connection. For each $$i \in \mathcal{S}_Z,$$ we also have a bias $$\lambda_i \in \mathbb{R}$$ representing the threshold potential of neuron $$i.$$ Let $$\lambda$$ denote the vector $$(\lambda_i, \lambda_{jk}: i\in \mathcal{S}_Z, jk \in \mathcal{E}_Q).$$ 

Similarly, for each edge $$ij \in \mathcal{E}_P$$ we have a weight $$\theta_{ij} \in \mathbb{R}$$, and for each $$i \in \mathcal{S}_Z$$ we have a bias $$\theta_i \in \mathbb{R}.$$ Let $$\theta$$ denote the vector $$(\theta_i, \theta_{jk}: i\in \mathcal{S}_Z \cup \mathcal{S}_X, jk \in \mathcal{E}_Q).$$ 


We use a discrete-time approximation when implementing the learning algorithm on a machine with a discrete processor clock.


## References

<a id="ref-KMMW19"></a>[[KMMW19]](#ref-KMMW19) Karimi, Belhal, Blazej Miasojedow, Éric Moulines, and Hoi-To Wai. "Non-asymptotic analysis of biased stochastic approximation scheme." _arXiv preprint arXiv:1902.00629_ (2019).
