---
date: 2014-08-02
excerpts: 1
---

# Boltzmann machines and hierarchical models

The [restricted Boltzmann machine](http://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) (RBM) is a key statistical model used in [deep learning](http://en.wikipedia.org/wiki/Deep_learning). They are special form of Boltzmann machines where the underlying graph is a bipartite graph. Personally, I am more interested in [Boltzmann machines](http://en.wikipedia.org/wiki/Boltzmann_machine) because they represent a class of discrete [energy models](http://www.cs.nyu.edu/~yann/research/ebm/) where the energy is quadratic. The dynamics of the model bears a lot of resemblance to those of [Hopfield networks](http://en.wikipedia.org/wiki/Hopfield_network) and [Ising models](http://en.wikipedia.org/wiki/Ising_model). As an aside, normal distributions are continuous energy models where the energy is quadratic and positive definite. 

If the energy of the model is a polynomial of higher degree (e.g. cubic, quartic), then the model is _hierarchical_. They are a kind of [graphical model](http://en.wikipedia.org/wiki/Graphical_model) where the underlying graph is a [simplicial complex](http://en.wikipedia.org/wiki/Simplicial_complex) (a special type of [hypergraph](http://en.wikipedia.org/wiki/Hypergraph)). Here are some slides and papers on hierachical models:

1. [Hierarchical models and monomial ideals](http://www.fields.utoronto.ca/programs/scientific/11-12/graphicmodels/Wynn_Fields2.pdf)<br />
_Daniel Bruynooghe and Henry Wynn_

2. [Toric ideals in algebraic statistics](http://www4.ncsu.edu/~smsulli2/Pubs/thesis.pdf)<br />
_Seth Sullivant_

3. [Decomposable models](http://www.math.vt.edu/people/fhinkel/AlgebraicStatistics/DecomposableModels2.pdf)<br />
_Franziska Hinkelmann_

4. [Betti numbers of Stanley-Reisner rings determine hierarchical Markov degrees](http://arxiv.org/abs/0910.1610)<br />
_Sonja Petrović and Erik Stokes_

5. [Hierarchical models for contingency tables from the viewpoint of abstract simplicial complex](http://park.itc.u-tokyo.ac.jp/atstat/takemura-talks/0812casta.pdf)<br />
_Akimichi Takemura_
