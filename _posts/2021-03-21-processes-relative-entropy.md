---
layout: post
title: Processes I - Relative Entropy
---

We derive a distributed learning algorithm for spiking neural networks with latent variables that minimizes some natural notion of relative entropy and provably converges over time. We will model these spiking neural networks with stochastic processes: both discrete-time and continuous-time processes, with or without latent variables.

The actual process that generates the data could possibly be different from the model that we are training on the data. This problem is called model misspecification, which we are not addressing in this article. Our goal is to find a distribution in the model that fits the data well according to some sensible metric.

The outline of this article is as follows.

1.  Training stochastic processes using relative entropy
2.  Training stochastic processes using relative entropy rate
3.  Training latent processes without knowing the future
4.  Training latent processes with biased stochastic approximation
5.  Training Markov processes with biased stochastic approximation

In (1) and (2), we give a general overview of information-theoretic approaches to training stochastic processes. In (3), we zoom in on stochastic processes with latent variables, and measure how the lack of knowledge of the future affects online learning. In (4), we explore online algorithms involving biased stochastic approximation which may be seen as generalizations of classical expectation maximization. In (5), we zoom in on latent processes which are Markovian, and derive conditions that lead to convergence of the biased stochastic approximation.

This post is a continuation from our series on [spiking networks, path integrals and motivic information](https://shaoweilin.wordpress.com/2020/08/28/spiking-networks-path-integrals-and-motivic-information/).
