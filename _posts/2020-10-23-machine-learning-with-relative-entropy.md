---
layout: post
title: Machine learning with relative entropy
---

We will reframe some common machine learning paradigms, such as maximum likelihood, stochastic gradients, stochastic approximation and variational inference, in terms of relative entropy.

This post is a continuation from our series on [spiking networks, path integrals and motivic information](https://shaoweilin.wordpress.com/2020/08/28/spiking-networks-path-integrals-and-motivic-information/).

### What is a statistical model?

Given a measurable space $$(\Omega, \mathcal{B})$$, let $$\Delta_\Omega$$ denote the set of distributions or probability measures on $$\Omega$$. A statistical model is a family of distributions

$$P_\theta : \Delta_{\Omega}, \quad \theta\in \Theta$$

parametrized by some space $$\Theta.$$

Suppose that the true distribution is some unknown $$Q \in \Delta_\Omega.$$ The goal of statistical learning is to approximation $$Q$$ with some model distribution $$P_\theta.$$

### How do we search for the best model?

Finding the best model is a common problem not just in machine learning but also in science. We want a model that explains reality well, but also with as few parameters as possible. This latter requirement is called Occam's Razor. It says that the simplest explanation is most likely the best one.

Finding the simplest explanation is easier said than done. We want to explain the observed data well, but we do not want to overfit the data. At the same time, we also do not want a model that is too overly simplistic, e.g. the uniform distribution. Many modern strategies use a complexity score or regularizer that penalizes the model based on the number of parameters or the dimension of the model. However, the choice of regularizer can seem rather arbitrary at times.

If we have access to the true distribution $$Q \in \Delta_\Omega,$$ life would be a lot easier. We could try to measure the distance of a model $$P_\theta$$ to the true distribution, and attempt to minimize this distance. A natural choice for the distance is the relative entropy to $$Q$$ from $$P_\theta$$. We may also interpret the relative entropy as the average number of informational bits required to encode data from true distribution using the model, up to some additive constant (see Minimum Description Length, Kolmogorov Complexity and Stochastic Complexity for related concepts). Therefore, minimizing relative entropy is beneficial from the computational resource management point of view.

However, we do not have access to the true distribution. There are two general strategies for overcoming this obstacle. For simplicity, let us assume there is a probability measure $$M$$ such that $$Q \ll M$$ and $$P_\theta \ll M$$, i.e. both $$Q$$ and $$P_\theta$$ are absolutely continuous with respect to $$M$$. In this case, the densities $$dQ/dM$$ and $$dP_\theta/dM$$ exist.

### First method (maximum likelihood)

We estimate the relative entropy from the observed data. We may therefore write the relative entropy as

$$\displaystyle \begin{array}{rl} H_{Q \Vert P_\theta} &=\displaystyle\int \log \frac{dQ/dM}{dP_\theta/dM} dQ \\ & \\&= \displaystyle\mathbb{E}_{X\sim Q} \left[ \log \frac{dQ}{dM}(X) \right] - \mathbb{E}_{X\sim Q} \left[ \log \frac{dP_\theta}{dM}(X) \right]. \end{array}$$

The first term is a constant that does not depend on $$\theta$$, so minimizing the relative entropy is equivalent to maximizing the second term. The second term may be approximated using the average

$$\displaystyle \mathbb{E}_{X\sim Q} \left[ \log P_\theta(X) \right] = \frac{1}{\left\vert  \mathcal{D}\right\vert }\sum_{x \in \mathcal{D}} \log P_\theta (x)$$

where $$\mathcal{D}$$ is a finite i.i.d. sample of $$Q$$. This approximation is called the _log -likelihood_ of the data set $$\mathcal{D}$$. The best model parameter $$\hat{\theta}$$ may be estimated by maximizing the log-likelihood. The resulting algorithm is known as the _maximum likelihood_ method.

If the true distribution can be represented by $$Q = P_{\theta^*}$$ for some parameter $$\theta^*,$$ it can be shown that as the sample size grows to infinity, the estimated parameter $$\hat{\theta}$$ converges quickly to the true parameter $$\theta^*.$$ The asymptotic behavior of the maximum likelihood method is analyzed in [Watanabe09].

### Second method (stochastic gradient)

We estimate the _gradient_ of the relative entropy from observed data. This gradient can be written as

$$\begin{array}{rl} \displaystyle \frac{d}{d\theta} H_{Q \Vert P_\theta} &= \displaystyle \frac{d}{d\theta} \mathbb{E}_{X\sim Q} \left[ \log \frac{dQ}{dM}(X) \right] - \frac{d}{d\theta} \mathbb{E}_{X\sim Q} \left[ \log \frac{dP_\theta}{dM}(X) \right] \\ & \\&= \displaystyle- \mathbb{E}_{X\sim Q} \left[\frac{d}{d\theta} \log \frac{dP_\theta}{dM}(X) \right] \end{array}$$

assuming that the expectation and derivative commute by some convergence theorem. This last term can be approximated by the average

$$\displaystyle \mathbb{E}_{X\sim Q} \left[ \frac{d}{d\theta} \log P_\theta(X) \right] = \frac{1}{\left\vert  \mathcal{B}\right\vert }\sum_{x \in \mathcal{B}} \frac{d}{d\theta} \log P_\theta (x)$$

where the _batch_ $$\mathcal{B}$$ is a finite i.i.d. sample of $$Q$$. This approximation is called the _score_. We often sample batches of a small fixed size with replacement from a large data set $$\mathcal{D}$$ and perform gradient ascent via the batch score. The resulting algorithm is known as the _stochastic gradient_ method.

Batch stochastic gradients have a regularizing effect on the estimator $$\hat{\theta}$$ as compared to the gradient of the log-likelihood of the full data set. They ensure that the estimator does not get stuck in a local minima of the log-likelihood function, which corresponds to overfitting of the model to the full data set. The asymptotic behavior of the stochastic gradient method is analyzed in [Watanabe09].

### How do we apply stochastic approximation via relative entropy?

In the stochastic approximation theory of Robbins-Monro, given an increasing function $$f(\theta)$$ which cannot be observed directly and has a unique root

$$f(\theta^*) = 0,$$

the goal is to find this root. We are however given a random variable $$F(\theta)$$ that equals $$f(\theta)$$ in expectation, i.e.

$$\mathbb{E}[F(\theta)] = f(\theta).$$

The proposed algorithm is to iteratively update

$$\theta_{n+1} = \theta_n - \eta_n F(\theta_n).$$

where the $$\eta_n$$ is a pre-determined sequence of step sizes. Under some weak conditions, the estimates $$\theta_n$$ will converge in probability to $$\theta^*.$$

We may apply stochastic approximation theory to stochastic optimization problems where the goal is to minimize some

$$g(\theta) = \mathbb{E} [ G(\theta) ].$$

and the minimizer $$\theta^*$$ satisfies $$dg/d\theta(\theta^*) = 0.$$ If the expectation and derivative commute, then

$$\displaystyle \frac{dg}{d\theta}(\theta) = \mathbb{E} \left[ \frac{dG}{d\theta}(\theta) \right]$$

so we may apply stochastic approximation techniques with $$F = dG/d\theta$$.

Applying stochastic optimization to relative entropy minimization, we may let

$$\displaystyle G = \log \frac{dP_\theta}{dM}$$

where $$P_\theta$$ and $$M$$ were defined above for stochastic gradients. As a result, we get a proof that the stochastic gradient algorithm is consistent, i.e. the parameter updates $$\theta_n$$ tend to the true parameter $$\theta^*$$. Stronger results may be attained if the model $$P_\theta$$ satisfy additional regularity conditions.

### How do we frame variational inference in terms of relative entropy?

A latent variable is a random variable for which we have no data. A latent variable model is a statistical model $$P_\theta$$ with some observed variable $$X$$ and some latent variable $$Z$$. The marginal distribution of $$X$$ is given by the integral

$$P_\theta(X) = \int P_\theta(X,Z) dZ.$$

For example, we could have a Gaussian mixture model where we have samples from $$k$$ Gaussian distributions $$\mathcal{N}_{\mu_1, \Sigma_1}, \ldots, \mathcal{N}_{\mu_k, \Sigma_k}$$, but for each sample $$X$$ we do not know the label $$Z \in \{1, \ldots, k\}$$ of the Gaussian distribution it was drawn from. The joint distribution of $$X, Z$$ is

$$P_\theta(X,Z) = \alpha_Z \mathcal{N}_{\mu_Z, \Sigma_Z}(X)$$

where $$\alpha_1, \ldots, \alpha_k$$ are the non-negative mixing weights that sum to one, and

$$\theta = (\alpha_1, \ldots, \alpha_k, \mu_1, \ldots,\mu_k, \Sigma_1, \ldots, \Sigma_k).$$

The marginal distribution of $$X$$ is

$$P_\theta(X) = \alpha_1 \mathcal{N}_{\mu_1, \Sigma_1}(X) + \cdots + \alpha_k \mathcal{N}_{\mu_k, \Sigma_k}(X).$$

Training latent variable models is notoriously difficult, because the usual maximum likelihood or stochastic gradient methods involve computing the derivative of the logarithm of the above integral which is often tedious.

Instead, we will consider a variational approach. Here, _variational_ means that we will introduce a new _functional_ parameter to the optimization problem, e.g. a parameter which is some function $$q: S \rightarrow \mathbb{R}$$. If the set $$S$$ is infinite, we can think of $$q$$ as an infinite-dimensional vector with entries $$q(s), s\in S.$$

Again, we begin with the goal of minimizing the relative entropy  
$$H_{Q \Vert P_\theta}(X)$$ to the true distribution $$Q(X)$$ from the model distribution $$P_\theta(X)$$ for the observable $$X$$. We now introduce a variational parameter $$q(Z\vert X)$$ and let

$$q(Z,X) = q(Z\vert X) Q(X).$$

By the [chain rule](https://shaoweilin.wordpress.com/2020/09/18/conditional-relative-entropy-and-its-axiomatizations/) for relative entropy,

$$\begin{array}{rl} H_{q\Vert P_\theta}(Z, X) &= H_{q\Vert P_\theta}(Z\vert X) + H_{q\Vert P_\theta}(X) \\& \\&= H_{q\Vert P_\theta}(Z\vert X) + H_{Q\Vert P_\theta}(X) .\end{array}$$

Therefore,

$$H_{q\Vert P_\theta}(Z, X) \geq H_{Q\Vert P_\theta}(X) $$

with equality if and only if $$q(Z\vert X) = P_\theta(Z\vert X).$$

Now, if $$q(Z\vert X)$$ is allowed to be any distribution and if a pair $$(q^*, \theta^*)$$ minimizes $$H_{q\Vert P_\theta}(Z, X)$$, then $$\theta^*$$ minimizes $$H_{Q\Vert P_\theta}(X).$$ To see this, let $$\theta'$$ be a minimizer of $$H_{Q\Vert P_\theta}(X)$$ and let $$q'(Z\vert X) = p_{\theta'}(Z\vert X)$$. Then,

$$\begin{array}{rl} H_{Q\Vert P_{\theta^*}}(X) &\geq H_{Q\Vert P_{\theta'}}(X) \\ & \\ &= H_{q' \Vert P_{\theta'}}(Z,X) \\ & \\ &\geq H_{q^* \Vert P_{\theta^*}}(Z,X)\geq H_{Q\Vert P_{\theta^*}}(X) \end{array}$$

where the first and third inequalities follow from the minimizer assumptions, and the second and fourth equality/inequality follow from the chain rule. Hence, all the inequalities are equalities and $$H_{Q\Vert P_{\theta^*}}(X) = H_{Q\Vert P_{\theta'}}(X)$$, so $$\theta^*$$ is a minimizer as claimed.

This result suggests a variational strategy for training the latent variable model --- introduce a variational parameter $$q(Z\vert X)$$ and minimize the relative entropy $$H_{q\Vert P_\theta}(Z, X)$$ over $$q$$ and $$\theta.$$

Moreover, we may perform alternating minimization:

1.  holding $$\theta$$ constant while minimizing over $$q$$;
2.  holding $$q$$ constant while minimizing over $$\theta$$;
3.  repeat.

Traditionally, the first step is called Expectation while the second step is called Maximization (of the log-likelihood, as opposed to minimization of the relative entropy). This technique is known as the _EM algorithm_.

Sometimes, the variational parameter $$q(Z\vert X)$$ is constrained to a space of tractable conditional distributions, or a space of distributions for which the maximization step has an exact solution. In these cases, the optimal value of $$H_{q\Vert P_\theta}(Z, X)$$ may not equal the optimal value of $$H_{Q \Vert P_\theta}(X)$$, but it will be an upper bound to the latter. The EM algorithm becomes an approximate inference technique, because it minimizes an upper bound and not the desired relative entropy itself.

### References

[Watanabe09] Watanabe, Sumio. _Algebraic geometry and statistical learning theory_. Vol. 25\. Cambridge university press, 2009.