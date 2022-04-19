---
layout: post
title: Machine learning with relative information
---

We will reframe some common machine learning paradigms, such as maximum likelihood, stochastic gradients, stochastic approximation and variational inference, in terms of relative information.

This post is a continuation from our [series](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

## What is a statistical model?

Given a measurable space $$(\Omega, \mathcal{B})$$, let $$\Delta_\Omega$$ denote the set of distributions or probability measures on $$\Omega$$. A statistical model is a family $$\{P_\theta\}$$ of distributions

$$P_\theta : \Delta_{\Omega}, \quad \theta\in \Theta$$

parametrized by some space $$\Theta.$$

Suppose that the true distribution is some unknown $$Q_* \in \Delta_\Omega.$$ The goal of statistical learning is to approximation $$Q_*$$ with some model distribution $$P_\theta.$$

## How do we search for the best model?

Finding the best model is a common problem not just in machine learning but also in science. We want a model that explains reality well, but also with as few parameters as possible. This latter requirement is called Occam's Razor. It says that the simplest explanation is most likely the best one.

Finding the simplest explanation is easier said than done. We want to explain the observed data well, but we do not want to overfit the data. At the same time, we also do not want a model that is too overly simplistic, e.g. the uniform distribution. Many modern strategies use a complexity score or regularizer that penalizes the model based on the number of parameters or the dimension of the model. However, the choice of regularizer can seem rather arbitrary at times.

If we have access to the true distribution $$Q_* \in \Delta_\Omega,$$ life would be a lot easier. We could try to measure the distance of a model distribution $$P_\theta$$ to the true distribution, and attempt to minimize this distance. A natural choice for the distance is the relative information to $$Q_*$$ from $$P_\theta$$. 

We may also interpret the relative information as the average number of informational bits required to encode data from true distribution using the model, up to some additive constant (see Minimum Description Length, Kolmogorov Complexity and Stochastic Complexity for related concepts). Therefore, minimizing relative information is beneficial from the computational resource management point of view.

However, we do not have access to the true distribution. There are two general strategies for overcoming this obstacle. For simplicity, let us assume there is a probability measure $$M$$ such that $$Q_* \ll M$$ and $$P_\theta \ll M$$, i.e. both $$Q_*$$ and $$P_\theta$$ are absolutely continuous with respect to $$M$$. In this case, the densities $$dQ_*/dM$$ and $$dP_\theta/dM$$ exist.

## First method (maximum likelihood)

We estimate the relative information from the observed data. We may therefore write the relative information as

$$\displaystyle \begin{array}{rl} I_{Q_* \Vert P_\theta} &=\displaystyle\int \log \frac{dQ_*/dM}{dP_\theta/dM} dQ_* \\ & \\&= \displaystyle\mathbb{E}_{X\sim Q_*} \left[ \log \frac{dQ_*}{dM}(X) \right] - \mathbb{E}_{X\sim Q_*} \left[ \log \frac{dP_\theta}{dM}(X) \right]. \end{array}$$

The first term is a constant that does not depend on $$\theta$$, so minimizing the relative information is equivalent to maximizing the second term. The second term may be approximated using the average

$$\displaystyle \mathbb{E}_{X\sim Q_*} \left[ \log P_\theta(X) \right] = \frac{1}{\left\vert  \mathcal{D}\right\vert }\sum_{x \in \mathcal{D}} \log P_\theta (x)$$

where $$\mathcal{D}$$ is a finite i.i.d. sample of $$Q_*$$. This approximation is called the _log -likelihood_ of the data set $$\mathcal{D}$$. The best model parameter $$\hat{\theta}$$ may be estimated by maximizing the log-likelihood. The resulting algorithm is known as the _maximum likelihood_ method.

If the true distribution can be represented by $$Q_* = P_{\theta^*}$$ for some parameter $$\theta^*,$$ it can be shown that as the sample size grows to infinity, the estimated parameter $$\hat{\theta}$$ converges quickly to the true parameter $$\theta^*.$$ The asymptotic behavior of the maximum likelihood method is analyzed in [[W09]](#ref-W09).

## Second method (stochastic gradient)

We estimate the _gradient_ of the relative information from observed data. This gradient can be written as

$$\begin{array}{rl} \displaystyle \frac{d}{d\theta} I_{Q_* \Vert P_\theta} &= \displaystyle \frac{d}{d\theta} \mathbb{E}_{X\sim Q_*} \left[ \log \frac{dQ_*}{dM}(X) \right] - \frac{d}{d\theta} \mathbb{E}_{X\sim Q_*} \left[ \log \frac{dP_\theta}{dM}(X) \right] \\ & \\&= \displaystyle- \mathbb{E}_{X\sim Q_*} \left[\frac{d}{d\theta} \log \frac{dP_\theta}{dM}(X) \right] \end{array}$$

assuming that the expectation and derivative commute by some convergence theorem. This last term can be approximated by the average

$$\displaystyle \mathbb{E}_{X\sim Q_*} \left[ \frac{d}{d\theta} \log P_\theta(X) \right] = \frac{1}{\left\vert  \mathcal{B}\right\vert }\sum_{x \in \mathcal{B}} \frac{d}{d\theta} \log P_\theta (x)$$

where the _batch_ $$\mathcal{B}$$ is a finite i.i.d. sample of $$Q_*$$. This approximation is called the _score_. We often sample batches of a small fixed size with replacement from a large data set $$\mathcal{D}$$ and perform gradient ascent via the batch score. The resulting algorithm is known as the _stochastic gradient_ method.

Batch stochastic gradients have a regularizing effect on the estimator $$\hat{\theta}$$ as compared to the gradient of the log-likelihood of the full data set. They ensure that the estimator does not get stuck in a local minima of the log-likelihood function, which corresponds to overfitting of the model to the full data set. The asymptotic behavior of the stochastic gradient method is analyzed in [[W09]](#ref-W09).

## How do we apply stochastic approximation via relative information?

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

Applying stochastic optimization to relative information minimization, we may let

$$\displaystyle G = \log \frac{dP_\theta}{dM}$$

where $$P_\theta$$ and $$M$$ were defined above for stochastic gradients. As a result, we get a proof that the stochastic gradient algorithm is consistent, i.e. the parameter updates $$\theta_n$$ tend to the true parameter $$\theta^*$$. Stronger results may be attained if the model distributions $$P_\theta$$ satisfy additional regularity conditions.

## How do we frame variational inference in terms of relative information?

A latent variable is a random variable for which we have no data. A latent variable model is a statistical model $$\{P_\theta\}$$ with some observed variable $$X$$ and some latent variable $$Z$$. The marginal distribution of $$X$$ is given by the integral

$$P_\theta(X) = \int P_\theta(X,Z) dZ.$$

For example, we could have a Gaussian mixture model where we have samples from $$k$$ Gaussian distributions $$\mathcal{N}_{\mu_1, \Sigma_1}, \ldots, \mathcal{N}_{\mu_k, \Sigma_k}$$, but for each sample $$X$$ we do not know the label $$Z \in \{1, \ldots, k\}$$ of the Gaussian distribution it was drawn from. The joint distribution of $$X, Z$$ is

$$P_\theta(X,Z) = \alpha_Z \mathcal{N}_{\mu_Z, \Sigma_Z}(X)$$

where $$\alpha_1, \ldots, \alpha_k$$ are the non-negative mixing weights that sum to one, and

$$\theta = (\alpha_1, \ldots, \alpha_k, \mu_1, \ldots,\mu_k, \Sigma_1, \ldots, \Sigma_k).$$

The marginal distribution of $$X$$ is

$$P_\theta(X) = \alpha_1 \mathcal{N}_{\mu_1, \Sigma_1}(X) + \cdots + \alpha_k \mathcal{N}_{\mu_k, \Sigma_k}(X).$$

Training latent variable models is notoriously difficult, because the usual maximum likelihood or stochastic gradient methods involve computing the derivative of the logarithm of the above integral which is often tedious.

Instead, we will consider a variational approach. Here, _variational_ means that we will introduce a new _functional_ parameter to the optimization problem, e.g. a parameter which is some function $$Q: S \rightarrow \mathbb{R}$$. If the set $$S$$ is infinite, we can think of $$Q$$ as an infinite-dimensional vector with entries $$Q(s), s\in S.$$

Again, we begin with the goal of minimizing the relative information  
$$I_{Q_* \Vert P_\theta}(X)$$ to the true distribution $$Q_*(X)$$ from the model distribution $$P_\theta(X)$$ for the observable $$X$$. We now introduce a variational parameter $$Q(Z\vert X)$$ and let

$$Q(Z,X) = Q(Z\vert X) Q_*(X).$$

By the chain rule ([CR](https://shaoweilin.github.io/conditional-relative-information-and-its-axiomatizations/#mjx-eqn-CR)) for relative information,

$$\begin{array}{rl} I_{Q\Vert P_\theta}(Z, X) &= I_{Q\Vert P_\theta}(Z\vert X) + I_{Q\Vert P_\theta}(X) \\& \\&= I_{Q\Vert P_\theta}(Z\vert X) + I_{Q_*\Vert P_\theta}(X) .\end{array}$$

Therefore,

$$I_{Q\Vert P_\theta}(Z, X) \geq I_{Q_*\Vert P_\theta}(X) $$

with equality if and only if $$Q(Z\vert X) = P_\theta(Z\vert X).$$

Now, if $$Q(Z\vert X)$$ is allowed to be any distribution and if a pair $$(\hat{Q}, \hat{\theta})$$ minimizes $$I_{Q\Vert P_\theta}(Z, X)$$, then $$\hat{\theta}$$ minimizes $$I_{Q_*\Vert P_\theta}(X).$$ To see this, let $$\tilde{\theta}$$ be a minimizer of $$I_{Q_*\Vert P_\theta}(X)$$ and let $$\tilde{Q}(Z\vert X) = p_{\tilde{\theta}}(Z\vert X)$$. Then,

$$\begin{array}{rl} I_{Q_*\Vert P_{\hat{\theta}}}(X) &\geq I_{Q_*\Vert P_{\tilde{\theta}}}(X) \\ & \\ &= I_{\tilde{Q} \Vert P_{\tilde{\theta}}}(Z,X) \\ & \\ &\geq I_{\hat{Q} \Vert P_{\hat{\theta}}}(Z,X)\geq I_{Q_*\Vert P_{\hat{\theta}}}(X) \end{array}$$

where the first and third inequalities follow from the minimizer assumptions, and the second and fourth equality/inequality follow from the chain rule. Hence, all the inequalities are equalities and $$I_{Q_*\Vert P_{\hat{\theta}}}(X) = I_{Q_*\Vert P_{\tilde{\theta}}}(X)$$, so $$\hat{\theta}$$ is a minimizer as claimed.

This result suggests a variational strategy for training the latent variable model --- introduce a variational parameter $$Q(Z\vert X)$$ and minimize the relative information $$I_{Q\Vert P_\theta}(Z, X)$$ over $$Q$$ and $$\theta.$$

Moreover, we may perform coordinate-wise minimization:

1.  holding $$\theta$$ constant while minimizing over $$Q$$;
2.  holding $$Q$$ constant while minimizing over $$\theta$$;
3.  repeat.

This minimization can be done in an alternating fashion, i.e. fix $$\theta_n$$ to find $$Q_{n+1},$$ then fix $$Q_{n+1}$$ to find $$\theta_{n+1}$$; or it can be done in parallel, i.e. fix $$\theta_n$$ to find $$Q_{n+1},$$ and fix $$Q_n$$ to find $$\theta_{n+1}$$ at the same time.

In information geometry [[A95]](#ref-A95), the first step is called _exponential projection_ (e-projection) while the second step is called _mixture projection_ (m-projection) (of the log-likelihood, as opposed to minimization of the relative information). This _exponential-mixture algorithm_ (em algorithm) is also related to the _expectation-maximization algorithm_ (EM algorithm) [[DLR77]](#ref-DLR77).

Sometimes, the variational parameter $$Q(Z\vert X)$$ is constrained to a space of tractable conditional distributions, or a space of distributions for which the maximization step has an exact solution. In these cases, the optimal value of $$I_{Q\Vert P_\theta}(Z, X)$$ may not equal the optimal value of $$I_{Q_* \Vert P_\theta}(X)$$, but it will be an upper bound to the latter. The em algorithm becomes an approximate inference technique, because it minimizes an upper bound and not the desired relative information itself.

From a computational point of view, gaining efficiency in inference at the cost of approximation is a good thing, especially if there are performance guarantees in the form of an upper bound.

## How do we interpret the variational parameter $$Q(Z\vert X)$$?

We consider a variety of contexts where the interpretations of $$Q(Z\vert X)$$ are different.

### Context of Bayesian statistics

Historically, variational inference was introduced [[JGJS99]](#ref-JGJS99) to approximate posterior distributions in the context of Bayesian statistics, as an alternative to Markov chain Monte Carlo methods [[BKM17]](#ref-BKM17). 

Suppose we have a distribution $$P(X\vert Z)$$ and prior $$P(Z)$$ for some observed variable $$X$$ and some latent variable $$Z.$$ The goal is to find the posterior distribution $$P(Z\vert X_*)$$ given some data $$X_*.$$ 

Variational inference frames this goal as an optimization problem. Let $$Q_*(X)$$ be the atomic distribution with $$Q_*(X_*)=1.$$ Let $$\Delta$$ be the space of joint distributions on $$Z$$ and $$X$$ which factors as 

$$Q(Z,X) = Q_*(X) Q(Z\vert X)$$

for some variational parameter $$Q(Z\vert X).$$

We claim that the desired posterior $$P(Z\vert X_*)$$ is the value of the parameter $$Q(Z\vert X_*)$$ which minimizes the conditional relative information $$I_{Q\Vert P}(Z, X)$$ as $$Q(Z,X)$$ varies over $$\Delta.$$ To see this, recall that 

$$\begin{array}{rl} I_{Q\Vert P}(Z, X) = I_{Q\Vert P}(Z\vert X) + I_{Q_*\Vert P}(X) .\end{array}$$

In this sum, the second term is a constant that does not depend on the parameter $$Q(Z\vert X).$$ The first term expands to

$$\begin{array}{rl}  I_{Q\Vert P}(Z\vert X) &= \displaystyle \int Q(X) \int Q(Z\vert X) \log \frac{Q(Z\vert X)}{P(Z\vert X)} \,dZ \,dX \\ &= \displaystyle \int Q(Z\vert X_*) \log \frac{Q(Z\vert X_*)}{P(Z\vert X_*)} \,dZ \end{array}
$$

which is minimized precisely when $$Q(Z\vert X_*) = P(Z\vert X_*).$$

Often, we only want to consider distributions $$Q(Z\vert X)$$ which approximate the posterior $$P(Z\vert X)$$ and have good computational properties. For example, we may want to store $$Q(Z\vert X)$$ efficiently in a low-dimensional space, or we may want to compute $$Q(Z\vert X)$$ quickly as a composition of simple functions. Through variational inference, we may find this approximation by 
optimizing over a smaller space $$\Delta'$$ of distributions with the desired computational properties. 

In the context of Bayesian statistics, the variational parameter $$Q(Z\vert X)$$ is a computationally-efficient approximation of the model posterior $$P(Z\vert X).$$

### Context of statistical learning

A second context we may consider is the goal of learning the true distribution $$Q_*(X).$$ We solve this problem by proposing two models - a generative model $$\{P(Z,X)\}$$ where the marginals $$P(X)$$ _approximates_ $$Q_*(X),$$ and a discriminative model $$\{Q(Z\vert X)\}$$. From the discriminative model, we define joint distributions $$Q(Z,X) = Q_*(X)Q(Z\vert X)$$ which factor through the true distribution $$Q_*(X).$$ Note that the marginal $$Q(X)$$ _equals_ the true distribution $$Q_*(X)$$. Abusing terminology, we will also call $$\{Q(Z,X)\}$$ a _discriminative_ model.

Variational inference is then a joint search over the space of pairs of models. It provides bounds on the distance to $$Q_*(X)$$ from $$P(X),$$ and it limits the search to computationally efficient discriminative models $$\{Q(Z, X)\}$$ and computationally efficient generative models $$\{P(Z,X)\}.$$ Variational inference also supplies a distribution $$Q(Z\vert X)$$ for approximate inference of $$Z$$ from $$X,$$ and a distribution $$P(Z,X)$$ for approximate sampling from $$Q_*(X).$$

In this context, the variational parameter $$Q(Z\vert X)$$ is therefore interpreted as a disciminative computational model trained on the true distribution.

### Context of optimization

A third context we may consider is the goal of optimizing a distance function over some rough low-dimensional landscape. The distance to $$Q_*(X)$$ from $$P(X)$$ is minimized over the space $$\Lambda$$ of pairs of distributions $$(Q_*(X),P(X))$$ where the first component $$Q_*(X)$$ is fixed and the second component $$P(X)$$ is the marginalization of some joint distribution $$P(Z,X).$$ 

Variational inference overcomes the roughness of the low-dimensional landscape by lifting the optimization problem to a higher dimensional space where the landscape is smoother. It considers instead the distance to $$Q(Z,X) = Q_*(X)Q(Z\vert X)$$ from $$P(Z,X)$$ which is minimized over the space $$\Lambda'$$ of pairs of distributions $$(Q(Z,X),P(Z,X)).$$ This space $$\Lambda'$$ is of a higher dimension than the original space $$\Lambda$$, and could be infinite-dimensional if $$Q(Z\vert X)$$ is variational. 

In this context, instead of projecting $$P(Z,X)$$ to the base space of distributions on $$X$$ and optimizing the distance between $$Q_*(X)$$ and $$P(X)$$ in the base space, the disciminative model distribution $$Q(Z\vert X)$$ enables us to lift $$Q_*(X)$$ from the base space to a section $$Q(Z,X)$$ in the bundle of distributions on $$(Z,X)$$ so that we can optimize the distance between $$Q(Z,X)$$ and $$P(Z,X).$$

## Why do we need a better name for variational inference?

In the statistical learning community, variational inference is sometimes called _approximate inference_ because we are inferring the latent $$Z$$ from the observed $$X$$ and approximating the Bayes posterior $$P(Z\vert X)$$ by solving an variational optimization problem over a space of functions $$Q(Z\vert X).$$ For the same reason, it is also called _variational Bayes_.

In recent work, such as with variational autoencoders, the optimization is performed not over an infinite-dimensional variational space of functions but over a finite-dimensional parametric space of functions. It seems strange to continue using the word _variational_ in describing these techniques.

In other work, the goal has shifted away from inferring the Bayes posterior to inferring model parameters that give the best approximation of the true distribution. Here, it seems strange to continue using the word _Bayes_ in describing these approaches.

Since the goal at hand is still inference and since all of these methods hinge on the use of relative information, specifically on measuring the information loss between the discriminative model and the generative model, I propose that we use the name _relative inference_ instead. 

Moreover, instead of calling $$Q(Z\vert X)$$ the _variational_ parameter, I propose calling it and the related joint distribution $$Q(Z,X) = Q(Z\vert X)Q_*(X)$$ the _discriminative_ model, to distinguish it from the generative model $$P(Z,X).$$

## Why should we consider mutable variables rather than latent variables?

In this section, we consider models that contain distributions which are only partially known. For example, in the discriminative model of distributions $$Q(Z,X)$$ described above, the marginals $$Q(X)$$ are all equal to the true distribution $$Q_*(X)$$ which is unknown.

Suppose we have a model and random variables $$Z, X$$ in the model. We say that $$Z$$ is _mutable_ given $$X$$ if the conditional distribution $$Q(Z \vert X)$$ is known in the model and if there exists at least two distributions where $$Q(Z \vert X)$$ is different. This terminology matches with the idea of mutability in data storage systems where a file is mutable if it has at least two different states and it can be changed to any desired state in a reliable way. 

We say that the model is _mutable_ if it contains variables $$Z ,X$$ such that $$Z$$ is mutable given $$X$$. Otherwise, the model is _immutable_. Likewise, a model for a process $$\{Y_t\}$$ is _mutable_ if $$\{Y_t\}$$ contains subprocesses $$\{Z_t\}, \{X_t\}$$ such that $$\{Z_t\}$$ is mutable given $$\{X_t\}.$$ When the model is clear from the context, we abuse notation and say that the process itself is mutable.

The starting point in relative inference is a pair of models - the discriminative model $$Q(Z,X)$$ and the generative model $$P(Z,X)$$ - over variables $$Z, X.$$ If the discriminative model is immutable, then relative inference is reduced to the good old-fashioned maximum likelihood method where the goal is to minimize the relative information between the true distribution $$Q_*(X)$$ and a generative distribution $$P(X).$$ 

If the discriminative model is mutable, then more interesting methods can be derived. One can think of the mutable variables $$Z$$ as stochastic computational units that depend on the given inputs $$X.$$ If we ignore learning/training for now, then specifying a mutable model of stochastic discrminative distributions is analogous to specifying a programmable class of deterministic input-driven algorithms. Training the mutable model for a suitable distribution would then be analogous to searching the programmable class for a suitable algorithm. 

How then do we make sense of the observable variables and latent variables appearing in traditional applications of relative inference? The strategy for inferring possible distributions for the latent variables is to represent them with variables $$Z$$ that are mutable given the observed variables $$X.$$ During training, the conditionals $$Q(Z\vert X)$$ are adjusted until the relative information between the discriminative distribution $$Q(Z,X)$$ and the generative distribution $$P(Z,X)$$ is minimized. Therefore, by _using_ mutable processes, we may _uncover_ latent processes to some extent.

In other applications, such as in modelling spiking neural networks, we represent the universe with a random variable $$X$$ which may not be fully observed by the spiking network. In other words, there may be components of $$X$$ which are hidden or latent. The only assumption we make about the distribution of $$X$$ is that it cannot be changed to a desired distribution in a reliable way. 

Our goal for spiking networks is to learn a generative model $$P(Z,X)$$ with the help of computational units $$Z$$ such that the marginal $$P(X)$$ approximates the true distribution. We are not interested in getting guarantees that the variables $$Z$$ reflect actual latent variables in the universe $$X$$. To accomplish our goal, we will assume that in the discriminative model $$Q(Z,X),$$ the variables $$Z$$ are mutable given $$X,$$ i.e. the conditionals $$Q(Z\vert X)$$ are controllable (changeable in a known way). 

For the rest of this [series](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/), we will talk more generally of learning _by using mutable processes_, as opposed to just learning _to uncover latent processes_.



## References

<a id="ref-A95"></a>[[A95]](#ref-A95) Amari, Shun-ichi. "Information geometry of the EM and em algorithms for neural networks." _Neural networks_ 8, no. 9 (1995): 1379-1408.

<a id="ref-BKM17"></a>[[BKM17]](#ref-BKM17) Blei, David M., Alp Kucukelbir, and Jon D. McAuliffe. "Variational inference: A review for statisticians." _Journal of the American statistical Association_ 112, no. 518 (2017): 859-877.

<a id="ref-DLR77"></a>[[DLR77]](#ref-DLR77) Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. "Maximum likelihood from incomplete data via the EM algorithm." _Journal of the Royal Statistical Society: Series B (Methodological)_ 39, no. 1 (1977): 1-22.

<a id="ref-JGJS99"></a>[[JGJS99]](#ref-JGJS99) Jordan, Michael I., Zoubin Ghahramani, Tommi S. Jaakkola, and Lawrence K. Saul. "An introduction to variational methods for graphical models." _Machine learning_ 37, no. 2 (1999): 183-233.

<a id="ref-W09"></a>[[W09]](#ref-W09) Watanabe, Sumio. _Algebraic geometry and statistical learning theory_. Vol. 25\. Cambridge university press, 2009.



