---
date: 2020-12-01
excerpts: 2
---

# Biased stochastic approximation

We explore the convergence of continuous-time ordinary differential equations and their discrete-time analogs, such as stochastic approximation and gradient descent, through the lens of Lyapunov theory {cite}`bottou1998online` {cite}`li2015stochastic`. From this perspective, we will study biased stochastic approximation {cite}`karimi2019non` where the expectation of the stochastic updates conditioned on the past (which we call the _conditional expectation_) is not the same as the expectation of the stochastic updates under the stationary distribution (which we call the _total expectation_).

This post is a continuation from our [series](2020-08-28-motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

## Continuous-time dynamics

Suppose we have a continuous-time dynamical system with state space $\mathcal{H} \subset \mathbb{R}^d$ and update rule

$$\dot{\eta} = -h(\eta),$$

and suppose we have a function $V: \mathcal{H} \rightarrow \mathbb{R}$ that has a finite lower bound. We are primarily interested in the conditions that will lead to convergence of the system to two kinds of points, namely

1.  To a fixed point, i.e. $h(\eta) = 0,$
2.  To a critical point, i.e. $\nabla V(\eta) = 0.$

In the special case where $h(\eta) = \nabla V(\eta),$ the two kinds of points are the same.

If the value of $V(\eta(t))$ along any path $\eta : \mathbb{R} \rightarrow \mathcal{H}$ in the dynamical system decreases strictly with time $t$, then we say that $V$ is a Lyapunov function for the dynamical system. Here, the system converges to a point that is both a fixed point of the system and a critical point of $V$.

Suppose $V$ is $C^1$-smooth and near each point $\eta$ it has the expansion

$$V(\eta') = V(\eta) + \langle \nabla V(\eta), \eta'-\eta \rangle + O(\Vert \eta'-\eta \Vert^2).$$

In continuous-time, the updates $\eta'-\eta$ are infinitesimal, so the change in $V$ is dominated by the first order term. Therefore, if $0 < \langle \nabla V(\eta), h(\eta) \rangle$ for all $\eta : \mathcal{H}$ away from the zeros of $h,$ then $V$ is a Lyapunov function for the dynamical system.

In the special case where $h(\eta) = \nabla V(\eta),$ we have

$$\Vert h(\eta) \Vert^2 = \langle \nabla V(\eta), h(\eta)\rangle$$

which will be strictly positive away from the zeros of $h.$

## Discrete-time dynamics

For discrete-time dynamical systems such as gradient descent or stochastic approximation, the situation is more delicate. The value of the function $V: \mathcal{H} \rightarrow \mathbb{R}$ may fluctuate up and down with each discrete time step, but the _total expectation_ of $\Vert h(\eta) \Vert$ or of $\nabla V(\eta)$ decreases over time so the system converges to a fixed point or to a critical point.

Specifically, suppose that we have discrete-time stochastic process

$$\eta_{n+1} = \eta_{n} - \gamma_{n+1} H_{\eta_n}$$

where $\gamma_{n+1}$ is the time-dependent learning rate and $H_{\eta_n}$ is the update at time $(n+1)$ that depends on the previous system state $\eta_n.$ The update could be deterministic as in the case of gradient descent, or more generally it could be stochastic as in the case of stochastic approximation.

We will assume that any stochastic update $H_{\eta_n}(X_{n+1})$ is a function of some parameter-controlled stochastic process $\{X_n\}$, i.e. the distribution of $X_{n+1}$ conditioned on its past $X_n, X_{n-1}, \ldots$ is controlled by the parameter $\eta_n.$ We will also assume that for each $\eta \in \mathcal{H}$, the $\eta$-controlled stochastic process $\{X_n\}$ has a unique stationary distribution $\pi_\eta$. Let $h(\eta)$ be the total expectation or _mean field_ of the $\eta$-controlled stochastic process, i.e. the expectation of the stochastic updates $H_{\eta}(x)$ with respect to $x \sim \pi_\eta.$

As before, let $V: \mathcal{H} \rightarrow \mathbb{R}$ be a function with a finite lower bound. We will again be interested in conditions that guarantee the convergence of the discrete-time dynamical system to two kinds of points, namely

1.  To a fixed point, i.e. $h(\eta) = 0,$
2.  To a critical point, i.e. $\nabla V(\eta) = 0.$

The general strategy for proving the convergence of the system is to show that the _total expectation_ of $V(\eta_n)$ is _eventually_ a decreasing function. To show that this decrease happens, it is often sufficient to check that for all $\eta', \eta,$

$$\displaystyle V(\eta') \leq V(\eta)+ \langle \nabla V(\eta), \eta'-\eta \rangle + \frac{\ell}{2}\Vert \eta'-\eta \Vert^2$$

for some constant $\ell>0.$ When the domain $\mathcal{H}$ is convex, this condition is equivalent to the $\ell$-smoothness of $V,$ i.e. for all $\eta', \eta,$

$$\Vert \nabla V(\eta')- \nabla V(\eta) \Vert \leq \ell \Vert \eta' - \eta \Vert.$$

which is to say that $\nabla V$ is Lipschitz continuous.

Going further, we may want to prove some results about the speed of convergence. First, let us represent the update

$$H_{\eta_n}(X_{n+1}) = h(\eta_n) +E_{\eta_n}(X_{n+1})$$

as the sum of the mean field and a correction term. Substituting this representation into the $\ell$-smoothness condition, we get

$$\displaystyle \begin{array}{rl}V(\eta_{n+1}) & \leq V(\eta)- \gamma_{n+1} \langle \nabla V(\eta),H_{\eta_n}(X_{n+1}) \rangle \\ & \\ & \quad \quad + \frac{\ell\gamma_{n+1}^2}{2}\Vert H_{\eta_n}(X_{n+1}) \Vert^2 \\ & \\ & \leq V(\eta_n) - \gamma_{n+1}\langle \nabla V(\eta), h(\eta_n) \rangle \\ & \\ & \quad \quad - \gamma_{n+1}\langle \nabla V(\eta), E_{\eta_n}(X_{n+1}) \rangle \\ & \\ & \quad \quad + \frac{\ell \gamma_{n+1}^2}{2}\Vert E_{\eta_n}(X_{n+1})\Vert^2 \\ & \\ & \quad \quad + \frac{\ell \gamma_{n+1}^2}{2}\Vert h(\eta_n)\Vert^2 \end{array}.$$

To manage the speed of convergence, one effective way is to bound the terms involving $\langle \nabla V(\eta), h(\eta_n) \rangle,$ $\langle \nabla V(\eta), E_{\eta_n}(X_{n+1}) \rangle$ and $\Vert E_{\eta_n}(X_{n+1})\Vert^2$ with some scalar multiple of $\Vert h(\eta_n) \Vert^2.$

Starting with $\langle \nabla V(\eta), h(\eta_n) \rangle,$ we could require that for all $\eta,$

$$\Vert h(\eta) \Vert^2 \leq C \langle \nabla V(\eta), h(\eta)\rangle.$$

This condition is automatically satisfied for gradient dynamical systems where $h(\eta) = \nabla V(\eta)$, so the inequality becomes an equality with $C=1.$

As for $\Vert E_{\eta_n}(X_{n+1})\Vert^2,$ we could require that this correction term be uniformly bounded, or that its conditional expectation (i.e. expected value when conditioned on the past $X_n, X_{n-1}, \ldots$) be bounded by a scalar multiple of $\Vert h(\eta_n) \Vert^2.$

Lastly, for $\langle \nabla V(\eta), E_{\eta_n}(X_{n+1}) \rangle,$ recall that

$$H_{\eta_n}(X_{n+1}) = h(\eta_n) +E_{\eta_n}(X_{n+1})$$

so the total expectation of the correction $E_{\eta_n}(X_{n+1})$ is zero by definition. If we further require that the _conditional expectation_ of $H_{\eta_n}(X_{n+1})$be equal to the mean field $h(\eta_n),$ then the _conditional expectation_ of $E_{\eta_n}(X_{n+1})$ will be equal to zero. Here, the expectation of $\langle \nabla V(\eta), E_{\eta_n}(X_{n+1}) \rangle$ vanishes when we condition on the past and we get a good handle on the speed of convergence. We refer to this scenario as _unbiased stochastic approximation_.

However, the conditional expectation of $E_{\eta_n}(X_{n+1})$ is often non-zero for many important applications. We refer to this scenario as _biased stochastic approximation_. Here, the object of interest is the discrete-time stochastic integral

$$\displaystyle \sum_{k=0}^n E_{\eta_k}(X_{k+1}).$$

If the conditional expectation of $E_{\eta_n}(X_{n+1})$ is zero, then the above stochastic integral is a martingale. To tackle the scenario where the conditional expectation of $E_{\eta_n}(X_{n+1})$ is non-zero, we will need to find another suitable martingale by solving a Poisson equation, so as to control the behavior of the stochastic integral. We introduce this strategy in the next section.

## Martingales, stochastic integrals and the Poisson equation

Given a (discrete-time or continuous-time) stochastic process $\{H_t\}_{0 \leq t},$ let $\mathcal{F}_s$ denote the filtration (i.e. sequence of sigma algebras with $\mathcal{F}_s \subseteq \mathcal{F}_t$ for all $s \leq t$) generated by the random variations $\{H_t\}_{0\leq t \leq s}.$ Recall that $H_t$ is a martingale if for all $s \leq t,$

$$\mathbb{E}[ H_t \vert  \mathcal{F}_s] = H_s.$$

Note that this condition implies that for discrete time stochastic processes, the expectation of the _martingale difference_ $E_{n+1} = H_{n+1} - H_n$ conditioned on $\mathcal{F}_n$ is zero.

Martingales play an important role in stochastic integration {cite}`cattiaux2012central`. For example, let $\{X_t\}_{0\leq t}$ be a continuous-time stochastic process driven by Brownian motion, e.g.

$$dX_t = b(X_t) dt + \sigma(X_t) dB_t$$

where $X_t$ is an $n$-dimensional vector, $B_t$ is $m$-dimensional Brownian motion, $b(X_t)$ is an $n$-dimensional vector and $\sigma(X_t)$ is an $n \times m$ matrix. Let $L$ be the infinitesimal generator of this process, i.e. $L$ acts on the space of measurable functions such that for any measurable function $g$ and state $x,$

$$\begin{array}{rl} \displaystyle Lg(x) &:= \displaystyle \lim_{t \rightarrow 0} \frac{ \mathbb{E}[ g(X_t) \vert X_0 =x] - g(x)} {t} \\ & \\ & = \displaystyle \lim_{t \rightarrow 0} \frac{ P^t g(x) - g(x)} {t} \\ & \\ & = \displaystyle \left. \partial_t P^t g(x) \right\vert_{t=0} \end{array}$$

where $P^t$ is the transition operator of the stochastic process $\{X_t\}.$

Then, one can show that

$$\displaystyle \int_0^t Lg(X_s) ds = g(X_t) - g(X_0) - \int_0^t \sum_{i} \frac{\partial g}{\partial x_i}(X_s) \sigma_i (X_s) dB_s,$$

thanks to the Itô formula, e.g. see Lemma 7.3.2 of {cite}`oksendal2013stochastic`.

The last term

$$\displaystyle M_t = \int_0^t \sum_{i} \frac{\partial g}{\partial x_i}(X_s) \sigma_i (X_s) dB_s$$

is a martingale, so its expected value conditioned on $X_0$ is the initial value $M_0$ which is zero. Therefore,

$$\displaystyle \mathbb{E} \left[\int_0^t Lg(X_s) ds \middle\vert  X_0\right] = \mathbb{E} [g(X_t) \vert X_0] - g(X_0)$$

after taking expectations of the Itô formula.

Interestingly, this formula gives us a strategy for computing expectations of stochastic integrals of the form

$$\displaystyle \mathbb{E} \left[\int_0^t f(X_s) ds \middle\vert  X_0 \right].$$

Indeed, if we are able to solve the _Poisson equation_

$$Lg = f$$

with some solution $g$ that is sufficiently regular, then we can use the above formula to compute the desired answer.

Under some regularity conditions on $\{X_t\}$ and assuming that the total expectation of $g(x)$ is zero, a candidate solution to the Poisson equation is

$$\displaystyle g(x) = - \int_0^\infty P^s f(x) ds$$

if the integral is well-defined {cite}`cattiaux2012central`. Indeed, if $Lg = f,$ then

$$\begin{array}{rl} P^t g(x) - g(x) &= \displaystyle \int_0^t \partial_s P^s g(x) ds \\ & \\ & = \displaystyle \int_0^t L P^s g (x) ds \\ & \\ & = \displaystyle \int_0^t P^s Lg(x) ds = \displaystyle \int_0^t P^s f(x) ds. \end{array}$$

Taking limits as $t \rightarrow \infty$ and observing that under strong ergodicity $P^t g(x)$ goes to the total expectation of $g(x)$ which is zero, the candidate solution follows.

The strategy for integrals of functions of discrete-time stochastic processes works in a similar way, where solutions to the Poisson equation provide martingales whose expected values vanish. We will use this strategy for analyzing biased stochastic approximation algorithms.

In a future post, we will explore martingales, stochastic integrals and the Poisson equation through the lens of regularity structures {cite}`hairer2014theory`.

## Biased updates

For the rest of this post, we assume that our discrete-time parameter-controlled stochastic process $\{X_n\}_{0 \leq n}$ is Markov. Explicitly, let $(\mathcal{X}, \Sigma)$ be a measurable space. A function $P:\mathcal{X} \times \Sigma \rightarrow [0,1]$ is a Markov kernel if $P(x, \cdot): \Sigma \rightarrow [0,1]$ is a distribution for all $x \in \mathcal{X}$, and $P(\cdot, A): \mathcal{X} \rightarrow [0,1]$ is measurable for all $A \in \Sigma.$ A Markov kernel generalizes the notion of a transition matrix beyond finite-state Markov chains.

Suppose we have a Markov kernel $P_\eta$ for each $\eta \in \mathcal{H}$. Let $L_\eta$ be the generator of the process with Markov kernel $P_\eta$, i.e. $L_\eta$ acts on the space of measurable functions such that for any function $g$ and state $x,$

$$\displaystyle L_\eta g(x) := P_\eta g(x)- g(x).$$

We also assume that for all $\eta \in \mathcal{H}$, the Markov kernel $P_\eta$ has a unique stationary distribution $\pi_\eta$, i.e. for all $A \in \Sigma,$

$$\displaystyle \pi_\eta(A) = \pi_\eta P_\eta(A) := \int_\mathcal{X} \pi_\eta(dx) P_\eta(x, A).$$

Let $X_0, X_1, \ldots$ be random variables on this space such that for all bounded measurable functions $\varphi$ and integer $n \geq 0$, we have

$$\displaystyle \mathbb{E} [ \varphi(X_{n+1}) \vert  \mathcal{F}_n] = P_{\eta_n} \varphi(X_n) := \int_\mathcal{X} P_{\eta_n} (X_n, dx) \varphi(x)$$

where the parameter updates are given by

$$\eta_{n+1} = \eta_{n} - \gamma_{n+1} H_{\eta_n}(X_{n+1})$$

$$H_{\eta_n}(X_{n+1}) = h(\eta_n) +E_{\eta_n}(X_{n+1})$$

$$h(\eta) = \displaystyle \int \pi_\eta(dx) H_{\eta}(x)$$

as before. We will primarily be interested in the stochastic integral

$$\displaystyle \sum_{k=0}^n E_{\eta_k}(X_{k+1}).$$

For simplicity, we first fix the parameters $\eta_n = \eta$ for all $n$ and drop the subscripts $\eta$ in notations such as $E_\eta,$ $P_\eta$ and $L_\eta$. Later, we will generalize the approach to the case with parameter updates {cite}`karimi2019non`.

Suppose that we have a solution $\hat{H}$ to the Poisson equation $L \hat{H} = E.$ Then

$$\begin{array}{rl} & \displaystyle \sum_{k=0}^n E(X_{k+1}) \\ & \\ &= \displaystyle \sum_{k=0}^n P\hat{H}(X_{k+1}) - \hat{H}(X_{k+1}) \\ & \\ & = \displaystyle P\hat{H}(X_{n+1}) - \hat{H}(X_{1}) + \sum_{k=1}^n P\hat{H}(X_{k}) - \hat{H}(X_{k+1}) \end{array}.$$

The last sum is a martingale, because each of the summands $P\hat{H}(X_{k}) - \hat{H}(X_{k+1})$ is a martingale difference, i.e. conditioned on $X_s$ for some $s \leq k,$

$$\begin{array}{rl} &\mathbb{E} [P\hat{H}(X_{k}) - \hat{H}(X_{k+1}) \vert X_s] \\ & \\ & = \displaystyle \int P^{k+1-s}(X_s, dx) \hat{H}(x) - \int P^{k+1-s}(X_s, dx) \hat{H}(x) = 0.\end{array}$$

This martingale term vanishes under both conditional and total expectation.

Now, we tackle the general case where the parameter $\eta_n$ is updated. We are primarily interested in the stochastic integral

$$\displaystyle \sum_{k=0}^n \gamma_{k+1} \langle \nabla V(\eta_k) \vert E_{\eta_k}(X_{k+1}) \rangle.$$

Suppose that for all $\eta$, we have a solution $\hat{H}_\eta$ to the Poisson equation $L_\eta \hat{H}_\eta = E_\eta.$ We may then decompose the above stochastic integral as the sum of the following expressions.

$$S_0 := \gamma_{n+1} \langle \nabla V(\eta_n) \vert P_{\eta_n} \hat{H}_{\eta_n}(X_{n+1}) \rangle - \gamma_{1} \langle \nabla V(\eta_0) \vert \hat{H}_{\eta_0}(X_{1}) \rangle $$

$$S_1 := \sum_{k=1}^n \gamma_{k+1} \langle \nabla V(\eta_k) \vert P_{\eta_k} \hat{H}_{\eta_k}(X_{k}) - \hat{H}_{\eta_k}(X_{k+1}) \rangle$$

$$S_2 := \sum_{k=1}^n \gamma_{k+1} \langle \nabla V(\eta_k) \vert P_{\eta_{k-1}} \hat{H}_{\eta_{k-1}}(X_{k}) - P_{\eta_k} \hat{H}_{\eta_k}(X_{k}) \rangle$$

$$S_3 := \sum_{k=1}^n \gamma_{k+1} \langle \nabla V(\eta_{k-1}) - \nabla V(\eta_k) \vert P_{\eta_{k-1}} \hat{H}_{\eta_{k-1}}(X_{k}) \rangle$$

$$S_4 := \sum_{k=1}^n (\gamma_{k} - \gamma_{k+1}) \langle \nabla V(\eta_{k-1}) \vert P_{\eta_{k-1}} \hat{H}_{\eta_{k-1}}(X_{k}) \rangle$$

Here, the expressions $S_0$ and $S_1$ arise naturally from the application of the Poisson equation solutions. In particular, the terms $P_{\eta_k} \hat{H}_{\eta_k}(X_{k}) - \hat{H}_{\eta_k}(X_{k+1})$ appearing in $S_1$ are martingale differences so $S_1$ vanishes under conditional and total expectations.

The expressions $S_2,$ $S_3$ and $S_4$ are correction terms coming from updates to the parameters and step sizes, and they can be bounded by some suitable assumptions on the regularity of $P_\eta \hat{H}_\eta(x),$ $\nabla V (\eta)$ and $\gamma_{n}$ respectively.

To summarize, we have the following convergence result for biased stochastic approximation {cite}`karimi2019non`, starting with some regularity conditions. Note that $h(\eta)$ varies like $\frac{\partial V}{\partial \eta}(\eta)$ so we are guaranteed convergence only to a critical point of the Lyapunov function $V(\eta).$


----

**A1 (Direction of mean field).** There exists $c_0 \geq 0, c_1 \geq 0$ such that for all $\eta \in \mathcal{H},$

$$\displaystyle c_0 + c_1 \left\langle \frac{\partial V}{\partial \eta}(\eta) , h(\eta) \right\rangle \geq \Vert h(\eta) \Vert^2.$$

**A2 (Length of mean field).** There exists $d_0 \geq 0, d_1 \geq 0$ such that for all $\eta \in \mathcal{H},$

$$\displaystyle d_0 + d_1 \Vert h(\eta) \Vert \geq \left\Vert \frac{\partial V}{\partial \eta}(\eta) \right\Vert.$$

**A3 ($\ell$-smoothness of Lyapunov function).** There exists $\ell < \infty$ such that for all $\eta, \eta' \in \mathcal{H},$

$$\displaystyle \left\Vert \frac{\partial V}{\partial \eta}(\eta) - \frac{\partial V}{\partial \eta}(\eta') \right\Vert \leq \ell \Vert \eta - \eta' \Vert.$$

**A4 (Solution of Poisson equation).** There exists a Borel measurable function $\hat{H} : \mathcal{H} \times \mathcal{X} \rightarrow \mathcal{H}$ such that for all $\eta \in \mathcal{H}, x \in \mathcal{X}$

$$L_{\eta} \hat{H}_{\eta}(x) = E_{\eta}(x).$$

**A5 (Regularity of solution).** There exists $\ell_0, \ell_1 < \infty$ such that for all $\eta, \eta' \in \mathcal{H}, x \in \mathcal{X},$

$$\Vert \hat{H}_{\eta} (x) \Vert \leq \ell_0, \quad \Vert P_{\eta} \hat{H}_{\eta}(x) \Vert \leq \ell_0,$$

$$\Vert P_{\eta} \hat{H}_{\eta}(x) - P_{\eta'} \hat{H}_{\eta'} (x) \Vert \leq \ell_1 \Vert \eta - \eta' \Vert.$$

**A6 (Correction bound).** There exists $\sigma < \infty$ such that for all $\eta \in \mathcal{H}, x \in \mathcal{X},$

$$\Vert E_{\eta} (x) \Vert \leq \sigma.$$

----

**<a id="theorem-convergence-of-biased-stochastic-approximation"></a>Theorem (Convergence of Biased Stochastic Approximation).** Suppose that we have parameter updates

$$\eta_{k+1} = \eta_{k} - \gamma_{k+1} H_{\eta_k}(X_{k+1})$$

for $0 \leq k \leq n,$ using step sizes $\gamma_k = \gamma_0 k^{-1/2}$ for sufficiently small $\gamma_0 \geq 0,$ and using a random stop time $0 \leq N \leq n$ with $\mathbb{P}(N = l) := (\sum_{k=0}^n \gamma_{k+1})^{-1} \gamma_{l+1}.$ Then assuming A1-A6, we have

$$\mathbb{E}(\Vert h(\eta_N) \Vert^2) = O(c_0 + \log n / \sqrt{n} ).$$

----

## References

```{bibliography}
:filter: docname in docnames
```