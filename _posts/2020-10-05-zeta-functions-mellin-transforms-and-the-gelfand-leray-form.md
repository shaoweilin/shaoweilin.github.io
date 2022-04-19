---
layout: post
title: Zeta functions, Mellin transforms and the Gelfand-Leray form
---

We outline the similarities between zeta functions appearing in number theory and in statistical learning.

This post is a continuation from our [series](https://shaoweilin.github.io/motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

## Gelfand-Leray functions

Let $$\omega$$ be a volume form or a form resulting from a measure on some space $$W$$. Let $$f: W \rightarrow \mathbb{R}$$ be a non-negative energy function or a Kullback-Leibler divergence. A _Gelfand-Leray form_ of $$\omega$$ is any differential form $$\phi$$ that satisfies

$$df \wedge \phi = \omega.$$

We will denote $$\phi$$ by $$\omega /df$$. See [[AGV88]](#ref-AGV88) and §4 of [[M10]](#ref-M10) for expositions about the Gelfand-Leray form.

The _Gelfand-Leray function_ is

$$J(t) =\displaystyle \int_{f(w)=t} \frac{\omega}{df}.$$

These functions often have asymptotic expansions of the form

$$J(t) = \displaystyle \sum_\alpha \sum_{i=1}^d m_{\alpha,i} t^\alpha (\log t)^{i-1}.$$

## Zeta functions

Informally, zeta functions are complex functions of the form

$$\zeta(z) = \displaystyle \sum_\lambda m_\lambda \lambda^{-z}$$

where the sum is over some possibly infinite (multi-)set of complex numbers $$\lambda$$ and the coefficients $$m_\lambda$$ are complex.

For example, the Riemann zeta function is

$$\zeta(z) = \displaystyle \sum_{n=1}^\infty n^{-z}.$$

Zeta functions often have analytic continuations to the whole complex plane, and possess Laurent expansions of the form

$$\zeta(z) = \zeta_0(z) + \sum_\alpha \sum_{i=1}^d c_{\alpha i} (z-\alpha)^{-i}$$

where $$\zeta_0(z)$$ is holomorphic and the $$\alpha$$ are the poles.

A zeta function can often be written as the Mellin transform

$$\zeta(z) = \displaystyle \int_0^\infty t^{z-1} J(t) dt = \int_W f(w)^{z-1} \omega$$

of a function $$J(t)$$, and $$J(t)$$ is often some kind of Gelfand-Leray function.

For example, $$\zeta(z) = \sum_\lambda m_\lambda \lambda^{-z}$$ is the Mellin transform

$$\zeta(z) = \displaystyle \frac{1}{\Gamma(s)} \int_0^\infty \theta(t) t^{z-1} dt$$

of the theta function

$$\theta(t) = \displaystyle \sum_{\lambda} m_\lambda e^{- \lambda t}.$$

However, I do not know if the theta function can be expressed as the Gelfand-Leray function of a differential form with respect to an energy function.

Zeta functions are also useful for defining infinite products [[M95]](#ref-M95)

$$Z = \displaystyle \prod_\lambda \lambda^{m_\lambda}$$

which we define to be

$$Z = \exp \left( - \zeta'(0)\right)$$

where

$$\zeta(z) = \displaystyle \sum_\lambda m_\lambda \lambda^{-z}.$$

Indeed, for finite sums, we see that

$$\zeta'(0) = - \displaystyle \sum_\lambda m_\lambda \log \lambda.$$

## Statistical Learning

In statistical learning, given a prior $$\omega = \psi(w)dw$$ on the parameters $$w$$ of a statistical model $$p: W \rightarrow \Delta,$$ and given the relative information $$K(w)$$ to the true distribution $$q$$ from the model distribution $$p(w),$$ the Gelfand-Leray function

$$J(t) = \displaystyle \frac{d}{dt} \int_{0 \leq K(w) \leq t} \psi(w) dw$$

is called the state density function [[W09]](#ref-W09).

The zeta function of the statistical model is

$$\zeta(z) = \int_W K(w)^z \psi(w) dw = \int_0^\infty t^z J(t)dt$$

which (up to some linear change of variable in $$z$$) is the Mellin transform of the state density function $$J(t).$$

## Number Theory

In number theory, we have the notorious Riemann zeta function which becomes more interesting [[B05]](#ref-B05) if we write it as

$$\zeta(z) = \displaystyle \sum_{n=1}^\infty \left(n^2\right)^{-\frac{z}{2}}.$$

This representation suggests that we study the infamous Jacobi theta function

$$\theta(t) =\displaystyle \sum_{n=-\infty}^\infty e^{-n^2 (\pi t) }.$$

Note the extra factor of $$\pi$$ in the exponent and the changing of the lower bound of summation to $$-\infty$$.

The modularity of the theta function

$$\theta(t) = \displaystyle \frac{1}{\sqrt{t}} \theta\left(\frac{1}{t}\right)$$

is the crux behind the functional equation for the Riemann zeta function

$$\Lambda(z) = \Lambda(1-z)$$

where $$\Lambda(z) = 2^{-1/2}\pi^{-z/2}\Gamma(z/2)\zeta(z).$$

The function $$\Lambda(z)$$ itself can be written [[D91]](#ref-D91) as the infinite product

$$(\frac{z}{2\pi})(\frac{z-1}{2\pi})\Lambda(z) = \prod_\rho \frac{z-\rho}{2\pi}$$

over the zeros of the Riemann zeta function. This infinite product is in turn expressible using another zeta function and theta function.

The Hurwitz zeta function

$$\zeta_s(z) = \displaystyle \sum_{n=0}^\infty (n+s)^{-z}$$

is associated to the theta function $$\theta_s(t) = e^{-st} \theta(t)$$ where

$$\theta(t) =\sum_{n=0}^\infty e^{-nt} = \frac{1}{1-e^{-t}}.$$

Note that the Riemann zeta function is the special case $$\zeta_1(z).$$

The Hasse-Weil zeta function of a smooth absolutely irreducible curve $$V$$ over a finite field $$\mathbb{F}_q$$ is

$$\begin{array}{rl} Z_{V}(z) &= \displaystyle \sum_\alpha \left(q^{ \deg(\alpha)}\right)^{-z} \\ & \\ &= \exp\left(\displaystyle\sum_{m\geq 1} \frac{\# V(\mathbb{F}_{q^m})}{m}q^{-zm} \right) \end{array}$$

where the first sum is over effective zero-cycles $$\alpha$$. The zeta function can also be written as

$$(1-q^{-z})(1-q^{1-z})Z_{V}(z) = \displaystyle \prod_{j=1}^{2g} (1-\phi_j q^{-z})$$

where $$g$$ is the genus of $$V$$ and each $$\phi_j$$ is an algebraic integer. Each of the factors can be written as infinite products

$$1-\lambda q^{-z} = \displaystyle \prod_{\rho \vert  q^\rho = \lambda} \frac{\log q}{2\pi i}(z-\rho).$$

which is in turn expressible using another zeta function and theta function.

## References

<a id="ref-AGV88"></a>[[AGV88]](#ref-AGV88) Arnold, V. I., S. M. Gusein-Zade, and A. N. Varchenko. "Elementary integrals and the resolution of singularities of the phase." In _Singularities of Differentiable Maps_, pp. 215-232\. Birkhäuser Boston, 1988.

<a id="ref-B05"></a>[[B05]](#ref-B05) Baez, John. _This Week's Finds in Mathematical Physics_ [week 217](http://math.ucr.edu/home/baez/week217.html), 2005.

<a id="ref-D91"></a>[[D91]](#ref-D91) Deninger, Christopher. "On the Γ-factors attached to motives." _Inventiones mathematicae_ 104, no. 1 (1991): 245-261.

<a id="ref-M95"></a>[[M95]](#ref-M95) Manin, Yuri. "Lectures on zeta functions and motives." _Astérisque_ 228 (1995): 121-163.

<a id="ref-M10"></a>[[M10]](#ref-M10) Marcolli, Matilde. _Feynman motives_. World Scientific, 2010.

<a id="ref-W09"></a>[[W09]](#ref-W09) Watanabe, Sumio. _Algebraic geometry and statistical learning theory_. Vol. 25\. Cambridge university press, 2009.