---
layout: post
title: Path integrals and the Dyson formula
excerpt_separator: <!--more-->
---

One of the deepest results in quantum field theory, to me, is the Dyson formula [[N21]](#ref-N21). It describes the solution to the differential equation

$$ i\frac{\partial}{\partial t} \Psi(t) = A(t)\Psi(t) $$

in terms of the exponential of the path integral of the operator $$A(t)$$,

$$\begin{array}{rl} \Psi(t) & = U(t,0) \, \Psi(0) \\ & \\
U(t,0) & = \displaystyle \mathcal{T}\exp \left\{ i\int_0^t A(s) ds \right\} \end{array} $$

where $$\mathcal{T}$$ is the time-ordering operator. Here, $$U(t,0)$$ is known as the _time-evolution operator_. 
<!--more-->

The proof of this formula is given by Picard integration and iterated integrals [[B13]](#ref-B13).

On the other hand, we have the Feynman path integral

$$ \begin{array}{rl}
\psi(y,t) & = K(y,t; x,0)  \, \psi(x,0) \\ & \\ 
K(y,t; x,0) & = \displaystyle \int \exp\left\{iS(q,\dot{q})\right\} Dq \\ & \\ &  = \langle y \vert \, U(t,0)  \, \vert x\rangle\end{array}
$$

where $$\psi(y,t)$$ is the value of the state vector $$\Psi(t)$$ at position $$y,$$ $$\int Dq$$ is an integral over all paths $$q$$ from $$x$$ to $$y$$, and $$S(q,\dot{q})$$ is the action of the path $$q.$$ Here, $$K(y,t;x,0)$$ is known as the _kernel_ or _propagator_.

The time-ordered exponential in the Dyson series can be related to the Feynman path integral by viewing the Hamiltonian operators in a von Neumann infinite tensor product Hilbert space [[G17]](#ref-G17).

The Dyson formula frames the dynamics of the system in terms of the time-evolution operator, while the path integral frames the dynamics in terms of the kernel.

The action gives a "particle" interpretation to the quantum or stochastic dynamical system. These "particles" are real only to the extent that this action is well-defined. 

It would be interesting to study the Dyson formula through the algebraic lens of regularity structures.

There should also be a generalization of the Dyson formula or the path integral to coends in category theory.

[Freeman Dyson - Linking the ideas of Feynman, Schwinger and Tomanaga]<https://www.youtube.com/watch?v=i3RcN5UGwgI>

## References

<a id="ref-B13"></a>[[B13]](#ref-B13) Brown, Francis. "[Iterated integrals in quantum ﬁeld theory](http://math.bu.edu/people/dkreimer/houches/ColumbiaNotes7.pdf)." _Geometric and Topological Methods for Quantum Field Theory: Proceedings of the 2009 Villa de Leyva Summer School_ (2013): 188.

<a id="ref-G17"></a>[[G17]](#ref-G17) Gill, Tepper L. "The feynman-dyson view." In _Journal of Physics: Conference Series_, vol. 845, no. 1, p. 012023. IOP Publishing, 2017.

<a id="ref-N21"></a>[[N21]](#ref-N21) https://ncatlab.org/nlab/show/Dyson+formula

