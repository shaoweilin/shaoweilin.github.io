---
layout: post
title: Path integals and the Dyson formula
---

One of the deepest results in quantum field theory, to me, is the Dyson formula. It describes the solution to the differential equation

$$ \frac{d}{dt} X(t) = A(t)X(t) $$

in terms of the exponential of the path integral of the operator $$A(t)$$,

$$ X(t) = \mathcal{T}\exp \left(\int_0^t A(s) ds \right) X(0). $$

where $$\mathcal{T}$$ is the time-ordering operator. The proof of this formula is given by Picard integration and iterated integrals [B13].

In cases where the path integral of the operator is not convergent or well-defined, I think that this formula can be quoted formally as the definition of the exponentiation of the path integral. 

The path integral gives a "particle" interpretation to the quantum or stochastic dynamical system. These "particles" are real only to the extent that this path integral is well-defined. 

It would be interesting to study the Dyson formula through the algebraic lens of regularity structures.

There should also be a generalization of the Dyson formula to co-ends in category theory.


### References

[B13] Brown, Francis. "[Iterated integrals in quantum ﬁeld theory](http://math.bu.edu/people/dkreimer/houches/ColumbiaNotes7.pdf)." _Geometric and Topological Methods for Quantum Field Theory: Proceedings of the 2009 Villa de Leyva Summer School_ (2013): 188.

[N21] https://ncatlab.org/nlab/show/Dyson+formula

