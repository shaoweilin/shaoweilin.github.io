---
layout: post
title: Information type theory
excerpt_separator: <!--more-->
---






### Posetal subcategories

Given a category $$B$$ of spaces with bundles as morphisms, we require that all diagrams commute so the category is in fact a poset, i.e. a catagory where each $$\text{Hom}$$ set has at most one morphism. We call $$B$$ a _bundle poset_. We assume that $$B$$ has a terminal object $$*$$.

We also consider a family $$q$$ of spaces and sections where all diagrams commute. Therefore, $$q$$ is also a poset that is a right inverse to $$B$$ in a suitable way. We call $$q$$ a _section poset_ to $$B$$. Since $$B$$ has a terminal object $$*$$, it follows that $$q$$ has an initial object $$*$$.

If $$p$$ is another section poset to $$B$$, we may then consider the relative information between $$q$$ and $$p$$ for a given morphism $$f: X\rightarrow Y$$ in $$B$$ weighted by the point $$q_Y : * \rightarrow Y$$. 

Incidence algebras, group algebras, category algebras: functions from a poset to an additive monoid. Mobius inversion formulas and other similar theorems.

Net bundles over a poset. https://doi.org/10.1016/j.aim.2008.08.004



### Natural models of type theory

Going beyond bundles, we can consider more generally presheaves, but we want to put additional structure on these presheaves so that our characterization of relative information makes sense.

Specifically, suppose we have a category of presheaves, and we want relative information to be a functor from this category to the additive monoid $$[0,\infty]$$. We want to add products and sums to this category.

Steve Awodey [[A18]](#ref-A18) has a nice characterization of when natural models admit type formers $$\Sigma$$ and $$\Pi$$.

### Information geometry

Given a manifold, we can define a metric on that manifold using the Fisher information. This creates a Riemannian manifold, much like what happens in general relativity.

Given a topos which is highly geometric in nature, we can define a metric on the topos using relative information. We hope that this creates a metric space on which we can study gravity

http://nlab-pages.s3.us-east-2.amazonaws.com/nlab/show/geometry+of+physics+--+categories+and+toposes

### Quantum information

Partial boolean algebras (partial boolean operator is 
tensor product)
https://arxiv.org/pdf/1306.3951.pdf

Quantum information
https://arxiv.org/pdf/2011.03064.pdf



## Weighted or normed categories














<!--more-->
## Motivation

In the classical definition of conditional relative information $$I_{q\Vert p}(Y|X),$$ we have probabilistic measures $$p, q$$ over the measure space $$(\mathcal{X},\Sigma)$$. These measures can be thought of as assigning numbers to pure states, or numbers to subsets of states. 

In the quantum setting, the distinction between the state space and the state of the system becomes less clear. The system is in a quantum entanglement of states, and it is incorrect to think of it as uncertainty over being in one particular pure state. To allow for these kinds of mixed states, we prefer now to think of $$p,q$$ as being generalized states of a system and as points of some space $$\Omega$$, the collection of generalized states.

It does not matter if these generalized states are truths of the universe, or just constructs of our beliefs. We will treat them as objects of some geometric space, and we are interested in measuring the distance between these objects.

Drawing inspiration from the geometric semantics of homotopy type theory where equalities between terms of a type are seen as paths between points of a space, we will more generally think of $$p, q$$ as terms of some type $$A$$ separated by some distance.

Of course, there is a lot of structure in actual measures $$p,q$$ that we are ignoring when we only think of them as terms of a type. Most of this structure in the measures are _generated_ by the measure of primitive subsets, such as atomic measures for finite state spaces, interval measures for the real line, and cylinder measures for Markov chains. Therefore, we prefer to pass the burden of book-keeping to _constructions_ that we can perform with types. We may specify the information content of certain primitive types and construct our way to the information content of more complex types. We then try to derive the desired probabilities or measures from this information content. Let $$\mathcal{U}$$ denote the universe of all types which can be generated or constructed from the primitive ones.

In classical information theory, $$X,Y$$ are random variables, i.e. measurable maps from the state space $$\mathcal{X}$$ to some real vector space $$\mathbb{R}^n.$$ Traditionally, we expect $$X, Y$$ to take certain states in $$\mathbb{R}^n$$ when observed. We generalize this by thinking of an observation or _view_ as a map $$f:\Omega \rightarrow A$$ that produces a distribution of $$(X,Y)$$ given the original distribution $$q$$ in $$\Omega$$. We also have a view $$g:\Omega \rightarrow B$$ where we think of $$B$$ classically as the space of distributions of $$X.$$ 

Given a view $$g: \Omega \rightarrow B,$$ we may ask what is the distance or relative information from the distribution $$g(p)$$ to $$g(q).$$ As we gather more information, we could have another view $$f: \Omega \rightarrow A.$$ We could ask about the increase in relative information between $$f(q)$$ and $$f(p)$$ given that we already know the distance between $$g(q)$$ and $$g(p).$$ 

To compute the relative information $$I_{q\Vert p}(Y|X),$$ there are two important maps between spaces $$A$$ and $$B$$ of distributions that we need to consider. The first is the marginalization $$\pi : A \rightarrow B,$$ that classically maps $$q(X,Y)$$ to $$q(X),$$ so we have a bundle over $$B.$$ The second is the inference map $$\sigma : B \rightarrow A,$$ that classically maps $$q(X)$$ to $$p(Y|X)q(X)$$ via multiplication by the conditional distribution $$p(Y|X)$$ of the model, so we have a section of the bundle. Classically, 

$$ \displaystyle
I_{q\Vert p}(Y|X) = \int q(dy,dx) \log \frac{q(y,x)}{p(y|x)q(x)}$$

where $$q(x)$$ is defined by the bundle $$\pi$$ and $$p(y|x)q(x)$$ is defined by the section $$\sigma.$$



We write this increase in information as $$I_{q \Vert p}(c|b).$$ Therefore, relative information should satisfy the chain rule

$$ \displaystyle
I_{q \Vert p}(c,b) = I_{q \Vert p}(c|b) + I_{q \Vert p}(b).$$

A category-theoretic way to think about the chain rule is to consider the projections

$$ f : C\times B \rightarrow B$$

$$ g : B \rightarrow *$$

$$ g \circ f : C\times B \rightarrow *$$

where $$*$$ is the unit type or terminal type. Then the chain rule can be written as

$$ \displaystyle
I_{q \Vert p}(g \circ f) = I_{q \Vert p}(f) + I_{q \Vert p}(g)$$

so $$I_{q \Vert p}$$ is a functor between the appropriate category of views and their morphisms. We call these morphisms _transforms_ because they are measure-preserving transformations in the context of measure theory and dynamical systems. The relative information $$I_{q \Vert p}(f)$$ is the _loss_ in relative information when a view passes through the transform $$f$$. 

In our theory, we propose that information should always be relative. While it is possible to define information as an absolute quantity, such as with classical entropy, problems arise when we try to generalize that definition to new contexts, such as with differential entropy which is an attempt to extend classical entropy to continuous random variables. Classical entropy can be reframed as a relative concept by defining it as the relative information to a uniform distribution (up to an additive constant) or as the relative information between a random variable and two independent copies of itself.

Eventually, our goal is to write down axiomatic properties satisfied by relative information. We will then derive all other important results in information theory by means of these axioms. These axioms also provide a way to define a general ring of values which relative information can take. We first construct a ring of tuples $$(A,q,p,b,c,f)$$ which we will define later, that is closed under some binary operations $$\oplus$$ and $$\otimes$$ between the tuples. We then consider a suitable quotient by the axiomatic relations of relative information. The result is a Grothendieck ring or motivic ring of typed objects which carry important structural information about the types.

## Types and terms

Let $$a : A$$ denote a term $$a$$ of type $$A.$$

We may interpret $$a$$ as a probability measure and $$A$$ as the space of all probability measures or distributions over some given state space $$X$$ and some sigma algebra $$\mathcal{X}.$$

Alternatively, we may interpret $$a$$ as a mixed state or a pure state and $$A$$ as the space of all mixed or pure states over some finite set of pure states. A mixed state is a convex mixture of pure states. A pure state could be a pure quantum state or an atomic state in a finite discrete probability space.

Given $$a:A$$ and $$b:B$$, we have 

$$ \displaystyle
a \oplus b : A \oplus B$$

where $$A \oplus B$$ is the sum or coproduct of the two types. This coproduct is sometimes written as the Cartesian product $$A \oplus B := A \times B$$ which is the type of pairs $$a \oplus b := (a,b)$$ of terms $$a : A$$ and $$b: B.$$ We will avoid the terminology "Cartesian product" so that there is no confusion with dependent products which we will look at later. 

When $$A$$ and $$B$$ are spaces of measures over measure spaces $$(X_A,\mathcal{X}_A)$$ and $$(X_B,\mathcal{X}_B),$$ the sum $$A \oplus B$$ is the space of distributions on the measure space 

$$ \displaystyle
(X_A \sqcup X_B, \mathcal{X}_A \oplus \mathcal{X}_B)$$ 

where $$\mathcal{X}_A \oplus \mathcal{X}_B$$ is the sigma algebra generated by the sets in $$\mathcal{X}_A$$ and $$\mathcal{X}_B$$ viewed as subsets of the disjoint union $$X_A \sqcup X_B$$ by the inclusions $$X_A \hookrightarrow X_A \sqcup X_B$$ and $$X_B \hookrightarrow X_A \sqcup X_B.$$ Given measures $$a : A$$ and $$b:B,$$ we define

$$ \displaystyle
(a\oplus b)(S) = a(S\cap X_A) + b(S\cap X_B)$$

for all $$S \subset X_A \sqcup X_B$$, $$S \in \mathcal{X}_A \oplus \mathcal{X}_B.$$

Given $$a:A$$ and $$b:B$$, we have 

$$a \otimes b : A \otimes B$$

where $$A \otimes B$$ is the product or tensor product of the two types. This product is sometimes written as the function type $$A \otimes B := A \rightarrow B$$ which is the type of all functions from $$A$$ to $$B.$$ The function $$a\otimes b$$ maps the term $$a : A$$ to the term $$b:B.$$

For example, when $$A$$ and $$B$$ are vector spaces, then $$A \otimes B$$ is the space of linear transformations from $$A$$ to $$B.$$ The function $$a \otimes b$$ is the linear transformation that maps the vector $$a : A$$ to the vector $$b : B,$$ but maps to the zero vector $$0 :B$$ all other vectors $$a' : A$$ which are orthogonal to $$a.$$ 

More generally, given a multicategory $$\mathcal{U}$$ of objects (e.g. vector spaces) and multimorphisms (e.g. multilinear maps), a tensor product $$A \otimes B$$ (if it exists) is an object in $$\mathcal{U}$$ with a multimorphism $$A,B \rightarrow A \otimes B$$ such that all other multimorphisms $$A, B \rightarrow C$$ must factor through $$A \otimes B$$ as

$$\displaystyle
A,B \rightarrow A\otimes B \rightarrow C.$$

Typically, one can construct the tensor product by first constructing a free object in $$\mathcal{U}$$ that is generated by all formal pairs $$a \otimes b.$$ The pairs represent the image of $$(a,b)$$ under $$A,B \rightarrow A\otimes B.$$ We then quotient out relations between the pairs $$a \otimes b$$ that arise from the multimorphic properties of $$A,B \rightarrow A\otimes B,$$ e.g. $$(a_1+a_2)\otimes b = a_1\otimes b + a_2\otimes b.$$


When $$A$$ and $$B$$ are spaces of measures over measure spaces $$(X_A,\mathcal{X}_A)$$ and $$(X_B,\mathcal{X}_B),$$ the product $$A \otimes B$$ is the space of distributions on the measure space 

$$ \displaystyle
(X_A \times X_B, \mathcal{X}_A \otimes \mathcal{X}_B)$$ 

where $$X_A \times X_B$$ is the Cartesian product of the two spaces and $$\mathcal{X}_A \otimes \mathcal{X}_B$$ is the tensor product of the two sigma algebras. Given measures $$a : A$$ and $$b:B,$$ their tensor product $$a\otimes b$$ satisfies

$$ \displaystyle
(a\otimes b)(S) = a(S_A)  b(S_B)$$

for all $$S \subset X_A \times X_B$$, $$S \in \mathcal{X}_A \otimes \mathcal{X}_B,$$ where $$S_A$$ and $$S_B$$ is the projections of $$S$$ on $$X_A$$ and $$X_B$$ respectively.

The above notions of sums and products in simple type theory generalizes to dependent sums and dependent products in dependent type theory. Given a family $$B(\cdot)$$ of types over $$A,$$ we have pairs and the _dependent sum_

$$ \displaystyle 
(a,b) : \sum_{a : A} B(a),$$

and we have functions and the _dependent product_

$$ \displaystyle 
\lambda a.b(a) : \prod_{a:A} B(a).$$

In this post, we will not talk about the information theory of dependent sums and products. We will focus on studying the simpler case when they are pair types and function types respectively.

## Valuations, bundles and sections

Let $$R$$ be a commutative _rig_ (a ring without negatives; also known as _semiring_) [[BF14]](#ref-BF14), with addition $$+$$ and multiplication $$\times$$ as binary operations, and with additive unit $$0$$ and multiplicative unit $$1.$$ 

**Example**. The nonnegative reals $$[0,\infty)$$ with classical sum and product as $$+$$ and $$\times.$$

**Example**. The tropical semiring $$\mathbb{R} \cup \{\infty\}$$ with minimum and classical sum as $$+$$ and $$\times.$$

Let $$\mathcal{U}$$ be a universe of types that contains $$R$$ as a type. We assume that the universe $$\mathcal{U}$$ is equipped with a notion of type equality, and a notion of term equality within each type.
Suppose $$\mathcal{U}$$ is closed under binary operations $$\oplus$$ and $$\otimes$$ on types where $$\otimes$$ distributes over $$\oplus.$$ As a category with types as objects, $$\mathcal{U}$$ is a _rig category_ or a _bimonoidal category_.

Let $$v$$ be a _valuation_ mapping terms $$a : A,$$ $$A : \mathcal{U},$$ to $$v(a) : R$$ that satisfies the following.

1. **Identity**. For all $$r : R,$$

$$ \displaystyle
v(r)=r.$$

2. **Sum rule**. For all $$A, B :\mathcal{U},$$ $$a:A,$$ $$b:B,$$

$$ \displaystyle
v(a\oplus b) = v(a)+v(b).$$

3. **Product rule**. For all $$A,B :\mathcal{U},$$ $$a:A,$$ $$b:B,$$

$$ \displaystyle
v(a\otimes b) = v(a) v(b).$$

**Example** (measure theory). If $$a$$ is a measure in the set $$A$$ of finite measures on some measure space, then the total measure $$v(a)$$ is a valuation with values in $$[0,\infty).$$

**Example** (tropical geometry). Let $$K[x^{\pm}] :=K[x_1^{\pm}, \ldots, x_n^{\pm}]$$ be the ring of Laurent polynomials over a field $$K$$ with valuation $$\text{val}:K\rightarrow \Gamma.$$ Let $$\mathbb{k}$$ be the residue field of $$K.$$ Let $$\mathcal{U}$$ be the collection of homogeneous ideals $$A$$ in $$K[x^{\pm}]$$ which we will call _types_. We define _terms_ $$a$$ to be homogeneous principal ideals in $$K[x^{\pm}],$$ and we write $$a : A$$ if the principal ideal $$a$$ is contained in the homogeneous ideal $$A.$$ Let us fix weights $$w \in \Gamma^{n}.$$ For each type $$A : \mathcal{U}$$ and term $$a = \langle f \rangle : A,$$ we define the valuation 

$$ \displaystyle
v(a) := \text{in}_w f \,\,: \,\,R :=\mathbb{k}[x_1, \ldots, x_n].$$

to be the _initial form_ [[MS09]](#ref-MS09) of $$f$$ with respect to $$w.$$ One can check that this function is independent of the choice of generator $$f$$ and that it satisfies the required rules, where the sum and product of terms and types is the sum and product of ideals.

We assume that there is a collection of maps of the form $$\pi_{AB} : A \rightarrow B$$ where $$A, B :\mathcal{U},$$ which we call _bundles_. The bundles are closed under composition and they satisfy

1. **Valuation**. For all $$A : \mathcal{U},$$ the valuation $$v$$ restricted to $$A$$ is a bundle

$$ \displaystyle
v_A : A \rightarrow R.$$

2. **Identity**. For all $$A : \mathcal{U},$$ the identity map is a bundle 

$$ \displaystyle
\pi_{AA} := \text{id}_A :A\rightarrow A.$$

3. **Uniqueness**. For all $$A,B : \mathcal{U},$$ there is at most one bundle 

$$ \displaystyle
\pi_{AB} : A \rightarrow B.$$ 

Because the bundles are closed under composition and by uniqueness, we must have

$$ \displaystyle
\pi_{AC} = \pi_{BC} \circ \pi_{AB}.$$

for all bundles $$\pi_{AB}, \pi_{BC}, \pi_{AC}.$$ Consequently, the types of $$\mathcal{U}$$ form a poset, with $$A \geq B$$ if there is a bundle $$A \rightarrow B.$$ Moreover, by compositionality

$$ \displaystyle 
v_A = v_B \circ \pi_{AB},$$

all bundles $$\pi_{AB}$$ must be _value-preserving_.

In the category with types as objects and value-preserving maps as morphisms, $$R$$ is a terminal object because from every other type $$A$$ we have a unique value-preserving map $$v_A : A \rightarrow R,$$ namely the valuation $$v$$ restricted to $$A.$$

Our universe $$\mathcal{U}$$ has exactly one bundle collection but we allow many different collections of _sections_. Each collection $$\sigma$$ is closed under composition and satisfies

1. **Correspondence**. For all bundles $$\pi_{AB} : A \rightarrow B,$$ there is exactly one section

$$ \displaystyle
\sigma_{BA} : B \rightarrow A.$$

2. **Splitting**. For all bundles $$\pi_{AB} : A \rightarrow B,$$ we have

$$ \displaystyle
\pi_{AB} \circ \sigma_{BA} = \text{id}_B.$$

Again, by closure and correspondence, we must have for all sections $$\sigma_{BA}, \sigma_{CB}, \sigma_{CA},$$

$$ \displaystyle
\sigma_{CA} = \sigma_{BA} \circ \sigma_{CB}.$$

Because the bundles are value-preserving, the sections must also be value-preserving

$$ \displaystyle
v \,\sigma_{BA}(b) = v\, \pi_{AB} \,\sigma_{BA} (b) = v(b).$$

$$
\begin{array}{ccccc}
& & (w \otimes x) \otimes (y \otimes z)
\\
\\
& {}^{\mathllap{\alpha_{w \otimes x, y, z}}} \nearrow
& &
\searrow {}^{\mathrlap{\alpha_{w,x,y \otimes z}}}
\\
\\
((w \otimes x ) \otimes y) \otimes z
& &  & & 
(w \otimes (x \otimes (y \otimes z)))
\\
\\
{}_{\mathllap{\alpha_{w,x,y}} \otimes id_z }\,\, \big\downarrow 
& &  & &  
\big\uparrow \,\, {}_{\mathrlap{ id_w \otimes \alpha_{x,y,z} }}
\\
\\
(w \otimes (x \otimes y)) \otimes z
& &  \underset{\alpha_{w,x \otimes y, z}}{\longrightarrow} & & 
w \otimes ( (x \otimes y) \otimes z )
\end{array}
$$

## Relative information

Given two transforms $$h_i: A_i\rightarrow B_i,$$ from $$f_i : \Omega_i \rightarrow A_i$$ to $$g_i :\Omega_i\rightarrow B_i,$$ for $$i \in \{1,2\},$$ we define the sum $$h_1 \oplus h_2$$ as the transform from $$f_1 \oplus f_2$$ to $$g_1 \oplus g_2$$ as the map

$$ \displaystyle
h_1 \oplus h_2 : A_1 \times A_2 \rightarrow B_1\times B_2,$$

$$ \displaystyle
(h_1 \oplus h_2)(a_1,a_2) = (\, h_1(a_1),\, h_2(a_2)\,).$$

It is easy to check that $$h_1 \oplus h_2$$ commutes with $$f_1\oplus f_2$$ and $$g_1\oplus g_2.$$ We define the product $$h_1 \otimes h_2$$ as the transform from $$f_1 \otimes f_2$$ to $$g_1 \otimes g_2$$ as the map

$$ \displaystyle
h_1 \otimes h_2 : A_1 \otimes A_2 \rightarrow B_1\otimes B_2,$$

$$ \displaystyle
(h_1 \otimes h_2)(a_1 \otimes a_2) = h_1(a_1) \otimes h_2(a_2).$$




































The sections $$\sigma_{RA}:R \rightarrow A$$ for $$A : \mathcal{U}$$ are interesting, because for each value $$r : R,$$ the collection $$\sigma$$ gives us precisely one term $$\sigma_{RA}(r) : A$$ for each type $$A :\mathcal{U}.$$ 

In classical information theory, $$R$$ is often the extended real numbers, $$\mathbb{R} \cup \{\infty\}.$$ In motivic integration, $$R$$ could be a motivic ring like the Grothendieck ring of varieties.

Given a type $$\Omega:\mathcal{U},$$ an _$$\Omega$$-view_ is a grade-preserving map $$f: \Omega \rightarrow A$$ for some type $$A : \mathcal{U}.$$ We call it a _view_ when the annotation $$\Omega$$ is obvious.

Let $$f:\Omega\rightarrow A$$ and $$g:\Omega\rightarrow B$$ be two $$\Omega$$-views. An $$\Omega$$-transform from $$f$$ to $$g$$ is a grade-preserving map $$h:A\rightarrow B$$ that commutes with $$f, g,$$ namely $$g = h \circ f.$$ Again, we drop the annotation $$\Omega$$ when it is obvious.

From the category theory perspective, we have an _under category_ with $$\Omega$$-views as objects and the $$\Omega$$-transforms as morphisms.

Let the _identity $$\Omega$$-view_ be the identity map $$\text{id}_\Omega: \Omega \rightarrow \Omega$$ on $$\Omega.$$ Every view $$f:\Omega \rightarrow A$$ can now be interpreted as a transform from $$\text{id}_\Omega$$ to $$f.$$ 

Let the _terminal $$\Omega$$-view_ be the grade function $$Z_\Omega : \Omega \rightarrow R.$$ The terminal view is a terminal object in the under category, because for all views $$f:\Omega \rightarrow A$$ we have a unique transform from $$f$$ to $$Z_\Omega$$ induced by the grade function $$Z_A : A \rightarrow R.$$  

We may interpret the terminal view $$Z_\Omega : \Omega \rightarrow R$$ as a transform from $$\text{id}_\Omega$$ to $$Z_\Omega.$$ We will call it the _terminal $$\Omega$$-transform_.

Given two views $$f_i: \Omega_i \rightarrow A_i$$ for $$i \in \{1,2\},$$ we define the sum

$$ \displaystyle
f_1 \oplus f_2 : \Omega_1 \times \Omega_2 \rightarrow A_1 \times A_2,$$

$$ \displaystyle
(f_1 \oplus f_2)(\omega_1, \omega_2) = (\,f_1(\omega_1), \,f_2(\omega_2)\,),$$

and the product

$$ \displaystyle
f_1 \otimes f_2 : \Omega_1 \otimes \Omega_2 \rightarrow A_1 \otimes A_2,$$

$$ \displaystyle
(f_1 \otimes f_2)(\omega_1\otimes \omega_2) = f_1(\omega_1) \otimes f_2(\omega_2).$$

The image of $$f_1 \otimes f_2$$ for other terms of $$\Omega_1 \otimes \Omega_2$$ are generated from the relations above by multilinearity or multimorphism properties of the tensor product.



Let $$I_{q \Vert p}(f)$$ be an $$R$$-valued function that takes a type $$\Omega :\mathcal{U},$$ two terms $$p,q  : \Omega,$$ two views $$f: \Omega \rightarrow A,$$ $$g:\Omega \rightarrow B,$$ and a transform $$h: A \rightarrow B$$ as inputs. The inputs $$\Omega, A, B, f, g$$ are implicit in $$p,q,h.$$ We say that $$I$$ is a _relative $$\mathcal{U}$$-information_ if it satisfies the following.

1. **Equality**. For all $$p,q:\Omega,$$ we have $$p = q$$ if and only if 

$$ \displaystyle
I_{q\Vert p}(Z_\Omega) = 0.$$

2. **Substitution**. For all $$p,q:\Omega,$$ and view $$f : \Omega \rightarrow A,$$

$$ \displaystyle
I_{q\Vert p}(Z_A) = I_{f(q)\Vert f(p)}(Z_A).$$

3. **Chain rule**. For all $$p,q:\Omega,$$ views $$f_i : \Omega \rightarrow A_i$$ for $$i \in \{1,2,3\}$$ and transforms $$h_j: A_j\rightarrow A_{j+1}$$ for $$j \in\{1,2\},$$

$$ \displaystyle
I_{q\Vert p}(h_2\circ h_1) = I_{q\Vert p}(h_1) + I_{q\Vert p}(h_2). $$

4. **Sum rule**. For all $$p_i,q_i: \Omega_i,$$ views $$f_i:\Omega_i \rightarrow A_i,$$ $$g_i:\Omega_i \rightarrow B_i$$ and transforms $$h_i: A_i\rightarrow B_i$$ for $$i \in \{1,2\},$$ 

$$ \displaystyle
I_{q_1\oplus q_2 \Vert p_1\oplus p_2}(h_1\oplus h_2) = I_{q_1\Vert p_1}(h_1) + I_{q_2\Vert p_2}(h_2). $$

5. **Product rule**.  For all $$p_i,q_i: \Omega_i,$$ views $$f_i:\Omega_i \rightarrow A_i,$$ $$g_i:\Omega_i \rightarrow B_i$$ and transforms $$h_i: A_i\rightarrow B_i$$ for $$i \in \{1,2\},$$ 

$$ \displaystyle
I_{q_1\otimes q_2 \Vert p_1\otimes p_2}(h_1\otimes h_2) = Z(q_2) I_{q_1\Vert p_1}(h_1) + Z(q_1) I_{q_2\Vert p_2}(h_2). $$

6. **Continuity**. Given appropriate topologies on each type $$\Omega : \mathcal{U},$$ relative information $$I_{q\Vert p}(f)$$ is a continuous function of $$q$$ and $$p.$$

The last property is important only if the rig $$R$$ is defined as the completion of some space, such as $$\mathbb{R}$$ as a completion of $$\mathbb{Q}.$$ The types $$\Omega :\mathcal{U}$$ are often constructed from $$R$$ using the operations $$\oplus$$ and $$\otimes,$$ with topologies induced from that of $$R.$$

When the rig $$R$$ is a field and the types are vector spaces over $$R$$, we can imagine normalizing each term by its grade, i.e. defining their grade to be $$1.$$ We will also need to normalize the above notion of relative information. In that case, the sum rule is only applicable to convex sums of terms and not any general sum. Strictly speaking, this normalized relative information does _not_ satisfy the sum rule listed above. 

It is also tempting to remove the sum or product rule in favor of the other, because for many type universes, one of the constructions can be written in terms of the other simpler constructions. For example, in the Calculus of Inductive Constructions, the dependent sum can be written as an inductive family. We prefer the approach in Homotopy Type Theory where dependent sums, dependent products, natural numbers and identity types are seen as being more primitive than inductive families. 

## Measures over finite sets

As the first example, we will construction the information theory of measures over finite sets. This will include probabilistic measures as a special case.

Let $$R$$ be the rig $$[0,\infty]$$ of extended non-negative real numbers. We define $$x+\infty = \infty$$ for all $$x \in [0, \infty],$$ $$x \cdot \infty= \infty$$ for all $$x \in (0,\infty],$$ and $$0 \cdot \infty = 0.$$

We assume the universe $$\mathcal{U}$$ consists of $$R$$-modules, and it contains the module $$R = [0,\infty]$$ with grading $$Z(r) = r,$$ $$r \in [0,\infty].$$ Let $$A \oplus B$$ be the direct sum of $$R$$-modules $$A,B:\mathcal{U};$$ it contains pairs $$a\oplus b := (a,b),$$ $$a:A,$$ $$b:B$$ with grading $$Z(a\oplus b) = Z(a)+Z(b).$$ Let $$A \otimes B$$ be the tensor product of $$R$$-modules $$A,B:\mathcal{U};$$ it contains finite formal sums 

$$ \displaystyle 
a_1 \otimes b_1 + \cdots + a_n \otimes a_n$$   

satisfying multilinear relations of the form

$$ \displaystyle 
(a_1 + a_2)\otimes b =  a_1\otimes b + a_2 \otimes b,$$

$$ \displaystyle 
a\otimes (b_1+b_2) =  a\otimes b_1 + a \otimes b_2,$$

$$ \displaystyle 
(ra)\otimes b =  a\otimes (rb),$$

and is graded by 

$$ \displaystyle
Z(a_1\otimes b_1+\cdots+a_n\otimes b_n) = Z(a_1) Z(b_1)+\cdots+Z(a_n)Z(b_n).$$

Each $$R$$-module is isomorphic to some $$R^n$$ which can be thought of as the collection of non-negative measures on an $$n$$-element set.

As for the transforms, we consider grade-preserving linear maps $$h:R^m \rightarrow R^n$$ which can be written as a matrix

$$ \displaystyle
h = (h_{ji}), \quad h(x_1, \ldots, x_m)_j = \sum_{i=1}^m h_{ji}x_i,$$

satisfying 

$$ \displaystyle
\sum_{j=1}^n h_{ji} = 1 \quad \text{for all }1\leq i \leq m.$$

Given $$q=(q_1, \ldots, q_d) : R^d,$$ we denote

$$ \displaystyle 
q(X=i) = q_i, \quad q(Y=j \vert X=i) = h_{ji},$$

$$ \displaystyle
q(X=i, Y=j) = q(Y=j\vert X=i) q(X=i)$$

$$ \displaystyle 
q(Y=j) = \sum_{i=1}^m q(X=i, Y=j),$$

$$ \displaystyle 
q(X=i | Y=j) = \frac{q(X=i, Y=j)}{q(Y=j)}$$

and similarly for $$p=(p_1, \ldots, p_d) : R^d.$$ Note that these are not probabilities because the grade or total measure may not be one. Also, some care has to be taken when any of the above quantities are zero or infinity.

We may now define the relative information

$$ \displaystyle
I_{q\Vert p}(h) = \sum_{i,j} q(Y=j) q(X=i | Y=j)  \log \frac{q(X=i | Y=j)}{p(X=i | Y=j)}$$

where again we take some care when any of the quantities are zero or infinity. It is not hard to show that this form of relative information satisfies the properties listed above.

The converse is to show that if we have a function $$I_{q\Vert p}(h)$$ with the above properties, then it must be some constant multiple of this logarithmic formula. The proof [[BF14]](#ref-BF14) is tricky, so we will not be repeating it here. We will instead demonstrate how the proof constructs the value of relative information in some simple examples.

First we prove the following lemma. For all $$p,q:\Omega,$$ views $$f_i : \Omega \rightarrow A_i$$ for $$i \in \{1,2\}$$ and a transform $$h: A_1\rightarrow A_2,$$ because $$Z_A = Z_B\circ h,$$ by chain rule and substitution,

$$ \begin{array}{rl}
\displaystyle I_{q\Vert p}(h)  & =
\displaystyle I_{q\Vert p}(Z_A) - I_{q\Vert p}(Z_B)
\\ & \\ & =
\displaystyle I_{f(q)\Vert f(p)}(Z_A) - I_{g(q)\Vert g(p)}(Z_B).
\end{array} $$

Now, treating $$h$$ as an $$A$$-view and $$h, Z_A, Z_B$$ as $$A$$-transforms,

$$ \begin{array}{rl}
\displaystyle I_{f(q)\Vert f(p)}(h)  & =
\displaystyle I_{f(q)\Vert f(p)}(Z_A) - I_{f(q)\Vert f(p)}(Z_B)
\\ & \\ & =
\displaystyle I_{f(q)\Vert f(p)}(Z_A) - I_{h\circ f(q)\Vert h\circ f(p)}(Z_B)
\\ & \\ & =
\displaystyle I_{f(q)\Vert f(p)}(Z_A) - I_{g(q)\Vert g(p)}(Z_B).
\end{array} $$

Therefore, $$I_{q\Vert p}(h)  = I_{f(q)\Vert f(p)}(h).$$

For all $$\alpha,\beta : [0,\infty),$$ let $$\Omega = R \oplus R \oplus R = R^3$$ with 

$$ \displaystyle 
q = (1,0,0) : \Omega,$$

$$ \displaystyle
p = (\alpha\beta,\alpha(1-\beta),1-\alpha) : \Omega.$$ 

Let $$A = R^3,$$ $$B = R^2,$$ $$C = R^2,$$ $$D = R.$$ Consider the following transforms

$$ \displaystyle
\varphi : R^2 \rightarrow R, \quad \varphi(r_1, r_2) = r_1+r_2.$$

$$ \displaystyle
g_1 = \varphi \oplus \text{id} : A \rightarrow B, \quad g_2 = \varphi : B \rightarrow D$$

$$ \displaystyle
h_1 = \text{id} \oplus \varphi : A \rightarrow C, \quad h_2 = \varphi : C \rightarrow D$$

and views

$$ \displaystyle
f_A = \text{id} : \Omega \rightarrow A$$

$$ \displaystyle
f_B  = g_1 \circ f_A = g_1 : \Omega \rightarrow B$$

$$ \displaystyle
f_C = h_1 \circ f_A = h_1 : \Omega \rightarrow C$$

$$ \displaystyle
f_D = h_2 \circ h_1 \circ f_A = Z_\Omega : \Omega \rightarrow D$$

We have equality of compositions $$g_2 \circ g_1 = h_2 \circ h_1 = Z_\Omega.$$

$$ \displaystyle
\begin{array}{rl}
\displaystyle I_{q\Vert p}(Z_\Omega) & = 
\displaystyle I_{q\Vert p}(g_1) + I_{q\Vert p}(g_2)
\\ & \\ & =
\displaystyle I_{q\Vert p}(g_1) + I_{g_1(q)\Vert g_1(p)}(\varphi)
\\ & \\ & =
\displaystyle I_{q\Vert p}(g_1) + I_{(1,0)\Vert (\alpha,1-\alpha)}(\varphi)
\end{array} $$

## Poset of bundles

Let $$\mathcal{U}$$ be a poset (**look up definition**) category with an initial object $$\Omega$$ and a terminal object $$R.$$ We assume that $$R$$ is a rig. (**can we put a rig as an object?**) 

There is a morphism $$f_A$$ from $$\Omega$$ to each object $$A$$, which we call a view. From each object $$A$$, we have a unique morphism $$Z_A$$ to $$R,$$ which we call a grade function. Every morphism $$\pi: A \rightarrow B$$ is a bundle, to which we have a family of sections $$\sigma_p : B \rightarrow A$$ over $$p : \Omega.$$ The bundle and section must satisfy $$\pi \circ \sigma_p = \text{id}_B$$ the identity on $$B$$ for all $$p : \Omega.$$ Together, the bundles $$\pi$$ form a commutative diagram, and for each $$p : \Omega,$$ the sections $$\sigma_p$$ also form a commutative diagram.

Immediately, we note the following.

1. Given $$p :\Omega,$$ for the bundle $$Z_\Omega :\Omega \rightarrow R,$$ we have some section $$\sigma_p : R \rightarrow Z_\Omega$$ and it must satisfy $$\sigma_p \circ Z_\Omega(p) = p.$$

## References

<a id="ref-B11"></a>[[B11]](#ref-B11) Baez, John C., Tobias Fritz, and Tom Leinster. "A characterization of entropy in terms of information loss." _Entropy_ 13, no. 11 (2011): 1945-1957.

<a id="ref-BF14"></a>[[BF14]](#ref-BF14) Baez, J. C., and T. Fritz. "A Bayesian characterization of relative entropy." _Theory and Applications of Categories_ 29 (2014): 422-456.

<a id="ref-L19"></a>[[L19]](#ref-L19) Leinster, Tom. "A short characterization of relative entropy." _Journal of Mathematical Physics_ 60, no. 2 (2019): 023302.

<a id="ref-MS09"></a>[[MS09]](#ref-MS09) Maclagan, Diane, and Bernd Sturmfels. "Introduction to tropical geometry." Graduate Studies in Mathematics 161 (2009): 6.
