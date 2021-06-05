---
layout: post
title: Logical frameworks
---

Logical frameworks are formal meta-languages for specifying different kinds of object theories (e.g. logical theories and type theories).

## What is LF?

LF is a logical framework [[HHP93]](#ref-HHP93) that formalizes Martin-Lof's logical framework and is itself based on type theory. It was designed to unify similarities between two forms of judgments.

* hypothetical judgments ($$J_1,\ldots, J_n \vdash J$$)
* generic judgements ($$x \in X_1,\ldots, x\in X_n \vert x \in X$$)

Hence, LF may be thought of as a meta-language for judgments. Here, the $$X_i$$ are _syntactic categories_, i.e. collections of strings. When we use the word _syntax_, we mean that we are referring to the strings in the language and their grammatical structures.

The logical theory or type theory that we describe with LF will be called the _object language_.

We will be using the presentation in [[L06]](#ref-L06), the paper where Unified Type Theory (UTT) was introduced. This presentation sticks closely to the original logical framework by Martin-Lof. Do note, however, that since Martin-Lof's framework, there has been newer versions such as Canonical LF which incorporates signatures and canonical forms. We follow the presentation in [[L06]](#ref-L06) to keep this introduction simple.

## What are the rules of LF?

We first present the rules [[L06]](#ref-L06) before describing their semantics.

![logical-framework-rules](https://shaoweilin.github.io/images/logical-framework-rules.png)

There are four forms of expressions - LF-kinds $$K$$, LF-types $$A$$ and LF-constants $$x$$. We prefix them with LF to distinguish them from the types and constants of the object language. The LF-types and LF-constants are essentially the types and terms of the LF language, while the LF-kinds are type families such as the dependent product kind $$(x:K_1)K_2$$

We also have contexts, which are lists of LF constant-kind pairs or LF constant-type pairs. They store the LF-kind or LF-type information of the LF-constants.

There are five forms of judgments in LF.

*   well-formed context ($$\Gamma \text{ valid}$$)
*   well-formed kind ($$\Gamma \vdash K \text{ kind}$$)
*   equality of kinds ($$\Gamma \vdash K = K'$$)
*   well-formed constant ($$\Gamma \vdash k : K$$)
*   equality of constants ($$\Gamma \vdash k = k' : K$$)

There is a special LF-kind called $$\textbf{Type}$$ which is the LF-kind of all LF-types. There is a Tarski operator $$El$$ that lifts an LF-type $$\Gamma \vdash A : \textbf{Type}$$ to an LF-kind $$\Gamma \vdash El(A) \text{ kind}$$. We will omit this lifting operator when it is obvious from the expression.

From the rules, it is interesting that dependent product kinds are all that is needed to specify the logical framework [[n20]](#ref-n20).

## Is LF consistent?

Since LF is just a meta-language for specifying object languages, consistency is not a concern at the level of the framework, but only relevant at the level of the object language. Moreover, it is not possible to define consistency for LF in the same way we define consistency for object languages via the ability to construct a term of the empty type.

## What role do universes play in the object languages of LF?

A universe $$U : \textbf{Type}$$ can be thought of as syntactic category that puts a collection of LF-kinds into a neat package after the LF-kinds have been defined. For example, suppose we have LF-kinds

$$Nat \text{ kind}$$

$$Bool \text{ kind}$$

Instead of putting the LF-constants $$nat$$ and $$bool$$  in $$\textbf{Type}$$, we could package them in $$U$$ so $$nat : El(U)$$ and $$bool : El(U)$$ while $$U : \textbf{Type}$$. We will need a new lifting operator $$T$$, so that $$T(nat) : \textbf{Type}, T(bool) : \textbf{Type}$$ and $$Nat = El(T(nat)), Bool = El(T(bool))$$.  Every LF-constant representing a type eventually lifts up to some LF-kind. Thus, we can think of universe management as a way of organizing a large collection of LF-kinds to retain consistency, among other goals.

## What issues does LF have, and how do recent extensions of LF deal with them?

In more recent logical frameworks, signatures have also been introduced in addition to contexts. These signature store constants which may be expanded to longer terms by definition.

We also want the logical framework to faithfully represent types from the object language. In other words, object types in the object language should have a corresponding LF-kind in the logical framework. However, there could be several LF-kinds mapping the same object type, because of eta-conversions, such as $$\text{suc}$$ and $$[x:Nat](\text{suc } x)$$ both mapping to the successor function in the object language. The unique LF-representation would be called a _canonical form_. To solve this problem, LF-kinds are constrained to only contain _LF-families_, and LF-families contain LF-objects. The judgments are refined to be of the following forms [[H12]](#ref-H12).

* signatures $$\Sigma \vdash$$ and equality $$\Sigma = \Sigma' \vdash$$
* contexts $$\Gamma \vdash_\Sigma$$ and equality $$\Gamma = \Gamma' \vdash_\Sigma$$
* kinds $$\Gamma \vdash_\Sigma K$$ and equality $$\Gamma \vdash_\Sigma K = K'$$
* families $$\Gamma \vdash_\Sigma A:K$$ and equality $$\Gamma \vdash_\Sigma A=A': K$$
* objects $$\Gamma \vdash_\Sigma M:A$$ and equality $$\Gamma \vdash_\Sigma M=M' : A$$

Of course, these judgments are not enough to solve the problem of canonical forms, but they are the starting point. The usual substitution rules need to be adjusted to depend on the type of the replacement object.

These recent frameworks also have bidirectional type-checking. More precisely, instead of just having the judgement $$a:A$$, we have judgements $$a \Rightarrow A$$ (when A is not given, we may infer that the type of $$a$$ is $$A$$) and $$a \Leftarrow A$$ (when A is given, we verify that $$a$$ is of type $$A$$).

## Should we use LF's type-checking capabilities to perform type-checks for the object language?

This is known as the _analytic presentation_ of the object language, as opposed to the _syntactic presentation_ of the object language [[H12]](#ref-H12). Analytic presentations are useful for computations, but they cause problems when we are trying to prove certain properties of the object language. In particular, when there are certain definitional equalities such as beta or eta reductions in the object language, we will need to represent them as equalities of the logical framework, so that type checking can proceed correctly. However, such representations make it difficult to prove properties of the object language that depend on these equalities. A hybrid syntactic-analytic approach is also possible, where we represent equalities using identity types.

## What about equations satisfied by the generators of the object language?

Robert Harper has some notes about a _semantic logical framework_ that allows for such equations [[H21]](#ref-H21).

## References

<a id="ref-L06"></a>[[L06]](#ref-L06) Zhaohui Luo, "[A unifying theory of dependent types: the schematic approach](https://www.researchgate.net/publication/225233240_A_unifying_theory_of_dependent_types_the_schematic_approach)," Logical Foundations of Computer Science, 2006\. DOI: 10.1007/BFb0023883

<a id="ref-HHP93"></a>[[HHP93]](#ref-HHP93) Harper R, Honsell F, Plotkin G. [A framework for defining logics.](http://homepages.inf.ed.ac.uk/gdp/publications/Framework_Def_Log.pdf) Journal of the ACM (JACM). 1993 Jan 2;40(1):143-84.

<a id="ref-H12"></a>[[H12]](#ref-H12) Harper R, _[Notes on logical frameworks](http://www.cs.cmu.edu/~rwh/papers/lfias/lf.pdf)_, 2012.

<a id="ref-H21"></a>[[H21]](#ref-H21) Harper, Robert. "[A Semantic Logical Framework](http://www.cs.cmu.edu/~rwh/papers/slf/slf.pdf)." _Unpublished draft. Dec.(cit. on p.)_ (2020). 

<a id="ref-n20"></a>[[n20]](#ref-n20) [https://ncatlab.org/nlab/show/logical+framework](https://ncatlab.org/nlab/show/logical+framework), retrieved 2020.
