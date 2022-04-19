---
layout: post
title: Adjunctions
---

Read about the Curry-Howard-Lambek correspondence. Some call it the holy trinity of Logic, Computation and Categories. Lambek adds the "objects as propositions" and "arrows as proofs" part to the mix. You may need to learn some basic category theory.

[http://arca.di.uminho.pt/quantum-logic-1920/CategoriesAndLogic.pdf](http://arca.di.uminho.pt/quantum-logic-1920/CategoriesAndLogic.pdf)

Equalities, Equivalence and Adjunctions. Read about how adjunctions are a powerful generalization of equivalence. Might be useful in theorem-proving by transporting the problem to a different category.

[https://www.math3ma.com/blog/what-is-an-adjunction-part-2](https://www.math3ma.com/blog/what-is-an-adjunction-part-2)

Yoneda Lemma is very important in category theory : it tells us that all objects are completely determined by their relationship to other objects. It is the central dogma of category theory.

[https://www.math3ma.com/blog/the-yoneda-perspective](https://www.math3ma.com/blog/the-yoneda-perspective)

Yoneda Lemma in category theory has the same content as the eliminator or induction principle for identity types or path types in type theory. For directed spaces, there is a directed Yoneda Lemma which is like the induction principle for directed path types. See second last slide of

[http://www.math.jhu.edu/~eriehl/Leeds-HTT-UF.pdf](http://www.math.jhu.edu/~eriehl/Leeds-HTT-UF.pdf)

[https://www.quora.com/Homotopy-Type-Theory-How-is-the-Yoneda-lemma-related-to-path-induction](https://www.quora.com/Homotopy-Type-Theory-How-is-the-Yoneda-lemma-related-to-path-induction)

The role of adjunctions in type theory is not well-explored, and could potentially be very interesting from the point of view of constructing proofs.

In category theory, loosely, an adjunction is a pair of functors $$F : D \rightarrow C$$ and $$G : C \rightarrow D$$ such that $$\text{Hom}(FY, X)$$ is naturally isomorphic to $$\text{Hom}(Y, GX)$$ for all objects $$X$$ in $$C$$ and $$Y$$ in $$D$$.

In type theory, we could think of $$C$$ and $$D$$ as collections of propositions (objects) and their proofs (arrows) but using different representations. A functor $$F : D \rightarrow C$$ converts props and proofs from $$D$$ to $$C$$, and vice versa for $$G$$. If $$F$$ and $$G$$ are adjoints of each other, the two collections $$C$$ and $$D$$ are not strictly equivalent, but there is some kind of weak equivalence: given props $$X$$ in $$C$$ and $$Y$$ in $$D$$, there is a one-to-one correspondence between proofs of $$FY \rightarrow X$$ and $$Y \rightarrow GX$$.

In other words, if we are stuck trying to prove, say, $$Z \rightarrow X$$ but we know that $$Z = FY$$, then we can try to prove $$Y \rightarrow GX$$ instead. This kind of adjunction can be seen in the hom-tensor adjunction from category theory, which becomes currying in type theory. Instead of trying to prove $$A \rightarrow (B \rightarrow C)$$, we could instead try to prove $$(A \times B) \rightarrow C$$ where we see $$A \times B$$ as a sigma type. This happens all the time in Coq without us knowing it, when we shift things from the context to the goal and vice versa.