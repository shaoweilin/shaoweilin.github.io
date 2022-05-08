---
date: 2022-01-22
excerpts: 2
---

# Parametric typeclasses aid generalization in program synthesis

We envision programming being done in top-down fashion. The human describes the goal (e.g. sorting), and the machine reduces it to smaller subgoals based on well-known heuristics (e.g. divide and conquer). The easier subgoals could even be fulfilled automatically. This top-down heuristics approach will be more amenable to machine learning. See my [Topos Institute talk](2021-04-22-proofs-as-programs-challenges-and-strategies-for-program-synthesis.md/) for more info.

## Challenges in applying machine learning to program synthesis

The problem with the current approach in type theory is as follows.

1. Programs are presented in a bottom-up fashion (e.g. defining insertions before defining insertion-sort). It is hard to see the parts of the proof which can be generalized to produce more powerful principles (e.g. divide and conquer) for problem solving.

2. When solving for a goal, the human/machine needs to remember the specific name (e.g. list_recursion) of the theorem to apply, rather than a general name (e.g. tree_recursion) parametrized by types/operators - parameters which can then be solved during proof search. As a result, a learning machine may think an algorithm works because of a specific function, rather than a more general principle.

## Parametricity to the rescue

This second point about using parametric classes to organize types (e.g. reflexivity) and operators (e.g. equality) and to overload notation, is known as parametricity. This idea is not new and has been explored by both category theorists and type theorists.

However, I believe Haskell was the first programming language to make serious use of classes (not the same as interfaces in object-oriented programming) to simplify code, and Coq and other proof assistants followed suit. As Bas Spitters mentioned in his paper, there is more we need to do to really make this a game changer, e.g. unification hints and rewriting/normalization rules.

## Examples of parametric strategy

For example, equivalence between types can be captured by parametric classes, and the following seminal paper describes how to extend types equivalences to constructions involving those types in a systematic way. Automated transport along equivalences is important if we want to switch between unary and binary numbers for efficiency in automated programming.

- Nicolas Magaud and Yves Bertot. 2000. "Changing Data Structures in Type Theory: A Study of Natural Numbers" (TYPES 2000).

This next paper summarizes the latest strategies in parametric transport, and relates them to univalent transport from homotopy type theory. There is also heavy use of classes in their approach.

- Tabareau, Nicolas, Éric Tanter, and Matthieu Sozeau. 2021. "The Marriage of Univalence and Parametricity" (JACM).

## Example - Noetherian induction

Let me end with a specific example involving well-founded relations and Noetherian induction.

Given a type A and a relation (R : A -> A -> Prop), we recursively define a term (x : A) to be accessible if y is accessible for all {y : A | R y x}. We say that R is well-founded if all terms (x : A) are accessible. It is a theorem that given a well-founded relation R and a predicate (P : A -> Prop), we can do Noetherian induction, i.e. suppose for all (x :A), we have (P x) if (P y) for all {y : A | R y x}; then (P x) for all (x : A).

If we do this without classes, we will have to carry a lot of information around with us. Let "Rwf" be the proof of the well-foundedness of R. Let "wf_implies_noetherian" be the function that takes in a proof of well-foundedness, and spits out a proof of noetherian-ness. Then, to do noetherian induction, we would have to apply "wf_implies_noetherian Rwf" at the start of our proof. See <https://github.com/coq/coq/blob/master/theories/Init/Wf.v#L55>.

If we do this with classes, we could have a class "Wellfounded" of all relations that are well-founded. We could also have a class "Noetherian" of all relations for which we can perform Noetherian induction. In this class, we have a generic name "noetherian_rect" 
(noetherian recursion for types) for the proof of noetherian-ness. 

We would then have R as an instance of Wellfounded, and S as an instance of Noetherian for all well-founded S. If we have a situation where Noetherian induction might be applicable, we just need to apply "noetherian_rect" with no extra parameters. The proof assistant will now compare the goal with the statement of noetherian-ness to guess what the relation R may be. It will then try to find an instance of the "Noetherian R" class, which could be something from the "Wellfounded R" class. Finally, it tries to find an instance of "Wellfounded R" and it sees that an instance was previously registered. All of this searching happens automatically in the background and no extra information needs to be carried by the human or learning machine. See below for the code.

```coq

Inductive Acc `(R : relation A) (x : A) : Prop :=
  Acc_intro : (forall y:A, R y x -> Acc R y) -> Acc R x.

Class Wellfounded `(R : relation A) :=
  wellfoundedness : forall a:A, Acc R a.

Class Noetherian `(R : relation A) :=
  noetherian_rect : forall P:A -> Type,
    (forall x:A, (forall y:A, R y x -> P y) -> P x) -> forall a:A, P a.

Instance wellfounded_as_noetherian `(R : relation A) `(Wellfounded A R) : Noetherian R.
Proof. intro. intros; apply (@Acc_rect A R); auto. Qed.

Section BinaryTree.

Variable T : Type.

Definition lengthOrder (ls1 ls2 : list T) := length ls1 < length ls2.

Lemma lengthOrder_wf' : forall len, forall ls,
  length ls <= len -> Acc lengthOrder ls.
Proof. [...] Defined.

Instance lengthOrder_wf : Wellfounded lengthOrder.
Proof. [...] Defined.

Fixpoint combine (ls1 ls2 : list T) : list T :=
  match ls1, ls2 with
  | nil, l2 => l2
  | l1, nil => l1
  | h1 :: t1, h2 :: t2 => h1 :: h2 :: combine t1 t2
  end.

Theorem tree_rect : forall (P : list T -> Type),
   P nil ->
  (forall t : T, P (cons t nil)) ->
  (forall l0 : list T, P l0 -> forall l1 : list T, P l1 ->
   P (combine l0 l1)) ->
  (forall l : list T, P l).
Proof. intros. apply noetherian_rect. intros. [...] Qed.

End BinaryTree.

```

In Coq, a similar proof-search is done when we use the tactic "reflexivity." The proof assistant searches for instances of the Reflexive class. Users can extend the Reflexive class with their own instances to make the tactic more powerful.

## Conclusion

You can see why such parametric strategies make problem-solving extremely compositional and top-down. They are inherently categorical (generalized algebraic theoretical to be precise) because the classes encode the relational properties of the types/operators but hides the underlying implementations. Proof-search at the top level can automatically trigger appropriate proof-search at lower levels.

## References

```{bibliography}
:filter: docname in docnames
```