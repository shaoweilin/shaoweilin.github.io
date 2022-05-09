Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

Require Export Coq.Classes.Init.
Require Import Coq.Program.Basics.
Require Import Coq.Program.Tactics.
Require Import Coq.Relations.Relation_Definitions.

Require Import FunInd Arith List.

Generalizable Variables A R.

Inductive Acc `(R : relation A) (x : A) : Prop :=
  Acc_intro : (forall y:A, R y x -> Acc R y) -> Acc R x.

Check Acc_rect.
(** Acc_rect
    : forall P : A -> Type,
       (forall x : A,
        (forall y : A, R y x -> Acc y) -> (forall y : A, R y x -> P y) -> P x) ->
       forall x : A, Acc x -> P x
 *)

Class Wellfounded `(R : relation A) := 
  wellfoundedness : forall a:A, Acc R a.

Class Noetherian `(R : relation A) :=
  noetherian_rect : forall P:A -> Type,
    (forall x:A, (forall y:A, R y x -> P y) -> P x) -> forall a:A, P a.

Instance wellfounded_as_noetherian `(R : relation A) `(Wellfounded A R) : Noetherian R.
Proof.
  intro. intros; apply (@Acc_rect A R); auto.
Qed.



Section TreeBtree.

Variable T : Type.

Definition lengthOrder (ls1 ls2 : list T) := length ls1 < length ls2.

Lemma lengthOrder_wf' : forall len, forall ls, 
  length ls <= len -> Acc lengthOrder ls.
Proof.
unfold lengthOrder; induction len.
- intros; rewrite Nat.le_0_r,length_zero_iff_nil in H; rewrite H; constructor;
  intros; inversion H0.
- destruct ls; constructor; simpl; intros.
  + inversion H0.
  + simpl in H; apply le_S_n in H; apply lt_n_Sm_le in H0; apply IHlen; 
    eapply Nat.le_trans; eassumption.
Defined.

Instance lengthOrder_wf : Wellfounded lengthOrder.
Proof.
red; intro; eapply lengthOrder_wf'; eauto.
Defined.

Fixpoint split_alt (ls : list T) : list T * list T :=
  match ls with
  | nil => (nil, nil)
  | h :: nil => (h :: nil, nil)
  | h1 :: h2 :: ls' =>
      let (ls1, ls2) := split_alt ls' in
        (h1 :: ls1, h2 :: ls2)
  end.

  Functional Scheme split_alt_rect := Induction for split_alt Sort Type.

  Lemma split_wf : forall len ls, 2 <= length ls <= len
    -> let (ls1, ls2) := split_alt ls in lengthOrder ls1 ls /\ lengthOrder ls2 ls.
  Proof.
  unfold lengthOrder; induction len; intros.
  - inversion H; inversion H1; rewrite H1 in H0; inversion H0.
  - destruct ls; inversion H.
    + inversion H0.
    + destruct ls; simpl; auto. 
      destruct (le_lt_dec 2 (length ls)).
      * specialize (IHlen ls); destruct (split_alt ls); destruct IHlen; simpl.
        simpl in H1; apply le_S_n in H1; split; auto. apply le_Sn_le; auto. 
        split; rewrite <- Nat.succ_lt_mono; auto.
      * inversion l. 
        -- destruct ls; inversion H3; apply length_zero_iff_nil in H4; rewrite H4;
           simpl; auto.
        -- apply le_S_n in H3. inversion H3. 
           apply length_zero_iff_nil in H5; rewrite H5; simpl; auto.
  Defined.
  
  Ltac split_wf := intros ls ?; intros; generalize (@split_wf (length ls) ls);
    destruct (split_alt ls); destruct 1; auto.
  
  Lemma split_wf1 : forall ls, 2 <= length ls -> 
    lengthOrder (fst (split_alt ls)) ls.
  Proof.
  split_wf.
  Defined.
  
  Lemma split_wf2 : forall ls, 2 <= length ls -> 
    lengthOrder (snd (split_alt ls)) ls.
  Proof.
  split_wf.
  Defined.
  
  Fixpoint combine_alt (ls1 ls2 : list T) : list T :=
    match ls1, ls2 with
    | nil, l2 => l2
    | l1, nil => l1
    | h1 :: t1, h2 :: t2 => h1 :: h2 :: combine_alt t1 t2
    end.
  
  Lemma combine_split_alt: forall (l : list T), 
    combine_alt (fst(split_alt l)) (snd(split_alt l)) = l.
  Proof.
  intros; apply split_alt_rect; auto; intros ;simpl; rewrite <- H; rewrite e1; 
  simpl; auto.
  Qed.
  
  Theorem btree_rect : forall (P : list T -> Type),
     P nil ->
    (forall t : T, P (cons t nil)) ->
    (forall l0 : list T, P l0 -> forall l1 : list T, P l1 -> 
     P (combine_alt l0 l1)) ->
    (forall l : list T, P l).
  Proof.
  intros. apply noetherian_rect. intros.
  destruct (le_lt_dec 2 (length x)).
  - rewrite <- combine_split_alt; apply X1; apply X2.
    + apply split_wf1; auto.
    + apply split_wf2; auto.
  - destruct x; auto. simpl in l0; 
    apply le_S_n, le_S_n, Nat.le_0_r,length_zero_iff_nil  in l0; rewrite l0; auto.
  Qed.
