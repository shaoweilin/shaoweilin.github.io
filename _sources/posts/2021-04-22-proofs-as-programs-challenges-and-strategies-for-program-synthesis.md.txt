---
date: 2021-04-22
---

# Proofs as programs - challenges and strategies for program synthesis

## Abstract
The Curry-Howard correspondence between proofs and programs suggests that we can exploit proof assistants for writing software. I will discuss the challenges behind a naïve execution of this idea, and some preliminary strategies for overcoming them. As an example, we will organize higher-order information in knowledge graphs using dependent type theory, and automate the answering of queries using a proof assistant. In another example, we will explore how decentralized proof assistants can enable mathematicians or programmers to work collaboratively on a theorem or application. If time permits, I will outline connections to canonical structures, reflection (ssreflect), transport, unification and universe management.

## Details
[Topos Institute Colloquium](https://topos.site/topos-colloquium/)

[Slides](https://w3id.org/people/shaoweilin/public/20210422-topos.pdf)

[YouTube](https://www.youtube.com/watch?v=cEdoG9h-pYg)

## Notes
In this talk, I spell out the mathematical foundations for program synthesis in terms of dependent type theory, based on the Curry-Howard correspondence between proofs and programs. This introductory material is described using more layman terms in an earlier [talk](2017-05-08-artificial-general-intelligence-for-the-internet-of-things/).

I discuss joint work with my PhD student Zhangsheng Lai and collaborator Liang-Ze Wong on knowledge graphs.

I also discuss joint work with my PhD student Jin-Xing Lim and collaborators Georgios Piliouras and Barnabe Monnot on collaborative theorem proving via blockchain. 

One interesting example involves synthesizing different sorting algorithms, such as insertion sort, merge sort and quick sort. I will relate it to the problem of transport between equivalent types (e.g. binary and unary numbers), or from a subtype to its supertype (e.g. lists to binary trees)