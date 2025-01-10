---
date: 2024-10-01
excerpts: 2
---

# Program Synthesis

Here is a list of prior work I did with my collaborators and Ph.D. students on dependent type theory and program synthesis.

## AI-Assisted Program Synthesis

2024: Shaowei Lin. [Safety by shared synthesis](https://shaoweilin.github.io/posts/2024-09-24-safety-by-shared-synthesis/).

2024: Shaowei Lin. [Formal AI-assisted code specification and synthesis: concrete steps towards safe sociotechnical systems](https://shaoweilin.github.io/posts/2024-05-22-formal-ai-assisted-code-specification-and-synthesis-concrete-steps-towards-safe-sociotechnical-systems/).

2024: Shaowei Lin. [AI-assisted coding: correct by construction, not by generation](https://shaoweilin.github.io/posts/2024-05-08-ai-assisted-coding-correct-by-construction-not-by-generation/).

2021: Shaowei Lin. [Proofs as programs: challenges and strategies for program synthesis](https://shaoweilin.github.io/posts/2021-04-22-proofs-as-programs-challenges-and-strategies-for-program-synthesis/).

2018: Shaowei Lin. [Machine reasoning and deep spiking networks](https://shaoweilin.github.io/public/20180526-aisg.pdf).

2017: Shaowei Lin. [Artificial general intelligence for the internet of things](https://shaoweilin.github.io/posts/2017-05-08-artificial-general-intelligence-for-the-internet-of-things/).


## Collaborative Theorem Proving

2021: Lim, Jin Xing, Barnabé Monnot, Shaowei Lin, and Georgios Piliouras. *[A blockchain-based approach for collaborative formalization of mathematics and programs.](https://arxiv.org/pdf/2111.10824)* In 2021 IEEE International Conference on Blockchain (Blockchain), pp. 321-326. IEEE, 2021. [[YouTube]](https://youtu.be/5q2TK4RRBeg)

2021: Lim, Jin Xing, Barnabé Monnot, Georgios Piliouras, and Shaowei Lin. *[(Auto)Complete this Proof: Decentralized Proof Generation via Smart Contracts](https://aitp-conference.org/2021/abstract/paper_7.pdf)* In 6th Conference on Artificial Intelligence and Theorem Proving (AITP 2021). [[YouTube]](https://youtu.be/aBOYzx118WA)

2022: [Jin Xing LIM](https://jinxinglim.github.io/) ([Ph.D. thesis](https://sutd.primo.exlibrisgroup.com/permalink/65SUTD_INST/19hmrhl/alma999649164802406); with Georgios Piliouras)

> **Incentivized mechanism design for collaborative proofs and programs through blockchain and theorem provers**
>
> Scientific research, and particularly research in mathematics, is arguably one of the crowning achievements of our collective human intellect. Its creation increasingly requires collaboration between multiple researchers with different and sometimes complementary backgrounds. On the other hand, its verification requires a careful matching between the expertise of reviewers and authors. Unfortunately, errors do occur and sometimes are only corrected many years after they appear in print, if at all. Nevertheless, at least when it comes to mathematical research, computer-verified formalized proofs are possible as a final arbiter of mathematical truth. Formalization of mathematics is the process of digitizing mathematical knowledge, which allows for formal proof verification of not only mathematical results, but computer programs as well. However, they are not easy to produce even for relatively simple statements. Hence, such approaches typically lack far behind the current research frontier. Aimed at addressing the above-mentioned issues, the three main contributions of this thesis are as follows:
>
> 1. To promote collaboration and expert reviewing within mathematical research, we propose a novel blockchain-based system that allows mathematicians to collaborate by sharing their partial pen-and-paper proofs (proofs written in natural language), and for reviewers to review thereafter.
>
> 2. To close the gap between the set of formalized and unformalized (i.e., pen-and-paper) mathematical knowledge, we propose a blockchain-based system that allows mathematicians, computer scientists and even automated artificial intelligence (AI) tools to collaborate by sharing their partial formalized proofs (proofs in program code).
>
>3. To ensure the quality and correctness of each partial proof, we propose a proof of concept of incentive mechanisms that incentivize mathematicians to share their partial results immediately and expert reviewers to review.

## Knowledge Graph Reasoning

2020: Lai, Zhangsheng, Aik Beng Ng, Liang Ze Wong, Simon See, and Shaowei Lin. *[Dependently typed knowledge graphs.](https://arxiv.org/pdf/2003.03785)* arXiv preprint arXiv:2003.03785 (2020).

2020: [Zhangsheng LAI](https://zunction.github.io/) ([PhD. thesis](https://sutd.primo.exlibrisgroup.com/permalink/65SUTD_INST/19hmrhl/alma999582764802406); with Simon See)

> **Dependent types, canonical structures and reflection strategies for assisted reasoning and computation on knowledge graphs**
>
>The sustenance of reasoning is knowledge as it provides the environment for the
activity of reasoning to derive new information. While a lack of knowledge does not
fairly represent the strengths of reasoning, an overly abundance of knowledge leads
to information overload due to the limitations of the biological brain. A promising
way to overcome this problem is the use of machines with reasoning capabilities to
reason over large information stores in the form of knowledge graphs. This dissertation
demonstrates the suitability of dependent type theory and assisted theorem proving
technologies to develop the important reasoning and explanability aspects of querying
a knowledge graph.
>
>The first goal of the research described here is to show possibility of the queries-
as-types interpretation. We demonstrate this fact by using record types to form queries
for dependently typed knowledge graphs – knowledge graph represented by inductive
types. By using powerful machine-tactics, the answer and corresponding proof for the
query can be constructed directly or by traversing through an intermediate node; we
also show how automation of such queries is possible with custom tactics.
>
>The second contribution of our work generalizes the result of the first by building
hierarchical structures for knowledge graphs. To do so, knowledge graphs are repre-
sented as a universe of nodes that prevents fragile theorems from being proved, and are
also designed as parameters of our theories to keep the reasoning component separate
from the knowledge graphs. These hierarchical structures provide the subtyping and
querying in knowledge graphs, and are instantiated by predicates which hold the intent
of the type to be expressed. Furthermore, we see that predicates with enumeration are
compatible for doing queries, relying on machine-tactics provide assisted reasoning;
users are only tasked with the responsibility to make critical decisions that eventually
determines the final answer.
>
>The final contribution extends the second work with a reflection mechanism for in-
tent and enumeration representations. Our enumeration reflection, motivated by the
boolean reflection in Mathematical Components, allows a query to be expressed as ei-
ther an intent type or an enumeration function. The intent type verifies the answer
to the query and the enumeration provides the function used for computing the com-
plete sequence of answers to the query. In addition to showing its usage to obtaining
the terms of a subtype, we extend its usage to composite queries which returns the
concatenation of the answers obtained by traversing different reasoning paths.




## Human-AI Collaboration

2021: Ng, Aik Beng, Simon See, Zhangsheng Lai, and Shaowei Lin. *[Group-Assign: Type Theoretic Framework for Human AI Orchestration.](https://www.intechopen.com/chapters/75694)* Virtual Assistant (2021): 45.

* Developed the Group-Assign strategy for human-AI collaboration using ideas from dependent type theory. 
* Our paper won second place for the Best Blue Sky Idea award at HCOMP 2020.

2020: [Aik Beng NG](https://www.linkedin.com/in/aikbengng/) ([Ph.D. thesis](https://sutd.primo.exlibrisgroup.com/permalink/65SUTD_INST/19hmrhl/alma999583564502406); with Simon See)

> **Human-AI Collaboration: Type theoretic
composition, conglomeration and
communication of intents for an
AI-augmented knowledge workforce**
>
> There has never been a time in history where technology has been so pervasive. Today, Artificial Intelligence (AI) is transforming the industry. As a long-term research community goal, achieving general human-like intelligence is one of the highest aspirations for AI but is still out of reach. Here, we explore AI augmentation of the knowledge workforce as our vision, and type theory as a foundational means to make it a reality.
>
>In our first work, Group-Assign: Type Theoretic Framework for Human AI Orchestration, we develop a framework and an associated implementation, built upon a foundational set of type theoretic framework axioms, and Group and Assign as base methodologies for data handling and assignment of intents to associated implementations. With the correspondence of intents as types, the framework has type theoretic expressiveness, which enables composition and conglomeration of intents to construct work plans. The resulting work plan is imbued with type theoretic properties such as computability, constructivism and explanability. With this, work plans are coherent and transparent while allowing for distributed contribution of intents from diverse users, under a unified view of human and AI operations grounded in type theory.
>
>In our second work, Exemplar-based Grouping for Information Processing, we define an interactive framework between the human and AI such that, based on the human’s intuition, the AI will group contextually relevant information even of heterogeneous nature, and thereby lets the user act on related groups of information in a batched manner. At the human-individual level, this helps reduce task-switching overheads and bridges the gap from the conundrum of multi-tasking as a means to do more, only to be counterproductive. This proposed work is also technically synergistic with Group-Assign: Type Theoretic Framework for Human AI Orchestration, as a possible implementation of the Group method.
> 
>In our third work, Type theoretic-Resource Description Framework (TT-RDF), we develop a set of TT-RDF vocabulary and a parser that translates between type theoretic expressions and RDF-friendly format. TT-RDF bridges linked data and type theory to enable the sharing and computation of type theoretic objects across a network. While Group-Assign: Type Theoretic Framework for Human AI Orchestration provides the means to compose and conglomerate intents into type theoretic work plans, TT-RDF contributes the synergistic ability to encapsulate these work plans into well understood linked data objects, that can be communicated across vast networks.
>
>Altogether, our proposed works aim to increase human AI synergy to navigate an increasingly connected world that is rich and overwhelming with information.




