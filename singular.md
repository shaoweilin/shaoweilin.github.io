# Singular Learning

Some personal resources for understanding and applying singular learning theory. For more information, please read the [textbook](http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/singular-learning-theory.html) by Sumio Watanabe.

```{list-table}
* - [Thesis](https://w3id.org/people/shaoweilin/public/swthesis.pdf) 
  - Algebraic methods for evaluating integrals in Bayesian statistics
* - [Preprint](http://arxiv.org/abs/1003.5338) 
  - Asymptotic approximation of marginal likelihood integrals 
* - [Notes](https://w3id.org/people/shaoweilin/public/useful.pdf) 
  - Useful facts about RLCT 
* - [Slides](https://w3id.org/people/shaoweilin/public/chicago.pdf) 
  - What is singular learning theory? 
* - [Slides](https://w3id.org/people/shaoweilin/public/aim.pdf) 
  - Singular Learning Theory: A view from Algebraic Geometry 
* - [Slides](https://w3id.org/people/shaoweilin/public/curves.pdf) 
  - Computing resolutions: how and why 
* - [Slides](https://w3id.org/people/shaoweilin/public/slc.pdf) 
  - Computing integral asymptotics using toric blow-ups of ideals 
* - [Video](http://www.youtube.com/watch?v=NhdtWnieTgI) 
  - Studying model asymptotics with Singular learning theory 
* - [Website](https://w3id.org/people/shaoweilin/public/rlct.html) 
  - Macaulay2 library for computing RLCTs using Newton polyhedra
```

My slides and problem sets on Singular Learning Theory from the [Motivic Invariants and Singularities](http://www3.nd.edu/~cmnd/programs/mis2013/) workshop at the Center for Mathematics at Notre Dame (May 2013).

```{list-table}
* - Statistical Learning Theory
  - [Slides](https://w3id.org/people/shaoweilin/public/cmnd1SLT.pdf)
  - [Problems](https://w3id.org/people/shaoweilin/public/cmnd1prob.pdf)
* - Real Log Canonical Threshold
  - [Slides](https://w3id.org/people/shaoweilin/public/cmnd2RLCT.pdf)
  - [Problems](https://w3id.org/people/shaoweilin/public/cmnd2prob.pdf)
* - Singularities in Graphical Models
  - [Slides](https://w3id.org/people/shaoweilin/public/cmnd3GM.pdf)
  - [Problems](https://w3id.org/people/shaoweilin/public/cmnd3prob.pdf)
* - Blow-ups and Zeta Functions
  - [Notes](https://w3id.org/people/shaoweilin/public/cmnd4notes.pdf)
  - 
```

``````{list-table}
:header-rows: 1
:widths: 20 20 20

* - Syntax
  - Example
  - Result
* - ```md
    | a    | b    |
    | :--- | ---: |
    | c    | d    |
    ```
  - ```md
    |    Training   |   Validation   |
    | :------------ | -------------: |
    |        0      |        5       |
    |     13720     |      2744      |
    ```
  - |    Training   |   Validation   |
    | :------------ | -------------: |
    |        0      |        5       |
    |     13720     |      2744      |
* - ````md
     ```{list-table}
    :header-rows: 1

    * - Col1
      - Col2
    * - Row1 under Col1
      - Row1 under Col2
    * - Row2 under Col1
      - Row2 under Col2
    ```
    ````
  - ````md
     ```{list-table}
    :header-rows: 1
    :name: example-table

    * - Training
      - Validation
    * - 0
      - 5
    * - 13720
      - 2744
    ```
    ````
  - ```{list-table}
    :header-rows: 1
    :name: example-table

    * - Training
      - Validation
    * - 0
      - 5
    * - 13720
      - 2744
    ```
* - ````md
     ```{list-table} title
    :header-rows: 1

    * - Col1
      - Col2
    * - Row1 under Col1
      - Row1 under Col2
    * - Row2 under Col1
      - Row2 under Col2
    ```
    ````
  - ````md
     ```{list-table} Table with a title
    :header-rows: 1

    * - Training
      - Validation
    * - 0
      - 5
    * - 13720
      - 2744
    ```
    ````
  - ```{list-table} Table with a title
    :header-rows: 1

    * - Training
      - Validation
    * - 0
      - 5
    * - 13720
      - 2744
    ```
``````