---
date: 2022-05-28
excerpts: 4
---

# Likelihood, greed and temperature in sequence learning

Imagine we have a model $D(w)$ of a dynamical system with states $s \in S,$ that is parametrized by some weight $w \in W$. Each state $s$ comes with a set $N(s) \subset S$ of neighbors and an associated energy function $E(s'|s,w) \in \mathbb{R}$ that assigns an energy to each neighbor $s' \in N(s)$. 

For simplicity, we assume the following dynamics: when the system is in state $s$, it picks the neighbor $s'$ with the lowest energy $E(s'|s,w)$ and jumps to state $s'$ in the next time step (more to come later about what we mean by *time step*).

Suppose also that we have an observed sequence $\{\hat{s}_t\}$ with each $t\in \{0, 1, \ldots, T\}.$ Roughly, the goal is to find a dynamical system $D(\hat{w})$ that is able to reproduce $\hat{s}_t$. This is the *sequence learning* problem for the model $D(w)$.

More specifically, we want to answer the following questions.
1. Is there a likelihood-based method for sequence learning in discrete time?
2. Is there a likelihood-based method for sequence learning in continuous time?
3. Is there a simple and efficient greedy method for sequence learning?
4. How is the greedy method related to the likelihood method?
5. Are there other dynamical systems more suited for natural language processing or reinforcement learning?

## Likelihood method in discrete time

Given the (conditional) energy function $E(s'|s,w),$ we consider a discrete-time Markov chain with transition probabilities

$$
P(s'|s, w) = \displaystyle \frac{\exp(-\beta E(s'|s,w)) }{Z(s,w)} 
$$

where we have the *partition function*

$$
Z(s,w) = \displaystyle \sum_{s' \in N(s)} \exp(-\beta E(s'|s,w))
$$

and the *inverse temperature* $\beta$.

The traditional maximum likelihood method for sequence learning tells us to maximize the product

$$
L(w) = \prod_{t=0}^{T-1} P(\hat{s}_{t+1}|\hat{s}_t, w).
$$

The negative log-likelihood is

\begin{align*}
\ell(w) &= \displaystyle \sum_{t=0}^{T-1} \log Z(\hat{s}_t,w)+\beta \sum_{t=0}^{T-1} E(\hat{s}_{t+1}|\hat{s}_t, w) \\ 
&= \displaystyle \sum_{t=0}^{T-1} \log \sum_{s' \in N(\hat{s}_t)} e^{-\beta E(s'|\hat{s}_t,w)}+\beta \sum_{t=0}^{T-1} E(\hat{s}_{t+1}|\hat{s}_t, w) 
\end{align*}

The tricky issue here is the intractable logarithmic partition function $\log Z(s,w)$ when we have a large number of neighboring states.

## Likelihood method in continuous time

To overcome the intractable partition function, we could consider a continuous-time Markov chain with off-diagonal transition rates

$$
\Gamma(s'|s,w) = \exp(-\beta E(s'|s,w))
$$

and diagonal transition rates

$$ 
\Gamma(s|s,w) = - \sum_{s'\in N(s)} \Gamma(s'|s,w) = - Z(s,w)
$$

If we assume a small time interval $\delta > 0$ and construct a continuous time series $\hat{x}(t)$ such that 

$$
\hat{x}_{\delta t} =  \hat{s}_{\lfloor t\rfloor} ,
\quad \text{for all }0 \leq t \leq T,
$$

then the likelihood of $\hat{x}$ is

\begin{align*}
L(w) & =\displaystyle \prod_{t=0}^{N-1} 
Z(\hat{s}_{t},w) \exp \big(-\delta Z(\hat{s}_t,w) \big) \, \displaystyle \frac{\exp(-\beta E(s'|s,w))}{Z(\hat{s}_t,w)} \\ 
& =\displaystyle \prod_{t=0}^{N-1} 
  \exp \big(-\delta Z(\hat{s}_t,w) \big) \, \exp(-\beta E(s'|s,w)).
\end{align*}

The negative log-likelihood is

\begin{align*}
\ell(w) &= \displaystyle\delta \sum_{t=0}^{T-1} Z(\hat{s}_t,w) +\beta \sum_{t=0}^{T-1} E(\hat{s}_{t+1}|\hat{s}_t, w)\\ &= \displaystyle\delta\sum_{t=0}^{T-1} \sum_{s' \in N(\hat{s}_t)} e^{-\beta E(s'|\hat{s}_t,w)} +\beta \sum_{t=0}^{T-1} E(\hat{s}_{t+1}|\hat{s}_t, w).
\end{align*}

Again, the intractable partition function appears but without a logarithm, so it is easier to differentiate under the sum over the neighboring states. We can reduce its contribution by either making $\beta$ large or making $\delta$ small. Note that we cannot make the hyperparameter $\delta$ go away  in the objective function just by scaling $\beta.$


## Relative information

The negative log-likelihoods for both the discrete-time and continuous-time models can be interpreted in terms of (conditional) relative information. See [this post](https://shaoweilin.github.io/posts/2020-10-23-machine-learning-with-relative-information/) for more details. These *energy-based* methods can be thought of as *information-based* methods. 

## Greedy method

In the naive greedy method, we simply look for a weight $\hat{w}$ such that the following inequalities are satisfied for all $t\in \{0, \ldots, T-1\}$ and $s' \in N(\hat{s}_t), s' \neq \hat{s}_{t+1}:$

$$
E(s'|\hat{s}_t, w) \leq E(\hat{s}_{t-1}|\hat{s}_t, w).
$$

If such a weight $\hat{w}$ exists, then the dynamical system $D(\hat{w})$ will faithfully reproduce the sequence $\{\hat{s}_t\}.$ We say that $\hat{w}$ is *faithful* to $\{\hat{s}_t\}$.

If such a weight $\hat{w}$ is not faithful to $\{\hat{s}_t\}$, then we need to fall back to the likelihood methods to find a dynamical system $D(\hat{w})$ that best approximates the observed sequence. 

How is the greedy method related to the likelihood methods? 

Given a weight $w \in W$ and state $s_t \in S$ at time $t,$ let $s^*_{t+1} \in N(s_t)$ denote a neighboring state at time $t+1$ that minimizes the energy $E(s^*_{t+1}|s_t,w).$ We consider what happens in the discrete-time and continuous-time models as the inverse temperature $\beta$ tends to infinity.


## Temperature in discrete time

In the discrete-time model, the transition probability of $s^*_{t+1}$ conditioned on $s_t$ will tend to $1,$ while the transition probabilities of other neighbors of $s_t$ will tend to $0$.

If $w$ is faithful the the observed sequence $\{\hat{s}_t\},$ then the likelihood $L(w)$ will tend to $1$ as $\beta$ goes to infinity. If $w$ is not faithful, then $L(w)$ will tend to $0.$ If the region of faithful weights is non-empty, then as the maximum likelihood estimate $\hat{w}$ will lie in this region for sufficiently large $\beta.$

The log partition function 

$$
\log Z(s_t, w) = \log \sum_{s'\in N(s_t)}  e^{-\beta E(s'|s_t,w)}
$$

tends to $-\beta E(s^*_{t+1}|s_t,w)$ as $\beta$ goes to infinity (assuming there is a unique minimal energy state $\hat{s}^*_{t+1}.$) Consequently, the negative log-likelihood $\ell(w)$ tends to

$$
\beta \sum_{t=0}^{T-1} \Big( E(\hat{s}_{t+1}|\hat{s}_t, w) - E(\hat{s}^*_{t+1}|\hat{s}_t, w) \Big),
$$

which is $0$ if $w$ is faithful to the sequence but positive if $w$ is not faithful.

In the large $\beta$ limit, we generate an optimal sequence or path by running the dynamical system described at the beginning of this post - namely, at each state $s$, we pick the next state $s'$ that minimizes the energy $E(s'|s,w),$ and we repeat this for $T$ time steps. We say that the optimal path is generated by _stepwise energy minimization_.

## Temperature in continuous time

In the continuous-time model, as $\beta$ goes to infinity, the transition probability of $s^*_{t+1}$ given $s_t$ tends to $1$ while the transition probability of other neighbors $s'_{t+1}$ given $s_t$ tends to $0,$ just like in the discrete-time case. As seen previously, the negative log probability grows like $\beta \big( E(\hat{s}_{t+1}|\hat{s}_t, w) - E(\hat{s}^*_{t+1}|\hat{s}_t, w) \big) $ for large $\beta.$

The holding rates $Z(s_t, w)$ will, however, tend to $0.$ This means that the time held in state $s_t$ will increase to infinity as $\beta$ increases. 

In fact, for large $\beta,$ the negative log density grows like $\beta E(\hat{s}^*_{t+1}|\hat{s}_t, w),$ the transition energy that is mimimal over all $\beta E(s'|\hat{s}_t, w)$ for $s' \in N(\hat{s}_t).$ This means that for lower minimal transition energies $E(\hat{s}^*_{t+1}|\hat{s}_t, w),$ the holding time period in state $\hat{s}_t$ is shorter.

Consequently, the negative log-likelihood $\ell(w)$ tends to

$$
\beta \sum_{t=0}^{T-1} E(\hat{s}_{t+1}|\hat{s}_t, w),
$$ 

the sum of the transition energies. The objective here is different from that in the discrete-time model which 
subtracts the minimal transition energy from each time step.

Suppose that we have a time interval of length $T$ and the initial state is $s_0$. By the stochastic summation algorithm of Gillespie {cite}`weber2017master`, the negative log density of observing a path with states $s_0, \ldots, s_n$ and holding periods $\delta_0, \ldots, \delta_n$ satisfying $\delta_0 + \cdots +\delta_n = T,$ is given by

$$
\sum_{t=0}^n \delta_t Z(s_t,w) + \beta \sum_{t=0}^{n-1} E(s_{t+1}|s_t,w).
$$

For large $\beta$, the latter summand dominates the log density, and the optimal paths are determined by *pathwise energy minimization*. This pathwise minimization is analogous to the principle of least action in classical or quantum physics.


## Natural language processing

The difference between stepwise energy minimization and pathwise energy minimization is easily seen in natural language processing. 

In his blog {cite}`blog2015unreasonable` about the unreasonable effectiveness of recurrent neural networks (RNN), Karpathy trained a small character-level LSTM (long short-term memory) network on the writings of Paul Graham. 

He found that for large inverse temperatures $\beta,$ the network generated the following stepwise greedy sequence of words:

> is that they were all the same thing that was a startup is that they were all the same thing that was a startup is that they were all the same thing that was a startup is that they were all the same...

Granted that at lower inverse temperatures the network produced sentences with greater variation and that the network is technically not a Markov chain, the above example shows us what could go wrong with extreme stepwise greedy sentence generation.

Perhaps our minds behave more like networks that churn out sentences with small pathwise energies. 

Why would nature choose such a strategy over the simpler stepwise greedy approach, in natural language processing and, more generally, in reinforcement learning?

## Reinforcement learning

One possible answer is that many natural problems are best solved by *trial and error*. One has to try many *likely* solutions in a short period of time, as opposed to getting the correct solution but only after a long deliberation. 

In a continuous-time Markov chain, a path with optimal stepwise energy may suffer from large minimal transition energies $E(\hat{s}^*_{t+1}|\hat{s}_t, w)$ which lead to long holding times. 

On the other hand, a path with optimal pathwise energy may have non-optimal transitions, but the system can generate many such paths quickly because of their short holding times. One of these paths could be the key to the problem at hand.

Even in problems where small mistakes are catastrophic, the ability to brainstorm and quickly generate many possible solutions is critical. We can reason about the correctness of the solutions before testing them out. We can choose the best potential solution with the smallest risk.


## Minimum energy flow

We end with an interesting connection between the continuous-time likelihood method for sequence learning and the minimum energy flow (MEF) method for training Hopfield networks efficiently {cite}`hillar2021hidden`.

Recall that the negative log likelihood is

$$
\ell(w) = \displaystyle\delta\sum_{t=0}^{T-1} \sum_{s' \in N(\hat{s}_t)} e^{-\beta E(s'|\hat{s}_t,w)} +\beta \sum_{t=0}^{T-1} E(\hat{s}_{t+1}|\hat{s}_t, w).
$$

On the other hand, the energy flow function is

$$
\text{EF}(w) = \displaystyle\sum_{t=0}^{T-1} \sum_{s' \in N(\hat{s}_t)} e^{- E(s'|\hat{s}_t,w)},
$$

which is proportional to the limit of $\ell(w)$ at $\beta =1$ as $\delta$ tends to infinity.

This limit makes sense because energy flow is used in the paper for training Hopfield networks with attractor dynamics. The observed data $\{\hat{s}_t\}$ is not actually a sequence, but a list of attractors. Each  attractor may be thought of as a one-state sequence with infinite holding time $\delta$.

The negative log likelihood $\ell(w)$ provides a family of objective functions with hyperparameters $\delta$ and $\beta$ that could be used in sequence learning.

## References

```{bibliography}
:filter: docname in docnames
```