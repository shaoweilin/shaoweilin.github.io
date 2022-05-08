---
date: 2021-06-05
excerpts: 2
---

# Spiking neural networks

In this post, we study a class of spiking network models based on continuous-time [Markov](2020-10-14-path-integrals-and-continuous-time-markov-chains/#what-is-a-continuous-time-markov-chain) chains with [mutable](2021-03-22-relative-inference-with-mutable-processes/#what-is-a-mutable-process)  variables. 

Using a [relative inference](2021-06-01-convergence-of-biased-stochastic-approximation/) recipe for online learning, we derive local Hebbian learning rules for the spiking network which are provably convergent to local minima of the relative information objective.

This post is a continuation from our [series](2020-08-28-motivic-information-path-integrals-and-spiking-networks/) on spiking networks, path integrals and motivic information.

For background material in statistical learning, relative information, relative inference, process learning, mutable processes and biased stochastic approximation, you may follow the recommended sequence of posts under _Spiking Networks_ in this [outline](2020-08-28-motivic-information-path-integrals-and-spiking-networks/).


### What are the states and parameters of the spiking network model?

Let $\mathcal{V}_X$ be a finite set representing the collection of random variables which capture the state of the environment or universe. For [simplicity](2021-03-23-biased-stochastic-approximation-with-mutable-processes/#what-do-we-assume-about-the-true-distribution-the-model-and-the-learning-objective), we assume that each of these variables have the same state space as the neurons in our model, so we will call them _environment neurons_ for convenience. We also assume that these neurons contain enough information about the environment such that its true distribution is Markov. A subset of these neurons will be observed for inference and learning; the other neurons could remain unobserved or inaccessible.

Let $\mathcal{V}_Z$ be a finite set representing the collection of [mutable](2021-03-22-relative-inference-with-mutable-processes/#what-is-a-mutable-process) neurons which assist the spiking network computationally in learning an approximation of the true distribution. The states of these mutable neurons represent samples of beliefs that the spiking network has about the current state of the environment. Together, they act as the memory of the spiking network, so we will call them _memory neurons_.

At each time $t \geq 0$ and for each neuron $i \in \mathcal{V}:=\mathcal{V}_Z \cup \mathcal{V}_X,$ we have a signed state

$$S_{it} \in \mathbb{S} := \{-1,+1\}$$ 

representing the state of the voltage-gated sodium channels of the neuron, namely, $+1$ if the channels are _closed_, and $-1$ if the channels are _open_ or _inactive_. A transition from $+1$ to $-1$ is called _spiking_ while a transition from $-1$ to $+1$ is called _recovery_.

If the sodium channels are closed, the neuron is in a resting state and ready to spike at any time. When the membrane potential exceeds a certain threshold, the sodium channels open to allow sodium ions into the neuron, causing the neuron to spike. After spiking, the sodium channels are inactivated, the neuron enters a refractory state, the sodium ions leave the cell, and the membrane potential goes back to its resting level. When the sodium channels are open or inactive, the neuron is unable to spike. Finally, the channels return to their closed state and the process starts again.

The above process is often modelled deterministically, where the voltage-dependent channels open when the membrane potential crosses the threshold and the neuron then enters the refractory period for a fixed duration of time. Following {cite}`pfister2006optimal` and {cite}`rezende2014stochastic`, we will instead model the neuron spikes and channel recoveries stochastically, because of intrinsic noise in the transmission of input spikes from other neurons as well as random impulses from neurons which are not explicitly modeled. 

At each time $t \geq 0$ and for each pair $i,j \in \mathcal{V}$, we have a counting state

$$R_{ijt} \in \mathbb{N} := \{0,1,2,\ldots\}$$

representing the number of spikes from possibly some presynaptic neuron $i$ to some postsynaptic neuron $j$ since the last spike by the postsynaptic neuron $j.$ We consider these synaptic counts as belonging to the postsynaptic neuron $j,$ because a spike in neuron $j$ resets them and eventually the counts affect its membrane potential. For all $j \in \mathcal{V},$ we fix $R_{jjt} = 1.$ This count will later help to account for changes in the membrane potential which are not from other neurons.


Collectively, we denote 

$$S_t := (S_{it} : i \in \mathcal{V}),$$

$$R_t := (R_{ijt} : i,j \in \mathcal{V}),$$

$$X_t := (S_{jt},R_{ijt} : j \in \mathcal{V}_X, i \in \mathcal{V}),$$

$$Z_t := (S_{jt},R_{ijt} : j \in \mathcal{V}_Z, i \in \mathcal{V}).$$

Let $\mathcal{E}_Q$ be a finite set representing the collection of edges or connections from $\mathcal{V}$ to $\mathcal{V}_Z$ that participate in the discriminative model $Q$ of the spiking network. Intuitively, at time $T,$ this discriminative model computes from $(Z_T,X_T)$ the distribution or belief of $Z_{T+\delta}$ for infinitesimal $\delta > 0.$ The model plays an important role in inference. Note that there are no edges to $\mathcal{V}_X$ in this set of connections.

Let $\mathcal{E}_P$ be a finite set representing the collection of edges or connections from $\mathcal{V}$ to $\mathcal{V}$ that participate in the generative model $P$ of the spiking network. Intuitively, at time $T,$ this generative model estimates from $(Z_T,X_T)$ the distribution of $(Z_t,X_t)$ for all $t > T.$ The model plays an important role in prediction. 

For each edge $e \in \mathcal{E}_Q \cup \mathcal{E}_P,$ we denote its source by $e(0)$ and its target by $e(1).$ We assume that $\mathcal{E}_Q$ and $\mathcal{E}_P$ contain _all_ self-loops, i.e. edges $e$ with $e(0)=e(1).$ For simplicity, we do not allow multi-edges, i.e. two distinct edges with the same source and target.

Each edge $e \in \mathcal{E}_Q$ of the discriminative model has a weight $w^{(Q)}_{e} \in \mathbb{R}$ representing the synaptic strength of the corresponding connection. For self-loops $e,$ the bias $w^{(Q)}_e$ represent changes to the membrane potential that is not from other neurons, e.g. decay. As for the generative model $P,$ the weights $w^{(P)}_e, e \in \mathcal{E}_P,$ are similarly defined. 

Collectively, we denote

$$w^{(Q)} := (w^{(Q)}_{e}: e \in \mathcal{E}_Q),$$

$$w^{(P)} := (w^{(P)}_{e} : e \in \mathcal{E}_P).$$

We now describe the transitions allowed in our spiking network model. We require the states $(R_t,S_t)$ to be piecewise constant in time, with only finitely many transitions in any finite time interval. If we have a transition from $(R,S)$ to $(R',S'),$ and $S$ and $S'$ differ at exactly $k$ different neurons, then we say the transition is _$k$-hop_. For the environment process $\{X_t\},$ we will allow multi-hop transitions, even if such coupling of neurons is biologically rare or impossible.

### What are the dynamics of the generative model?

Suppose we fix the parameters $w^{(P)}$ and drop the $(P)$ annotation for convenience. We now define the generative model distribution $P_w$ as a $w$-controlled continuous-time [Markov](2020-10-14-path-integrals-and-continuous-time-markov-chains/#what-is-a-continuous-time-markov-chain) chain with rate kernel $\Gamma$ and a countable state space

$$(R_t, S_t) \quad \in \quad \mathbb{N}^{\lvert \mathcal{V} \rvert \times \lvert \mathcal{V} \rvert} \times \mathbb{S}^{\lvert \mathcal{V} \rvert}. $$

Before defining the rate kernel $\Gamma$, let us describe informally the dynamics of the spiking network. Suppose that the network is in state $(R_t,S_t)$ at time $t.$ We want the membrane potential of a neuron $j \in \mathcal{V}$  to be

$$ \displaystyle 
U_{jt} = \sum_{e \in \mathcal{E}_P} \mathbb{I}(e(1)=j)\, R_{et}\, w_e $$

where $R_{et}$ is notation for $R_{e(0)e(1)t}$ and $\mathbb{I}$ the indicator function. Note that only one of the self-loop biases $w_e, e(1)=j,$ appears in this sum with $R_{et}=1$ by default. The holding time until the next transition (spiking or recovery) is exponentially distributed with rate

$$ \displaystyle
\rho_{jt} = \rho_{S_{jt}} \exp \left( \, \beta_{S_{jt}} \, S_{jt} \, U_{jt} \, \right) $$

where the hyperparameters $\rho_+, \rho_- > 0$ and $\beta_+,\beta_- > 0$ are the rate constants and the inverse temperatures for the resting and refractory states respectively. The transition events for different neurons are independent of each other. Note that the transition rate depends on whether the neuron is in resting state $(S_{jt}=+1)$ or refractory state $(S_{jt}=-1).$ We will explain in the next section why the recovery rate varies inversely with the membrane potential for the refractory state.

If neuron $j$ is the first neuron to transition at time $t'$ and $S_{jt} = +1,$ the neuron spikes and sends an impulse to its downstream neurons. We flip its sign $S_{jt'}=-1,$ and reset the upstream counts

$$ \displaystyle
R_{ijt'} = 0 \quad \text{for all }i \in \mathcal{V}.$$

We also increment the downstream counts

$$ \displaystyle
R_{jkt'} = R_{jkt}+1 \quad \text{for all }k \in \mathcal{V}.$$

By default, $R_{jjt'}=1.$ On the other hand, if neuron $i$ is the first neuron to transition at time $t'$ but $S_{it} = -1,$ then no impulse is sent to the downstream neurons. We flip the sign of the neuron so $S_{it'}=+1$ but keep the counts $R_{t'} = R_t$ unchanged.

So far we have described the different ways a state $Y = (R_t,S_t)$ can transition to the next state $Y' = (R_{t'},S_{t'})$ which involves the spiking or recovery of some neuron $j$ with rate $\rho_{jt}.$ We record this formally as the transition rate

$$\displaystyle
\Gamma_{YY'} = \rho_{jt}.$$

It follows that the holding time to the first transition by _any_ neuron is exponentially distributed with rate

$$\displaystyle
\rho_{*t} = \sum_{j \in \mathcal{V}} \rho_{jt}$$

and this is formally recorded in the transition rate 

$$\displaystyle
\Gamma_{YY} = -\rho_{*t}.$$

The probability that the first transition occurs at neuron $j$ is then given by

$$\displaystyle
\mathbb{P}(j | Y) = \frac{\rho_{jt}}{\rho_{*t}}.$$

As mentioned previously, models typically represent the spiking of the neurons as a deterministic process, with the neurons firing when the membrane potential exceeds a fixed threshold. In our model, the membrane potentials $U_{jt}$ should be interpreted as the mean potential of neuron $j$ conditioned on the history of incoming spikes and the strength of synaptic connections.

### Why does the recovery rate vary inversely with the membrane potential in the refractory state?

We give three distinct reasons for introducing this inverse relationship in our model.

The first reason is statistical. Suppose the goal is to model the timings of certain point events. Let us represent each point event $j$ with a neuron $j,$ with the spiking of the neuron indicating the beginning of a time interval during which we have confidence event $j$ occurred, and the recovery of the neuron indicating the end of that interval. 

Suppose we are interested in information or evidence that event $j$ has not occurred yet but will be occurring soon, and let us represent the amount of evidence with the membrane potential of neuron $j.$ If we know that some event $i$ is a strong predictor that event $j$ did not happen before but will happen soon after, and if event $i$ is represented by neuron $i,$ we may instruct the membrane potential of neuron $j$ to increase by some fixed weight $w$ with every spike from neuron $i.$ 

Neuron $j$ may now use the evidence indicated by its membrane potential to decide when to spike and when to recover. If the neuron is in the resting state and the membrane potential is high, the neuron will have strong reasons to believe that event $j$ has not yet happened but will happen soon. It should increase its transition rate so that the spike (indicating the start of the confidence time interval) occurs sooner.

After the neuron spikes, the information in the membrane potential is deemed to be used in producing that estimate of the start of the confidence interval, so the neuron resets its potential. Now the neuron is in the refractory state. If the membrane potential continues to increase due to incoming spikes, the neuron will have strong reasons to believe that it was too early with its estimate of the start of the confidence time interval. It should try to drag out the length of the interval by reducing its transition rate, so that the interval has a higher chance of capturing the actual timing of event $j.$ Therefore the recovery rate varies inversely with the membrane potential.

The second reason is direct biological evidence from sodium channel studies. Early studies {cite}`hodgkin1952quantitative` described the time to recovery of the channels from inactivation as exponentially distributed  with faster rates at more negative voltages. More recent studies {cite}`kuo1994na+` proposed mechanisms to explain this inverse relationship.

The third reason is indirect biological evidence from Spike Timing Dependent Plasticity (STDP). When the presynaptic neuron spikes _after_ the postsynaptic neuron, there is a weakening in the synaptic strength between the neurons. Moreover, as the time interval between the presynaptic and postsynaptic spikes become shorter, the amount of weakening becomes exponentially greater. If STDP is the neural network's algorithm for reinforcement learning, then this weakening must be reinforcing the effect that the synaptic weight has on reducing the recovery time. More precisely, as the interspike timings become shorter, the need to decrease the weights become greater, so decreasing the weights probably has the effect of further reducing the recovery time. In other words, decreasing the membrane potential increases the recovery rate. 

### How does our spiking network model compare to others in the literature?

Compared to the spiking network model of {cite}`rezende2014stochastic`, our model is different in the following ways.

1. Their model adds an exponential filter to the adaptation potentials (decrease in potential after a neuron spikes) and evoked potentials (incoming potentials from neighboring neurons). We do not have this exponential filter explicitly. Instead, the exponential filter will turn out to be a consequence of the learning algorithm.

2. Their model does not model the refractory period of the neuron. We model the refractory period as following an exponential distribution whose rate is inversely related to the membrane potential, i.e. the higher the membrane potential, the slower the recovery of the neuron after spiking.

3. In their model, the membrane potential at time $t$ depends on the full history of spikes from time $0$ to time $t$ so it is not Markov. In our Markov model, the membrane potential $U_{jt}$ of neuron $j$ at time $t$ depends on the incoming synaptic spike counts $R_{ijt}$ which reset with every outgoing spike. These spike counts can be implemented biologically by neuromodulator concentrations that are local to each synapse. Hence, the membrane potential only depends on incoming spikes since the last outgoing spike.

4. Because of the inverse relationship between the membrane potential and refractory period, we will be able to prove directly that our learning updates satisfy the full Spike Timing Dependent Plasticity (STDP). In {cite}`pfister2006optimal`, the learning updates satisfy only a simplified kind of STDP unless some form of out-of-model regularization is added to the objective function.

### What are the dynamics of the discriminative model?

For the discriminative model $Q_w$ with parameters $w = w^{(Q)},$ we start by characterizing the true distribution $Q_*(X).$ Because $\{X_t\}$ is a continuous-time Markov chain by assumption, it has a rate kernel $\Gamma^{(X)}.$ Recall that we allow multi-hop transitions in $\{X_t\}$.

Intuitively, the dynamics of the discriminative process is very similar to that of the generative process, except that the environment states $X_t$ cannot be controlled. For each spike in an environment or memory neuron, we update the synaptic counts in the possibly outgoing edges of the neuron. For each recovery in a memory neuron, we reset the synaptic counts in the possibly incoming edges of the neuron. 

We now define the rate kernel $\Gamma$ of the full process $\{Z_t,X_t\}$ in terms of $\Gamma_X$ and model parameters $w.$ We consider two kinds of transitions: environment transitions where the signed states of $X_t$ changes but not that of $Z_t,$ and memory transitions where the signed states of $Z_t$ changes but not that of $X_t.$

Suppose we have an environment transition from some $(R_X,S_X)$ to $(R'_X,S'_X)$ with 

$$\Gamma^{(X)}_{(R_X,S_X)(R'_X,S'_X)} > 0.$$

Let $\Delta R$ be the number of spiking environment neurons, i.e. neurons $i \in \mathcal{V}_X$ with 

$$(S_X)_i = +1, \quad (S'_X)_i = -1.$$

For all memory states $(R_Z,S_Z),$ let $R' = (R'_Z,R'_X)$ where

$$(R'_Z)_{ij} = (R'_Z)_{ij} + \Delta R$$

for all $j \in \mathcal{V}_Z.$ Let $R = (R_Z,R_X),$ $S = (S_Z,S_X)$ and $S' = (S_Z,S'_X).$ We then define

$$\Gamma_{(R,S)(R',S')} = \Gamma^{(X)}_{(R_X,S_X)(R'_X,S'_X)}.$$

As for the memory transitions, for every full state $(R,S)$ and every neuron $j \in \mathcal{V}_Z$ with $S_j = +1,$ we let

$$U_j = \sum_{e \in \mathcal{E}_P} \mathbb{I}(e(1)=j) \, R_{e} \, w_e$$

$$\rho_j = \rho_{S_j} \exp \left( \, \beta_{S_j} \, S_j \, U_j \, \right)$$

$$S'_j = -1$$

$$R'_{ij} = 0 \quad \text{for all }i \in \mathcal{V}$$

$$R'_{jk} = R_{jk} + 1 \quad \text{for all }k \in \mathcal{V}_Z$$

where $R_e$ is notation for $R_{e(0)e(1)}.$ By default, $R'_{jj}=1.$ All other entries of $R',S'$ not stated above are unchanged from $R,S$. We then define

$$\Gamma_{(R,S)(R',S')} = \rho_j.$$

Similarly, for every full state $(R,S)$ and every neuron $j \in \mathcal{V}_Z$ with $S_j = -1,$ we let

$$U_j =  \sum_{e \in \mathcal{E}_P} \mathbb{I}(e(1)=j) \, R_{e} \, w_e$$

$$\rho_j = \rho_{S_j} \exp \left( \,\beta_{S_j}\, S_j \, U_j \, \right)$$

$$S'_j = +1,$$

and all other entries of $R',S'$ are unchanged from $R,S.$ We then also define

$$\Gamma_{(R,S)(R',S')} = \rho_j.$$

As for the diagonal rates, we have

$$\Gamma_{(R,S)(R,S)} = \Gamma_{(R_X,S_X)(R_X,S_X)} - \rho_{*t},$$

$$\rho_{*} = \sum_{j \in \mathcal{V}_Z} \rho_j.$$

All other transition rates are defined to be zero.

### How do we simulate the continuous-time spiking neural networks on a discrete-time digital computer?

We have presented the discriminative model $Q$ and generative model $P$ as parametric spiking neural networks in continuous time. We have a relative inference [recipe](2021-06-01-convergence-of-biased-stochastic-approximation/) for online learning that updates the parameters in continuous time. 

For efficient simulation on a discrete-time digital computer, we could consider for each continuous-time Markov chain (CTMC) its [embedded](https://en.wikipedia.org/wiki/Markov_chain#Embedded_Markov_chain) discrete-time Markov chain (DTMC) which records the transitions and the holding times between transitions at each integer time step. However, the above recipe for deriving learning algorithms will not apply. This is because the resulting pair $(\hat{Z}_n,\hat{X}_n)$ of variables is always exactly one environment transition or one memory transition from $(\hat{Z}_{n-1},\hat{X}_{n-1}),$ so their states are coupled. This means that $\hat{Z}_n$ and $\hat{X}_n$ are not independent given their past in the DTMC, a condition that is needed for the recipe to work.

Alternatively, we could use the [$\delta$-skeletions](https://en.wikipedia.org/wiki/Markov_chain#Embedded_Markov_chain) of the CTMCs, where we impose a fixed wait time $\delta$ between observations. The issue here is that as $\delta$ tends to zero, some of the probabilistic integrals diverge and the original CTMC is not the limit of the $\delta$-skeletons. 

We now describe a slight variation of the $\delta$-skeleton method whose limiting behavior is the same as the original CTMC. 

We start with the [uniformization](https://en.wikipedia.org/wiki/Uniformization_(probability_theory)) of the CTMC, where we assume an independent Poisson process with rate $\hat{\rho}$ that generate a sequence of time stamps. Given a state $Y$ at one time stamp, the state $Y'$ at the next time stamp is either the same as $Y$ or one-hop from $Y$, and is stochastically generated by transition probabilities 

$$\mathbb{P}(Y'\vert Y) =  \delta \, \Gamma_{YY'}  \quad \text{for }Y' \neq Y,$$

$$\mathbb{P}(Y\vert Y) = 1-\delta \sum_{Y'} \Gamma_{YY'},$$

where $\delta = 1/\hat{\rho}.$ For these transition probabilities to make sense, we require

$$ \delta \sum_{Y'} \Gamma_{YY'} \,\,< \,\,1.$$

One can prove that this discrete-time Markov chain is isomorphic to the original CTMC via the obvious map from discrete time to continuous time.

We relax the one-hop condition on the transition probabilities by allowing the individual neuron transitions to be independent of each other when conditioned on their past. Namely, we define the Bernoulli probabilities for the transition of neuron $j$ to be

$$ \mathbb{P}(j \text{ transits} \,\vert\, Y) := 1- \exp (-\delta\,\Gamma_{YY'})  , $$

$$ \mathbb{P}(j \text{ unchanged} \,\vert\, Y) := \exp (-\delta\,\Gamma_{YY'}) , $$

where the state $Y'$ is one-hop from $Y$ via neuron $j$. Note that for small $\delta,$ we have

$$\begin{array}{rl} 
\mathbb{P}(Y\vert Y) & = 
\displaystyle \prod_{Y'} \exp (-\delta\,\Gamma_{YY'}) 
\\ & \\ & = 
\displaystyle \exp\left(-\delta \sum_{Y'} \Gamma_{YY'} \right) \,\,\approx\,\, 1- \delta \sum_{Y'} \Gamma_{YY'}.\end{array}$$

For states $Y'$ which are one-hop from $Y,$ we have

$$ \displaystyle
\mathbb{P}(Y'\vert Y) = \displaystyle \left( 1-\exp (-\delta\,\Gamma_{YY'} )\right) \prod_{Y'' \neq Y'} \exp (-\delta\,\Gamma_{YY''}) \,\, \approx\,\, \displaystyle \delta\,\Gamma_{YY'}, $$

where the product is over other states $Y''$ that are one-hop from $Y.$ For states $Y'$ which are at least two-hops from $Y,$ we have $\mathbb{P}(Y'\vert Y) \approx 0.$ This shows that our Bernoulli model tends to the uniformization model and the original CTMC as $\delta \rightarrow 0.$

In practice, there is nothing in the Bernoulli model that stops us from choosing 

$$1 \,\,\leq\,\, \delta \sum_{Y'} \Gamma_{YY'} $$

since the proposed Bernoulli probabilities satisfy $0 < \exp(-\delta\,\Gamma_{YY'}) < 1$ for all $\delta > 0.$ If $\delta$ is small, we get a better approximation of the original CTMC on the discrete-time digital computer, but the simulation may take a long time since states may remain unchanged through many transitions. We should adjust the value of $\delta$ so that a small but significant percentage of the neurons transition at each time step.

Note that the uniformization model and the Bernoulli model are approximations of the $\delta$-skeleton whose transition probabilities are given by 

$$ \displaystyle
\exp(\delta\Gamma) = I + \delta\Gamma + \frac{1}{2!} (\delta\Gamma)^2 + \cdots $$ 

in terms of the (infinite-dimensional) transition rate matrix $\Gamma.$

In quantum physics {cite}`gill2002foundations` {cite}`lindgren2019quantum`, the same uniformization limit as $\delta \rightarrow 0$ is used to define path integrals or the Schrödinger equation. The analysis involving uniformization, Bernoulli models and $\delta$-skeletons can also be applied to the quantum case.

For the rest of this post, we will focus on the Bernoulli model described above to derive efficient learning algorithms for digital computers.

Explicitly, let $\hat{Y}_n=(\hat{R}_n,\hat{S}_n)$ denote the state of the Bernoulli model at discrete time $n \in \{0,1,2,\ldots \}.$ Let $\mathcal{E} := \mathcal{E}_P$ be the edges and $\mathcal{V} := \mathcal{V}_Z \cup \mathcal{V}_X$ the neurons of the generative CTMC. Let $w := w^{(P)}$ be its parameters, and $ \delta := 1/\hat{\rho}$ with $0 < \delta \ll 1.$ 

At time $n=0,$ we assume all signed states are $\hat{S}_{in}=+1$ and all counting states are $\hat{R}_{ijn}=0.$ Given the state $\hat{Y}_n=(\hat{R}_n,\hat{S}_n),$ we define the membrane potential and transition rate at neuron $j$ to be

$$ \displaystyle 
\hat{U}_{j n} = \sum_{e \in \mathcal{E}} \mathbb{I}(e(1)=j) \,\hat{R}_{en} \,w_{e} \, , $$

$$ \displaystyle
\hat{\rho}_{j n}  = \rho_{\hat{S}_{jn}} \exp\left(\,\beta_{\hat{S}_{jn}}\, \hat{S}_{jn} \,\hat{U}_{jn}\,\right), $$

where $\hat{R}_{en}$ is notation for $\hat{R}_{e(0)e(1)n}.$ The Bernoulli transition probabilities are 

$$ \begin{array}{rl}
\mathbb{P}(j \text{ unchanged} \,\vert\, \hat{Y}_n) &= \exp (-\delta \hat{\rho}_{j n} ), 
\\ & \\ 
\mathbb{P}(j \text{ transits} \,\vert\, \hat{Y}_n) &= 1- \exp (-\delta \hat{\rho}_{j n} ) \,\,\approx\,\, \delta \hat{\rho}_{j n}  \exp (-\delta \hat{\rho}_{j n} ) . \end{array} $$

The probability of any signed state $\hat{S}_{n+1}$ given $\hat{S}_{n}$ can then be approximated as

$$ \displaystyle 
\mathbb{P}(\hat{S}_{n+1}\vert \hat{S}_{n})\quad  \approx \quad  \prod_{j \in \mathcal{V}}\, \exp (-\delta\hat{\rho}_{j n}) \,\,\prod_{j \in \mathcal{V}} \, \left( \delta\hat{\rho}_{j n} \right)^{\,\mathbb{I}(\hat{S}_{j(n+1)} \neq \hat{S}_{jn})}. $$

We denote this stochastic update as

$$ \displaystyle 
\hat{S}_{j(n+1)} \,\, \sim\,\, \text{Flip}(\hat{S}_{jn},1-\exp (-\delta \hat{\rho}_{j n} )) $$

where $1-\hat{p}_{jn}$ is the probability that sign $\hat{S}_{jn}$ of neuron $j$ is flipped in $\hat{S}_{j(n+1)}$.

Given the current $(\hat{R}_n,\hat{S}_n)$ and the next $\hat{S}_{n+1},$ we now describe the effect on the synaptic counts $\hat{R}_{n+1}.$ Let $\Delta S$ be the set of neurons which spiked from $\hat{S}_{n}$ to $\hat{S}_{n+1},$ i.e. neurons $j \in \mathcal{V}$ with $\hat{S}_{jn} = +1$ and $\hat{S}_{j(n+1)} = -1.$ We reset the upstream synaptic counts 

$$ \displaystyle
\hat{R}_{ij(n+1)} = 0 \quad \text{for all }i \in \mathcal{V}\setminus \Delta S,\,\, j \in \Delta S,$$

and increment the downstream synaptic counts

$$ \displaystyle
\hat{R}_{ij(n+1)} = \hat{R}_{ijn} + 1 \quad \text{for all }i \in \Delta S, \,\,j \in \mathcal{V} \setminus \Delta S.$$ 

As a convention, for pairs of spiking neurons, we set

$$ \displaystyle
\hat{R}_{ij(n+1)} = 1 \quad \text{for all }i \in \Delta S,\,\, j \in \Delta S.$$

The other synaptic counts are unchanged

$$ \displaystyle
\hat{R}_{ij(n+1)} = \hat{R}_{ijn}  \quad \text{for all }i \in \mathcal{V} \setminus \Delta S, \,\,j \in \mathcal{V} \setminus \Delta S .$$

We denote this update to the synaptic counts as

$$ \displaystyle
\hat{R}_{ij(n+1)} = \text{Fire}(\hat{R}_{ijn},\hat{S}_n,\hat{S}_{n+1}).$$

The discrete-time Bernoulli model for the discriminative process is similarly defined, except that we update only the memory states $\hat{R}_Z$ and $\hat{S}_Z$ and that the probabilities of the signed states of only the memory neurons are known. 
 
Finally, let the environment and memory components of $\hat{Y}_n$ be $\hat{X}_n$ and $\hat{Z}_n.$ Let $\{\hat{Q}\}$ and $\{\hat{P}\}$ denote the discrete-time discrimative and generative Bernoulli models derived from the continuous-time models $\{Q\}$ and $\{P\}$ respectively.

### What are the learning updates in discrete time?

Let $\bar{\pi}_*$ be the stationary distribution of the true distribution $\hat{Q}_*$ of the environment $\{\hat{X}_n\}$ and $\bar{\pi}_Q$ the stationary distribution of the full discriminative $\hat{Q}.$ Let $\bar{Q}$ be the discrete-time Markov chain with initial distribution $\bar{\pi}_Q$ and the same transition probabilities as $\hat{Q}.$

Using our relative inference [recipe](2021-06-01-convergence-of-biased-stochastic-approximation/) for online learning, the learning objective is the conditional relative information 

$$\begin{array}{rl} &
\displaystyle V(w^{(Q)}, w^{(P)}) 
\\ & \\ & =
\displaystyle I_{\bar{Q} \Vert \hat{P}}(\hat{Z}_1, \hat{X}_1 \vert \hat{Z}_0, \hat{X}_0) 
\\ & \\ & =
\displaystyle \sum_{\hat{Z}_0,\hat{X}_0} \bar{\pi}_*(\hat{X}_0) \bar{\pi}_Q(\hat{Z}_0\vert \hat{X}_0) \sum_{\hat{Z}_1,\hat{X}_1} \hat{Q}(\hat{Z}_1,\hat{X}_1\vert \hat{Z}_0, \hat{X}_0) \log \frac{\hat{Q}(\hat{Z}_1,\hat{X}_1\vert \hat{Z}_0, \hat{X}_0)}{\hat{P}(\hat{Z}_1,\hat{X}_1\vert \hat{Z}_0, \hat{X}_0)} 
\end{array}$$

where 

$$ \displaystyle
\hat{Q}(\hat{Z}_1,\hat{X}_1\vert \hat{Z}_0, \hat{X}_0) = \hat{Q}_*(\hat{X}_1\vert \hat{X}_0)  \hat{Q}(\hat{Z}_1\vert \hat{Z}_0, \hat{X}_0),$$

$$ \displaystyle
\hat{Q}(\hat{Z}_1\vert \hat{Z}_0, \hat{X}_0) \quad \approx \quad \prod_{j \in \mathcal{V}_Z} \,\exp \left( -\delta\hat{\rho}^{(Q)}_{j 0} \right) \,\,\prod_{j \in \mathcal{V}_Z}  \, \left( \delta\hat{\rho}^{(Q)}_{j 0} \right)^{\,\mathbb{I}(\hat{S}_{j1} \neq \hat{S}_{j0})}, $$

$$ \displaystyle
\hat{P}(\hat{Z}_1,\hat{X}_1\vert \hat{Z}_0, \hat{X}_0) \quad \approx \quad \prod_{j \in \mathcal{V}} \,\exp (-\delta \hat{\rho}^{(P)}_{j 0} ) \,\,\prod_{j\in \mathcal{V}} \, \left( \delta\hat{\rho}^{(P)}_{j 0} \right)^{\,\mathbb{I}(\hat{S}_{j1} \neq \hat{S}_{j0})}.$$

Taking logarithms, we have

$$ \begin{array}{rl}
\log \hat{Q}(\hat{Z}_1\vert \hat{Z}_0, \hat{X}_0) & \approx \quad 
\displaystyle \sum_{j \in \mathcal{V}_Z} \left(-\delta\,\hat{\rho}^{(Q)}_{j 0} + \mathbb{I}(\hat{S}_{j1} \neq \hat{S}_{j0})\, \log \hat{\rho}^{(Q)}_{j 0} \right)
\\ & \\ & \quad \quad
\displaystyle {}+  \sum_{j \in \mathcal{V}_Z} \mathbb{I}(\hat{S}_{j1} \neq \hat{S}_{j0}) \,\log \delta , \end{array} $$

$$ \begin{array}{rl}
\log \hat{P}(\hat{Z}_1,\hat{X}_1\vert \hat{Z}_0, \hat{X}_0) & \approx \quad 
\displaystyle \sum_{j \in \mathcal{V}} \left(-\delta\,\hat{\rho}^{(P)}_{j 0} + \mathbb{I}(\hat{S}_{j1} \neq \hat{S}_{j0})\, \log \hat{\rho}^{(P)}_{j 0} \right) 
\\ & \\ & \quad \quad
\displaystyle {}+  \sum_{j \in \mathcal{V}} \mathbb{I}(\hat{S}_{j1} \neq \hat{S}_{j0}) \,\log \delta. \end{array} $$

Let us estimate the true log-likelihood $\log \hat{Q}_*(\hat{X}_{1}\vert \hat{X}_0)$ with

$$ \begin{array}{rl}
\xi(\hat{X}_1 \vert \hat{X}_0) & = \quad 
\displaystyle \sum_{j \in \mathcal{V}_X} \left(-\delta\,\hat{\rho}^{(\xi)}_{j0} + \mathbb{I}(\hat{S}_{j1} \neq \hat{S}_{j0})\, \log \hat{\rho}^{(\xi)}_{j0} \right)
\\ & \\ & \quad \quad
\displaystyle {}+  \sum_{j \in \mathcal{V}_X} \mathbb{I}(\hat{S}_{j1} \neq \hat{S}_{j0}) \,\log \delta , \end{array} $$

where given a fixed estimate $\hat{U}^{(\xi)}$ of the environment membrane potential, we have

$$ \displaystyle
\hat{\rho}^{(\xi)}_{jn} = \rho_{\hat{S}_{jn}} \exp(\,\beta_{\hat{S}_{jn}}\, \hat{S}_{jn} \,\hat{U}^{(\xi)}\,).$$

Recall that the recipe prescribes the following updates.

$$\displaystyle X_{n+1} \sim Q_*(X_{n+1} \vert X_{n})$$

$$\displaystyle Z_{n+1} \sim Q_{\lambda_{n}}(Z_{n+1} \vert Z_{n}, X_{n})$$

$$\displaystyle \theta_{n+1} = \theta_{n} + \eta_{n+1} \left.\frac{d}{d\theta} \log P_\theta(Z_{n+1}, X_{n+1} \vert Z_{n},X_{n}) \right\vert _{\theta = \theta_{n}}$$

$$\displaystyle \alpha_{n+1} = \alpha_{n} + \left.\frac{d}{d\lambda} \log Q_{\lambda}(Z_{n+1} \vert  Z_{n},X_{n})\right\vert _{\lambda=\lambda_{n}}$$

$$\displaystyle \lambda_{n+1} = \lambda_{n} - \eta_{n+1} \alpha_{n+1} \left( \xi(X_{n+1}\vert X_n) + \log \frac{Q_{\lambda_{n}}(Z_{n+1}\vert Z_{n},X_{n})}{P_{\theta_{n}}(Z_{n+1},X_{n+1}\vert Z_{n},X_{n})} \right)$$ 

According to the recipe, the parameter updates are 

$$ \begin{array}{rll}
\displaystyle
\hat{U}^{(Q)}_{i n} &= 
\displaystyle
\sum_{e \in \mathcal{E}_Q} \mathbb{I}(e(1)=i) \,\hat{R}_{en} \,w^{(Q)}_{en} &
\text{for all }i\in\mathcal{V}_Z 
\\ & & \\
\displaystyle
\hat{U}^{(P)}_{i n} &= 
\displaystyle
\sum_{e \in \mathcal{E}_P} \mathbb{I}(e(1)=i) \,\hat{R}_{en} \,w^{(P)}_{en} &
\text{for all }i\in\mathcal{V}
\\ & & \\
\displaystyle
\hat{\rho}^{(Q)}_{i n} &= 
\displaystyle 
\rho_{\hat{S}_{in}} \exp(\,\beta_{\hat{S}_{in}}\, \hat{S}_{in} \,\hat{U}^{(Q)}_{in}\,) &
\text{for all }i\in\mathcal{V}_Z 
\\ & & \\
\displaystyle
\hat{\rho}^{(P)}_{i n} &= 
\displaystyle 
\rho_{\hat{S}_{in}} \exp(\,\beta_{\hat{S}_{in}}\, \hat{S}_{in} \,\hat{U}^{(P)}_{in}\,) &
\text{for all }i\in\mathcal{V}
\\ & & \\ 
\displaystyle 
\hat{S}_{i(n+1)} &\sim \text{Flip}(\hat{S}_{in},1-\exp(-\delta \hat{\rho}^{(Q)}_{i n})) &
\text{for all }i\in\mathcal{V}_Z 
\\ & & \\
\displaystyle 
\hat{X}_{n+1} &
\displaystyle
\sim \hat{Q}_*(\hat{X}_{n+1} \vert \hat{X}_{n}) & 
\\ & & \\
\displaystyle 
\hat{R}_{ij(n+1)} &= \text{Fire}(\hat{R}_{ijn},\hat{S}_{n},\hat{S}_{(n+1)}) &
\text{for all }i,j \in\mathcal{V} 
\\ & & \\
\displaystyle 
\alpha^{(P)}_{e(n+1)} &= 
\displaystyle
\phantom{\alpha^{(P)}_{en} +} \Delta\alpha^{(P)}_e & 
\text{for all }e\in\mathcal{E}_Q
\\ & & \\
\displaystyle 
\alpha^{(Q)}_{e(n+1)} &= 
\displaystyle
\alpha^{(Q)}_{en} + \Delta\alpha^{(Q)}_e & 
\text{for all }e\in\mathcal{E}_Q
\\ & & \\
\displaystyle 
w^{(P)}_{e(n+1)} & =
\displaystyle 
w^{(P)}_{en} + \eta_{n+1} \alpha^{(P)}_{e(n+1)} &\text{for all }e\in\mathcal{E}_P 
\\ & & \\
\displaystyle 
w^{(Q)}_{e(n+1)} &= 
\displaystyle
w^{(Q)}_{en} - \eta_{n+1}\, \alpha^{(Q)}_{e(n+1)} \gamma  &\text{for all }e\in\mathcal{E}_Q \end{array}$$

where

$$ \begin{array}{rl} 
\displaystyle 
\Delta \alpha^{(P)}_e & =
\displaystyle 
\left. \frac{d}{dw^{(P)}_e} \log \hat{P}_{w^{(P)}}(\hat{Z}_{n+1}, \hat{X}_{n+1} \vert \hat{Z}_{n},\hat{X}_{n})  \right\vert _{w^{(P)} = w^{(P)}_n} 
\\ & \\ & =
\displaystyle 
\left(-\delta\,\hat{\rho}^{(P)}_{e(1)n} + \mathbb{I}(\hat{S}_{e(1)(n+1)} \neq \hat{S}_{e(1)n}) \right) \frac{d}{dw^{(P)}_e} \log \hat{\rho}^{(P)}_{e(1)n}  
\\ & \\ & =
\displaystyle 
\left(-\delta\,\hat{\rho}^{(P)}_{e(1)n} + \mathbb{I}(\hat{S}_{e(1)(n+1)} \neq \hat{S}_{e(1)n}) \right) \,\beta_{\hat{S}_{e(1)n}} \,\hat{S}_{e(1)n} \,\hat{R}_{en},
\\ & \\ 
\displaystyle \Delta\alpha^{(Q)}_e & =
\displaystyle \left. \frac{d}{dw^{(Q)}_e} \log \hat{Q}_{w^{(Q)}}(\hat{Z}_{n+1} \vert  \hat{Z}_{n},\hat{X}_{n}) \right\vert _{w^{(Q)}=w^{(Q)}_{n}}
\\ & \\ & =
\displaystyle 
\left(-\delta\,\hat{\rho}^{(Q)}_{e(1)n} + \mathbb{I}(\hat{S}_{e(1)(n+1)} \neq \hat{S}_{e(1)n}) \right) \frac{d}{dw^{(Q)}_e} \log \hat{\rho}^{(Q)}_{e(1)n}  
\\ & \\ & =
\displaystyle 
\left(-\delta\,\hat{\rho}^{(Q)}_{e(1)n} + \mathbb{I}(\hat{S}_{e(1)(n+1)} \neq \hat{S}_{e(1)n}) \right) \,\beta_{\hat{S}_{e(1)n}} \,\hat{S}_{e(1)n} \,\hat{R}_{en},
\\ & \\
\displaystyle \gamma & = 
\displaystyle  \xi(\hat{X}_{n+1}\vert\hat{X}_{n}) + \log \frac{\hat{Q}_{w^{(Q)}_n}(\hat{Z}_{n+1}\vert \hat{Z}_{n}, \hat{X}_{n})}{\hat{P}_{w^{(P)}_n}(\hat{Z}_{n+1},\hat{X}_{n+1}\vert \hat{Z}_{n},\hat{X}_{n})} 
\\ & \\ & =
\displaystyle
-\delta \sum_{j \in \mathcal{V}_Z} \left(\hat{\rho}^{(Q)}_{j n}-\hat{\rho}^{(P)}_{j n}\right) -\delta \sum_{j \in \mathcal{V}_X} \left(\hat{\rho}^{(\xi)}_{j n}-\hat{\rho}^{(P)}_{j n}\right) 
\\ & \\ & \quad
\displaystyle 
{} + \sum_{j \in \mathcal{V}_Z} \mathbb{I}(\hat{S}_{j(n+1)} \neq \hat{S}_{jn})\, \beta_{\hat{S}_{jn}}\, \hat{S}_{jn} \left(\hat{U}^{(Q)}_{jn}-\hat{U}^{(P)}_{jn}\right) 
\\ & \\ & \quad
\displaystyle
{} + \sum_{j \in \mathcal{V}_X} \mathbb{I}(\hat{S}_{j(n+1)} \neq \hat{S}_{jn})\, \beta_{\hat{S}_{jn}}\, \hat{S}_{jn} \left(\hat{U}^{(\xi)}-\hat{U}^{(P)}_{jn}\right).
\end{array} $$

The reason for writing the $w^{(P)}_e,$ $w^{(Q)}_e$ updates in terms of auxiliary parameters $\alpha^{(P)}_e,$ $\alpha^{(Q)}_e$ is for comparison of the generative and discriminative rules. Note that $\alpha^{(Q)}_e$ accumulates the updates from $\Delta \alpha^{(Q)}_e$ but $\alpha^{(P)}_e$ does not. 

By the convergence [theorem](2021-06-01-convergence-of-biased-stochastic-approximation/#theorem-convergence-of-online-learning) for relative inference online learning, the proposed learning algorithm converges to a local minimum of the objective $V(w^{(Q)},w^{(P)})$ with

$$ \displaystyle
\mathbb{E}\left[\left\Vert \,\nabla V\left(w^{(Q)}_N, w^{(P)}_N\right)\, \right\Vert^2\right] = O(T^{-1/2}\log T)$$

where $N$ is the random stop time and $T$ is the maximum time horizon for the algorithm as described in this [post](2021-06-01-convergence-of-biased-stochastic-approximation/).

### What are the learning updates in continuous time?

In continuous time, we may derive the learning updates directly from our relative inference recipe, or in this case we may derive them from the discrete-time updates by considering the limit $\delta \rightarrow 0.$

The dynamics of the $w_t$-controlled continuous-time Markov chain is similar to the continuous-time dynamics described previously, except that the control parameter $w_t$ changes with time. As before, the environment process $\{X_t\}$ evolves with some fixed rate kernel $\Gamma^{(X)}$ that is unaffected by the parameter $w_t.$ The discriminative process $\{Z_t\}$ evolves with the $w_t$-controlled membrane potentials and transition rates

$$ \begin{array}{rll}
\displaystyle
\hat{U}^{(Q)}_{i n} &= 
\displaystyle
\sum_{e \in \mathcal{E}_Q} \mathbb{I}(e(1)=i) \,\hat{R}_{en} \,w^{(Q)}_{en} &
\text{for all }i\in\mathcal{V}_Z,
\\ & & \\
\displaystyle
\hat{\rho}^{(Q)}_{i n} &= 
\displaystyle 
\rho_{\hat{S}_{in}} \exp(\,\beta_{\hat{S}_{in}}\, \hat{S}_{in} \,\hat{U}^{(Q)}_{in}\,) &
\text{for all }i\in\mathcal{V}_Z,
\end{array}$$

and changes to the signs $S_t$ will drive the before-mentioned updates to the synaptic counts $R_t.$ We also compute the membrane potentials and transition rates of the generative process which will drive changes in the generative parameters $w^{(P)}.$

$$ \begin{array}{rll}
\displaystyle
\hat{U}^{(P)}_{i n} &= 
\displaystyle
\sum_{e \in \mathcal{E}_P} \mathbb{I}(e(1)=i) \,\hat{R}_{en} \,w^{(P)}_{en} &
\text{for all }i\in\mathcal{V},
\\ & & \\
\displaystyle
\hat{\rho}^{(P)}_{i n} &= 
\displaystyle 
\rho_{\hat{S}_{in}} \exp(\,\beta_{\hat{S}_{in}}\, \hat{S}_{in} \,\hat{U}^{(P)}_{in}\,) &
\text{for all }i\in\mathcal{V}.
\end{array}$$

There are two kinds of learning updates: the transition updates which occurs when there is a spike or recovery, and the holding updates which occurs in between the transitions.

At every transition in the network, we get an immediate update

$$ \begin{array}{rll}
\displaystyle 
\alpha^{(P)}_{et} &= 
\displaystyle
\phantom{\alpha^{(P)}_{et_-} +} \Delta\alpha^{(P)(T)}_e & 
\text{for all }e\in\mathcal{E}_Q
\\ & & \\
\displaystyle 
\alpha^{(Q)}_{et} &= 
\displaystyle
\alpha^{(Q)}_{et_-} + \Delta\alpha^{(Q)(T)}_e & 
\text{for all }e\in\mathcal{E}_Q
\\ & & \\
\displaystyle 
w^{(P)}_{et} & =
\displaystyle 
w^{(P)}_{et_-} + \eta_{t} \alpha^{(P)}_{et} &\text{for all }e\in\mathcal{E}_P 
\\ & & \\
\displaystyle 
w^{(Q)}_{et} &= 
\displaystyle
w^{(Q)}_{et_-} - \eta_{t}\, \alpha^{(Q)}_{et} \gamma^{(T)}  &\text{for all }e\in\mathcal{E}_Q \end{array}$$

with

$$ \begin{array}{rl} 
\displaystyle 
\Delta \alpha^{(P)(T)}_e & =
\displaystyle 
\mathbb{I}(\hat{S}_{e(1)t} \neq \hat{S}_{e(1)t_-}) \,\beta_{\hat{S}_{e(1)t_-}} \,\hat{S}_{e(1)t_-} \,\hat{R}_{et_-},
\\ & \\ 
\displaystyle \Delta\alpha^{(Q)(T)}_e & =
\displaystyle 
\mathbb{I}(\hat{S}_{e(1)t} \neq \hat{S}_{e(1)t_-}) \,\beta_{\hat{S}_{e(1)t_-}} \,\hat{S}_{e(1)t_-} \,\hat{R}_{et_-},
\\ & \\
\displaystyle \gamma^{(T)} & = 
\displaystyle 
\sum_{j \in \mathcal{V}_Z} \mathbb{I}(\hat{S}_{jt} \neq \hat{S}_{jt_-})\, \beta_{\hat{S}_{jt_-}}\, \hat{S}_{jt_-} \left(\hat{U}^{(Q)}_{jt_-}-\hat{U}^{(P)}_{jt_-}\right) 
\\ & \\ & \quad
\displaystyle
{} + \sum_{j \in \mathcal{V}_X} \mathbb{I}(\hat{S}_{jt} \neq \hat{S}_{jt_-})\, \beta_{\hat{S}_{jt_-}}\, \hat{S}_{jt_-} \left(\hat{U}^{(\xi)}-\hat{U}^{(P)}_{jt_-}\right).
\end{array} $$

where $t_-$ denotes the instant infinitesimally before $t,$ and $\eta_t$ is the learning rate at time $t.$

Note that the only generative weights $w^{(P)}_e$ being updated are those whose postsynaptic neuron has a transition. The $w^{(P)}_e$ update is a signed scaled version ($\eta_t \beta_+$ or $-\eta_t \beta_-$) of the number $\hat{R}_{et_-}$ of presynaptic spikes since the last postsynaptic spike. Therefore, the learning update is local: triggered by postsynaptic transitions and depending only on information available to the synapse itself.

Similarly, the local update to the auxiliary parameter $\alpha^{(Q)}_{e}$ for each synapse $e$ is triggered by postsynaptic transitions and uses only information available to the synapse.

The discriminative $w^{(Q)}_e$ update however is triggered by _any_ transition in the network. The immediate update is the product $\eta_t \alpha^{(Q)}_{E}\gamma^{(T)}$ where $\alpha^{(Q)}_e$ is an auxiliary parameter that is local to the synapse $e.$ It is modulated by an auxiliary parameter $\gamma^{(T)}$ that is global to the network, in the sense that it modulates every synapse to the same extent. If the transitions are all one-hop even for environmental neurons, the modulator becomes

$$ \displaystyle 
\gamma^{(T)} = \beta_{\hat{S}_{jt_-}}\, \hat{S}_{jt_-} \left(\hat{U}^{(Q)}_{jt_-}-\hat{U}^{(P)}_{jt_-}\right) $$

if a memory neuron $j$ transitions, or

$$ \displaystyle 
\gamma^{(T)} = \beta_{\hat{S}_{jt_-}}\, \hat{S}_{jt_-} \left(\hat{U}^{(\xi)}-\hat{U}^{(P)}_{jt_-}\right) $$

if an environment neuron $j$ transitions. The modulating signal can easily be computed and broadcasted by the transitioning neuron as a signed scaled version of the difference between the discriminative and generative membrane potentials. Following {cite}`rezende2014stochastic`, we will call this modulator _surprise_.

In the holding times between the network transitions, we have gradual updates defined by the differential equations

$$ \begin{array}{rll}
\displaystyle 
\alpha^{(P)}_{et} &= 
\displaystyle
\Delta\alpha^{(Q)(H)}_e & 
\text{for all }e\in\mathcal{E}_Q
\\ & & \\
\displaystyle 
\dot{\alpha}^{(Q)}_{et} &= 
\displaystyle
\Delta\alpha^{(Q)(H)}_e & 
\text{for all }e\in\mathcal{E}_Q
\\ & & \\
\displaystyle 
\dot{w}^{(P)}_{et} & = 
\displaystyle \phantom{-}
\eta_{t} \alpha^{(P)}_{et} &\text{for all }e\in\mathcal{E}_P 
\\ & & \\
\displaystyle 
\dot{w}^{(Q)}_{et} &= 
\displaystyle -
\eta_{t}\, \alpha^{(Q)}_{et} \gamma^{(H)}  &\text{for all }e\in\mathcal{E}_Q \end{array}$$

with

$$ \begin{array}{rl} 
\displaystyle 
\Delta \alpha^{(P)(H)}_e & = 
\displaystyle 
-\hat{\rho}^{(P)}_{e(1)t}  \,\,\beta_{\hat{S}_{e(1)t}} \,\hat{S}_{e(1)t} \,\hat{R}_{et},
\\ & \\ 
\displaystyle \Delta\alpha^{(Q)(H)}_e & =
\displaystyle 
-\hat{\rho}^{(Q)}_{e(1)t} \,\,\beta_{\hat{S}_{e(1)t}} \,\hat{S}_{e(1)t} \,\hat{R}_{et},
\\ & \\
\displaystyle \gamma^{(H)} & = 
\displaystyle 
- \sum_{j \in \mathcal{V}_Z} \left(\hat{\rho}^{(Q)}_{j n}-\hat{\rho}^{(P)}_{j n}\right) - \sum_{j \in \mathcal{V}_X} \left(\hat{\rho}^{(\xi)}_{j n}-\hat{\rho}^{(P)}_{j n}\right).
\end{array} $$

Depending on the state of the postsynaptic neuron, the generative weights $w^{(P)}_e$ and auxiliary discriminative parameters $\alpha^{(Q)}_e$ will decay and weaken if the postsynaptic neuron is in a resting state, but they will strengthen if the postsynaptic neuron is in a refractory state. The rate of weakening and strengthening increases with the postsynaptic transition rates and the synaptic counts.

The discriminative weights $w^{(Q)}_e$ will weaken or strengthen depending on the auxiliary discriminative parameters $\alpha^{(Q)}_e$ and the holding modulator $\gamma^{(H)},$ which is different from the transition modulator $\gamma^{(T)}.$ The holding modulator changes much more gradually, ebbing and flowing with the sum of spiking rates of the discriminative and generative networks. Generally, when considering the likelihood of the current holding state $(Z_t,X_t)$ given the past states, the modulator is positive if the discriminative process assigns a higher holding probability (which varies inversely to the transition rate) than the generative process, but the modulator is negative if the discriminative process assigns a lower holding probability.

Recall that if $e$ is a self-loop on neuron $j$ with $e(0) = e(1) =j,$ then the bias $w^{(P)}_e$ represents contributions to the membrane potential which are not from other neurons. These biases also get updated with the same transition and holding formulas above, with the synaptic count $\hat{R}_e = 1$ by default. If the neuron is holding in a resting state, then $w^{(P)}_e$ weakens, representing the gradual polarization of the membrane potential. If the neuron is holding in a refractory state, then $w^{(P)}_e$ strengthens, so the membrane potential depolarizes. 

These strengthenings and weakenings of the synaptic weights and the neural biases help to explain, using the context of learning, the rise and decay of synaptic connections and membrane potentials observed biologically during holding states. These changes are often hard-coded as exponential filters {cite}`rezende2014stochastic` in traditional models.

Interestingly, if the neuron is spiking, the transition update pulls up the bias $w^{(P)}_e$, which tallies with observations of a sharp depolarization of the membrane potential initially when the neuron spikes. The resetting of synaptic counts in the model causes the membrane potential to quickly fall back to the resting potential, which tallies with observations of a sharp repolarization of the neuron. If the neuron is recovering, the transition update pushes down the bias $w^{(P)}_e$. This prescription tallies with observations of hyperpolarization which occurs after the absolute refractory period.

We conjecture that in continuous time, the proposed learning algorithm converges to a local minimum of the objective $V(w^{(Q)},w^{(P)})$ with

$$ \displaystyle
\mathbb{E}\left[\left\Vert \,\nabla V\left(w^{(Q)}_N, w^{(P)}_N\right)\, \right\Vert^2\right] = O(T^{-1/2}\log T)$$

where $N$ and $T$ are continuous-time analogues of the random stop time and maximum time horizon described in this [post](2021-06-01-convergence-of-biased-stochastic-approximation/).

### How nature could derive such a complex learning algorithm?
 
In the previous section, we derived a statistically-sound online learning algorithm for updating parameters $w$ such that the given $w$-controlled stochastic process would eventually learn a good approximation of the true distribution.

We then hypothesized how biological neural networks could be implementing this learning algorithm, assuming that the given $w$-controlled stochastic process is a good model of actual neural dynamics. If our hypothesis is correct, it would imply that the biological algorithms are effective at learning the true distribution.

However, we are left with the meta-question of how biological networks could arrive at such a specific learning algorithm in the first place. There are two possible explanations.

The first is evolution. Organisms with effective learning algorithms will survive better in complex environments than those with poor algorithms. In the long run, powerful features in the algorithms are passed down for many generations, and the accumulation of these features is what we observe in biological networks today.

The second is information theory. Information is energy. The relative information learning objective $V(w^{(Q)},w^{(P)})$ could possibly manifest itself in the neural network as some kind of free energy {cite}`friston2010free`. In the process of minimizing the energy consumption, the biological processes naturally align themselves to cause learning to occur in the network. The direct goal was not learning but just simple energy minimization. As Karl Friston puts it {cite}`raviv2018genius`, 

> "All these contrived, anthropomorphized explanations of purpose and survival and the like all seemed to just peel away, and the thing you were observing just was. In the sense that it could be no other way." 

### Unfinished tasks

The $\alpha^{(Q)}_e$ updates described above may cause the learning algorithm to be numerically unstable and prevent the parameter updates from converging. This instability can be seen in the [condition](2021-06-01-convergence-of-biased-stochastic-approximation/) C5 that has to be satisfied by the corresponding Poisson equation solutions for the discriminative correction terms.

Instead, we propose the following updates, based on techniques in variance reduction. 

$$ \begin{array}{rll}
\displaystyle 
\alpha^{(P)}_{e(n+1)} &= 
\displaystyle
\phantom{\lambda\alpha^{(P)}_{en} + }\Delta\alpha^{(P)}_e & 
\text{for all }e\in\mathcal{E}_Q
\\ & & \\
\displaystyle 
\alpha^{(Q)}_{e(n+1)} &= 
\displaystyle
\lambda\alpha^{(Q)}_{en} + \Delta\alpha^{(Q)}_e & 
\text{for all }e\in\mathcal{E}_Q
\end{array}$$

where $\lambda$ is a discount factor that reduces the effect of old updates $\Delta \alpha^{(Q)}_e$ on $\alpha^{(Q)}_e.$ 

The first task is to prove that the solution of its Poisson equation is stable. The parameters will probably converge to a quasi-stationary point with a non-vanishing bias. The proof will be similar to that of Corollary 2 of {cite}`karimi2019non`.

The second task is to follow {cite}`liu2020biologically` in proving mathematically that for a given synaptic weight, the learning updates satisfy spike timing dependent plasticity (STDP). 

The third task is to develop an [active](2021-03-22-relative-inference-with-mutable-processes/#is-relative-inference-necessarily-passive) form of relative inference that uses mutable processes. This form of active online learning can then be applied to reinforcement learning problems in robots and machine [reasoning](2021-04-22-proofs-as-programs-challenges-and-strategies-for-program-synthesis/).

To see the analogy between our results here and those in reinforcement learning, we write down the following description from {cite}`karimi2019non` of the policy gradient method for average reward over infinite horizon. 

$$ \displaystyle
Q_\eta((s,a),(s',a')) = \Pi_\eta(a';s') P^a_{s,s'}$$

$$ \displaystyle
J(\eta) = \sum_{s,a} v(s,a) R(s,a)$$

$$ \displaystyle
\nabla J(\eta) = \lim_{T\rightarrow\infty}\mathbb{E}_\eta\left[R(S_T,A_T)\sum_{i=0}^{T-1}\nabla\log\Pi_\eta(A_{T-i};S_{T-i})\right]$$

$$ \displaystyle
\hat{\nabla}_T J(\eta) = R(S_T,A_T)\sum_{i=0}^{T-1}\lambda^i \nabla\log\Pi_\eta(A_{T-i};S_{T-i})$$

$$ \displaystyle
G_{n+1} = \lambda G_n + \nabla \log\Pi_{\eta_n}(A_{n+1},S_{n+1})$$

$$ \displaystyle
\eta_{n+1} = \eta_n + \gamma_{n+1} G_{n+1} R(S_{n+1},A_{n+1})$$

We briefly define the notations above.

1. Environment states $s, s'$
2. Agent actions $a, a'$
3. Model parameter $\eta$ 
4. Agent policy probability $\Pi_\eta(a';s')$ of choosing action $a'$ on state $s'$
5. State transition probability $P^a_{s,s'}$ of next state $s'$ given action $a$ on state $s$ 
6. State-action transition probability $Q_\eta((s,a),(s',a'))$
7. Average reward objective $J(\eta)$ for maximization
8. Stationary distribution $v(s,a)$ of state-action transition matrix
9. Reward $R(s,a)$ of action $a$ on state $s$
10. Time horizon $T$
11. Objective gradient $\nabla J(\eta)$ with respect to $\eta$
12. Expectation $\mathbb{E}_\eta$ over state-action Markov chain $Q_\eta$
13. Discount factor $\lambda$
14. Gradient estimator $\hat{\nabla}_T J(\eta)$
15. Auxiliary parameter $G_n$ at $n$-th iteration of algorithm
16. Model parameter $\eta_n$ at $n$-th iteration of algorithm
17. Learning rate $\gamma_n$ at $n$-th iteration of algorithm

From our algorithm, we list the analogous formulas in the same order.

$$ \displaystyle 
Q_{w^{(Q)}}(Z',X'\vert Z,X) = Q_{w^{(Q)}}(Z'\vert Z,X)Q_*(X'\vert X)$$

$$ \displaystyle
V(w^{(Q)},w^{(P)}) = \int \bar{\pi}_{w^{(Q)}}(dZ',dX',dZ,dX)  \left(\log \frac{Q_{w^{(Q)}}(Z',X' \vert Z,X)}{P_{w^{(P)}}(Z',X' \vert Z,X)} \right)$$

$$ \begin{array}{rl}
\displaystyle
\frac{\partial V}{\partial w^{(Q)}}(w^{(Q)},w^{(P)}) &= 
\displaystyle
\lim_{T\rightarrow \infty} \mathbb{E}_{Q_{w^{(Q)}}(Z_{0..(T+1)},X_{0..(T+1)})} \Bigg[ \left( \log \frac{Q_{w^{(Q)}}(Z_{T+1},X_{T+1}\vert Z_T,X_T)}{P_{w^{(P)}}(Z_{T+1},X_{T+1}\vert Z_T,X_T)} \right) 
\\ & \\ & \quad \quad \quad \quad
\displaystyle 
{}\times \sum_{t=0}^T \frac{d}{d{w^{(Q)}}} \log Q_{w^{(Q)}}(Z_{t+1} \vert  Z_{t},X_{t}) \Bigg] \end{array}$$

$$ \begin{array}{rl}
\displaystyle
\frac{\widehat{\partial V}_T}{\partial w^{(Q)}}(w^{(Q)},w^{(P)}) &= 
\displaystyle
 \left( \log \frac{Q_{w^{(Q)}}(Z_{T+1},X_{T+1}\vert Z_T,X_T)}{P_{w^{(P)}}(Z_{T+1},X_{T+1}\vert Z_T,X_T)} \right) 
\\ & \\ & \quad \quad \quad \quad
\displaystyle 
{}\times \sum_{t=0}^T \lambda^{T-t} \frac{d}{d{w^{(Q)}}} \log Q_{w^{(Q)}}(Z_{t+1} \vert  Z_{t},X_{t})  \end{array}$$

$$ \displaystyle
\alpha^{(Q)}_{e(n+1)} = \lambda \alpha^{(Q)}_{en}+ \frac{d}{dw^{(Q)}_e} \log Q_{w^{(Q)}}(Z_{n+1} \vert  Z_{n},X_{n}) $$

$$ \displaystyle
w^{(Q)}_{e(n+1)} = w^{(Q)}_{e(n+1)} - \eta_{n+1}\alpha^{(Q)}_{e(n+1)} \left( \log \frac{Q_{w^{(Q)}_n}(Z_{n+1},X_{n+1}\vert Z_n,X_n)}{P_{w^{(P)}_n}(Z_{n+1},X_{n+1}\vert Z_n,X_n)} \right)$$

with the following analogous notations.

1. Environment states $X, X'$
2. Memory/mutable states $Z, Z'$
3. Discriminative parameter $w^{(Q)}$ 
4. Discriminative conditional distribution $Q_{w^{(Q)}}(Z'\vert Z,X)$ 
5. True distribution $Q_*(X'\vert X)$
6. Discriminative joint distribution $Q_{w^{(Q)}}(Z',X'\vert Z,X)$
7. Relative information objective $V(w^{(Q)},w^{(P)})$ for minimization
8. One-step stationary distribution $\bar{\pi}_{w^{(Q)}}(dZ',dX',dZ,dX)$ of $Q_{w^{(Q)}}$ 
9. Surprise or log-likelihood ratio $\log Q_{w^{(Q)}}(Z',X' \vert Z,X)/P_{w^{(P)}}(Z',X' \vert Z,X) $
10. Time horizon $T$
11. Objective gradient $\partial V/ \partial w^{(Q)}$
12. Expectation $\mathbb{E}_{Q_{w^{(Q)}}(Z_{0..(T+1)},X_{0..(T+1)})}$ over state-action Markov chain $Q_{w^{(Q)}}$
13. Discount factor $\lambda$
14. Gradient estimator $\widehat{\partial V}_T/\partial w^{(Q)}$
15. Auxiliary parameter $\alpha^{(Q)}_{en}$ at $n$-th iteration of algorithm
16. Model parameter $w^{(Q)}_{en}$ at $n$-th iteration of algorithm
17. Learning rate $\eta_n$ at $n$-th iteration of algorithm

The analogy is not perfect, since the rewards in our problem depend on generative parameters $w^{(P)}$ that also need to be optimized, but the proof ideas are similar.

### References

```{bibliography}
:filter: docname in docnames
```