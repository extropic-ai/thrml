# Developer Documentation

## What is `THRML`?

As was discussed in previous documents, `THRML` is a [JAX](https://docs.jax.dev/en/latest/)‑based Python package for efficient [block Gibbs sampling](https://proceedings.mlr.press/v15/gonzalez11a/gonzalez11a.pdf) of graphical models at scale. `THRML` provides the tools to do block Gibbs sampling on any graphical model, and provides the tooling already for models such as [Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/cogscibm.pdf). 


## How does `THRML` work?

From a user perspective, there are three main components of `THRML` that they will interact with: blocks, factors, and programs. For detailed usage examples, see the example notebooks.

Blocks are fundamental to `THRML` since it implements block sampling. A `Block` is a collection of nodes of the same type with implicit ordering.

Factors and their associated conditionals are the backbone of sampling. Factors derive their name from factor graphs, and organize interactions between variables into a [bipartite graph of factors and variables](https://ocw.mit.edu/courses/6-438-algorithms-for-inference-fall-2014/3e3e9934d12e3537b4e9b46b53cd5bf1_MIT6_438F14_Lec4.pdf). Factors synthesize collections of interactions via `InteractionGroups` and must implement a `to_interaction_groups()` method. Below is the hierarchy of interactions and samplers provided for clarity.


Programs are the key orchestrating data structures. `BlockSamplingProgram` handles the mapping and bookkeeping for padded block Gibbs sampling, managing global state representations efficiently for JAX. `FactorSamplingProgram` is a convenient wrapper that converts factors to interaction groups. These programs coordinate free/clamped blocks, samplers, and interactions to execute the sampling algorithm.


From a developer perspective, the core approach to `THRML` is to represent as much as possible as contiguous arrays/pytrees, operate on these structures, then map to and from them for the user. Internally, this is often referred to as the "global" state (in opposition to the "block" state). This can be seen as a similar approach to data driven design (via SoA) and is similar to other JAX graphical model packages (e.g. [PGMax](https://github.com/google-deepmind/PGMax)). Taking PGMax as an example, an important distinction is that `THRML` supports pytree states and heterogeneous states. There is more than one way to approach this heterogeneity and `THRML` takes an approach that relies on splitting the nodes according to the pytrees and organizing a global state as a list of these pytrees (which are then stacked if there are multiple blocks that share a given pytree). Thus, the global state is a list of these pytree structures. Since JAX is optimized for efficient array/pytree operation we want to do as much as possible in that form, so we define a standard representation and order for this global structure (which itself doesn't really matter much, it just matters that we know how to get to and from this order) in this array format (in which all pytree structs of the same type get stacked together), then map indices there and back to other representations. The management of these indices and mapping is constructed/held by the program. 


Since JAX does not support ragged arrays, every block must be the same size (in the array leaves). In order to solve this problem (since blocks in the graph may be different sizes), `THRML` constructs the global representation by stacking the blocks (of the same pytree type) and pad them out as needed. There exists a tradeoff between padding out blocks which can add runtime overhead (from unnecessary computation) and other approaches, such as looping over blocks which could pay (a likely untenable) compile time cost instead.

Everything else that exists in `THRML` exists to provide convenience for creating and working with a program. With a focused core on block index management and padding, this allows for a lightweight and hackable code base (with only 1,000 LoC).


## What are the limitations of `THRML`?

While `THRML` is fast and efficient, users new to sampling may expect a panacea where none can exist. First and foremost, it is important to note that sampling is a very difficult problem. To generate samples from a distribution in high dimensional space can take (prohibitively) many steps even if we parallelize proposals. `THRML` is also very focused on Gibbs sampling, as Extropic seeks to provide hardware that accelerates this algorithm, but for general sampling it is unknown when Gibbs sampling as an MCMC method is substantially [faster](https://arxiv.org/abs/2007.08200) or [slower](https://arxiv.org/abs/1605.00139) than other MCMC methods and thus specific problems may require specific tools. As a pedagogical example, consider a two node Ising model with a single edge. If $J=-\infty, h=0$, Gibbs sampling will never mix between the ground states {-1, -1}, {1, 1} since it will never flip once it reaches one of these states (but an approach such as Uniform MH would be able to converge quickly).



## `THRML` Overviews

<img src="../flow.png" alt="A diagram which shows the flow of different components into the FactorSamplingProgram" width="400"/>

#### Factors:

- `AbstractFactor`
    - `WeightedFactor`: Parameterized by weights
    - `EBMFactor`: defines energy functions for Energy-Based Models
        - `DiscreteEBMFactor`: EBMs with discrete states (spin and categorical)
            - `SquareDiscreteEBMFactor`: Optimized for square interaction tensors
                - `SpinEBMFactor`: Spin-only interactions ({-1, 1} variables)
                - `SquareCategoricalEBMFactor`: Square categorical interactions
            - `CategoricalEBMFactor`: Categorical-only interactions  

#### Samplers:

- `AbstractConditionalSampler`
    - `AbstractParametricConditionalSampler`
        - `BernoulliConditional`: Spin-valued Bernoulli sampling
            - `SpinGibbsConditional`: Gibbs updates for spin variables in EBMs
        - `SoftmaxConditional`: Categorical softmax sampling  
            - `CategoricalGibbsConditional`: Gibbs updates for categorical variables in EBMs
