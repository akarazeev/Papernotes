## Turing Computation with Recurrent Artificial Neural Networks [[link](https://arxiv.org/pdf/1511.01427.pdf)]

Goal: Mapping `Turing Machines` to `RNNs`.

[Google DeepMind's Neural Turing Machines]

Nonlinear Dynamical Automata (NDA)

Map NDA onto a RNN.

Generalized Shift emulating a Turing Machine -> its dynamics on unit square via Gödelization procedure [defines piecewise-affine linear map on the unit square].

Formally `Turing Machines` are 7-tuples:
- Q - finite set of control states
- |\_| - blank symbol
- N - finite set of tape symbols containing the blank symbol
- T - input alphabet
- q_0 - starting state
- F - set of 'halting' states
- \delta - partial transition function, determines the dynamics of the machine: Q x N -> Q x N x {L, R}

Dotted sequences and Generalized Shifts

Gödel Codes

### NDAs to RNNs

ro - maps the orbits of the NDA (Phi) to orbits of the RNN (Zeta).

RNN is denoted with Zeta = ro(I, A, Phi, Theta):
- I - identity matrix that maps initial conditions of NDA to RNN
- A - matrix for network's architecture
- Phi - piecewise affine-linear map
- Theta - switching rule

Three layers:
1. MCL - Machine Configuration Layer - encoding the states
2. BSL - Branch Selection Layer - implementing switching rule
3. LTL - Linear Transformation Layer - generating an updated machine configuration from the previous one

### NDA-simulating first order RNN

NDA (Turing Machine simulation) by the RNN achieved by a combination of synaptic and neural computation among three neural types (MCL, BSL, LTL)
