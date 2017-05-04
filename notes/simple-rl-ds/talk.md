**SimpleDS** [[presentation](simpleds_karazeev.pdf)] [[link](https://arxiv.org/pdf/1601.04574.pdf)]

`Anton Karazeev`, Department of Innovation and High Technologies

Good afternoon ladies and gentlemen.
Let me introduce myself. I am Anton, a third year student at MIPT, applied mathematics and physics.
Today I would like to tell you about Reinforcement Learning and its application to Dialogue Systems.

As a source I chose a paper by Heriberto which is about the application of Reinforcement Learning to Dialogue Systems.

I am going to develop four main points. First, I would like to give a short introduction to Reinforcement Learning. Secondly, I will tell you about the application of Reinforcement Learning to the task of Dialogue Control. Thirdly, I am going to explain the novelty of system called “SimpleDS” by Heriberto who is author of the paper. Lastly, the compact summary of my presentation will be given.
After my talk there will be time for a discussion and any questions. That is all for the introduction.

The first part of my talk is about the application of Reinforcement Learning to Dialogue Systems.
I start with a question: “what is the dialogue system?”. It’s the system in which bot answers to user’s requests in human-like way and asks for some information if needed.
The common problem in Dialogue Systems is detecting user’s intentions. The goal is to formalise and extract this intentions from messages (e.g. query “what are the best restaurants in LA?” means that user wants to eat - it’s intention for restaurant or food). This numerical representation reflects probabilities for different user’s intentions that the system can recognise. And this intention vector alters after every dialogue action. The mechanism of system’s transition from one internal state to another is called Dialogue State Tracking. Let’s do this with Reinforcement Learning.

Now let’s move to the second part of my talk, which is about the Reinforcement Learning. This model shows us how human’s brain learns to pick an action depending on a given observation from environment. After execution of action the agent receives a reward. With every step agent tries to increase rewards and discovers initially unknown environment. We want to learn our agent to handle with the environment that can be deterministic or stochastic.

So now we come to the structure of “SimpleDS”.
Dialogue Systems community adopted Reinforcement Learning since it offered the opportunity to treat the dialogue as optimisation problem and RL-based systems improve their performance with experience.
Now I describe the Learning setup at the bottom of this figure: Environment generates the word sequence of the agent’s action, then User Simulator generates a word sequence as a response to that action, this response is “distorted” given some noise and after that the next dialogue state and reward are calculated.
“SimpleDS” uses templates for language generation and rule-based User Simulator.

At last I want to tell you about the Results of this paper. For each state there are the whole set of actions but in this paper they defined only 4-5 actions per state - that’s the difference in behaviour of “SimpleDS”. Advantage - the agent can be quickly learnt and potentially the dialogues can be learnt more sensible way.

I am approaching to the end of my talk. I will briefly summarize the main issues. I gave an introduction to Reinforcement Learning and showed its application to Dialogue Systems. The main goal of this work is to reduce intervention from system developers - from human.

In conclusion I want to mention this dialogue between human and bot which was printed on merchandise bags at ChatbotConf 2016.
Thank you for your attention. If you have any questions, I would be happy to answer them.
