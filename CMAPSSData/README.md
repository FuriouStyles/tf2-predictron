# TF2.0 Prediction
## End-to-End Machine Learning and Planning with Tensorflow 2.o

This is an implementation of the Predictron neural network architecture that was proposed by David Silver et al from Google's DeepMind Team. The Predictron is an abstract reinforcement learning model that can be represented by a Markov reward process.

"""
One of the key challenges of artificial intelligence is to learn models that are effective in the context of planning. In this document we introduce the *predictron* architecture. The predictron consists of a fully abstract model, represented by a Markov reward process, that can be rolled forward multiple "imagined" planning steps. Each forward pass of the predictron accumulates internal rewards and values over multiple planning depths. The predictron is trained end-to-end so as to make these accumulated values accurately approximate the true value function. We applied the predictron to procedurally generated random mazes and a simulator for the game of pool. The predictron yielded significantly more accurate predictions than conventional deep neural network architectures.
"""

This implementation borrows heavily from this Tensorflow 0.14 implementation with some improvements and updates: https://github.com/zhongwen/predictron with some improvements and updates. 

You can find the Predictron paper here: https://arxiv.org/abs/1612.08810.