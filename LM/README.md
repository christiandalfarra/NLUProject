# NLU Project: LSTM Language Modeling with Advanced Regularization Techniques

This repository implements and evaluates Long Short-Term Memory (LSTM) models for language modeling tasks, incorporating advanced regularization and optimization techniques as described in [Regularizing and Optimizing LSTM Language Models](https://openreview.net/pdf?id=SyyGPP0TZ).

## Overview

The project is divided into two main parts:

### Part 1.A
1. **LSTM Architecture**: Replaced basic RNN with LSTM for better long-term dependency modeling
2. **Dropout Regularization**: Added dropout after embedding and before output layer
3. **AdamW Optimizer**: Alternative to SGD with adaptive learning rates and weight decay

### Part 1.B
1. **Weight Tying**: Shares weights between embedding and output layers to reduce parameters
2. **Variational Dropout**: Applies the same dropout mask across time steps (not DropConnect)
3. **NT-AvSGD**: Non-monotonically Triggered Averaged SGD for improved convergence

## References

Merity, S., Keskar, N. S., & Socher, R. (2018). [Regularizing and Optimizing LSTM Language Models](https://openreview.net/pdf?id=SyyGPP0TZ). ICLR 2018.