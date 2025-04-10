**This file is not mandatory**
But if you want, here your can add your comments or anything that you want to share with us
regarding the exercise.
## Part 1.A
**Mandatory requirements**: For the following experiments the perplexity must be below 250 (***PPL < 250***).

1. Replace RNN with a Long-Short Term Memory (LSTM) network --> [link](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
2. Add two dropout layers: --> [link](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
    - one after the embedding layer, 
    - one before the last linear layer
3. Replace SGD with AdamW --> [link](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

## Part 1.B
**Mandatory requirements**: For the following experiments the perplexity must be below 250 (***PPL < 250***) and it should be lower than the one achieved in Part 1.1 (i.e. base LSTM).

Starting from the `LM_RNN` in which you replaced the RNN with a LSTM model, apply the following regularisation techniques:
- Weight Tying 
- Variational Dropout (no DropConnect)
- Non-monotonically Triggered AvSGD 

These techniques are described in [this paper](https://openreview.net/pdf?id=SyyGPP0TZ).

