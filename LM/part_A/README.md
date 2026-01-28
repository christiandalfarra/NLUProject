# NLU Project: RNN & LSTM Language Modeling

This repository contains scripts to train and evaluate Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) models for language modeling tasks. The project allows for flexible experimentation with different architectures, optimizers, and dropout regularization techniques.

## Project Structure

- `main.py`: The entry point for the command-line interface (CLI). It parses arguments and handles the logic for training or evaluation.
- `functions.py`: Contains the core logic for model definition, training loops (`training`), and testing loops (`testing`).
- `bin/`: Directory where trained model weights (`.pt` files) are saved and loaded from.

## Usage

The script is run via `main.py` using the `--mode` argument to switch between training and evaluation.

### 1. Training a Model

To train a model, you must set `--mode train` and provide the required architecture, learning rate, and optimizer.

**Syntax:**
```bash
python main.py --mode train --model_arch <ARCH> --lr <RATE> --optimizer <OPT> [options]
```

**Required Arguments:**

- `--model_arch`: The model architecture. Options: `RNN` or `LSTM`.
- `--lr`: Learning rate (float).
- `--optimizer`: The optimizer to use. Options: `SGD` or `AdamW`.

**Optional Arguments:**

- `--emb_size`: dimension of the embedding layer (default: `350`).
- `--hidden_size`: dimension of the hidden layer (default: `350`).
- `--emb_dropout`: Dropout probability for the embedding layer (default: `0.0`).
- `--out_dropout`: Dropout probability for the output layer (default: `0.0`).

**Examples:**

Train a simple RNN with SGD:
```bash
python main.py --mode train --model_arch RNN --lr 0.5 --optimizer SGD
```

Train an LSTM with AdamW and dropout:
```bash
python main.py --mode train --model_arch LSTM --lr 0.0002 --optimizer AdamW --emb_dropout 0.2 --out_dropout 0.2
```

### 2. Evaluating a Model

To evaluate a pre-trained model on the test set, set `--mode evaluate` and specify the `model_number`.

**Syntax:**
```bash
python main.py --mode evaluate --model_number <INT>
```

**Model Number Mapping:**

The script expects specific pre-trained files in the `bin/` folder based on the ID provided:

| Model Number | Architecture | Corresponding File |
|--------------|--------------|-------------------|
| `1` | RNN | `bin/RNN.pt` |
| `2` | LSTM | `bin/LSTM.pt` |
| `3` | LSTM Dropout | `bin/LSTM_DO.pt` |
| `4` | LSTM Dropout AdamW| `bin/LSTM_DO_AdamW.pt` |

**Example:**

Evaluate the model stored in `bin/RNN.pt`:
```bash
python main.py --mode evaluate --model_number 1
```

## Default Configuration

The training loop uses these default parameters defined in `main.py`:

- Embedding Size: 350
- Hidden Size: 350
- Gradient Clipping: 5
- Epochs: 100
- Patience (Early Stopping): 3 epochs