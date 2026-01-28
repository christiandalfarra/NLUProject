# NLU Project: LSTM Language Modeling with Advanced Techniques

This repository contains scripts to train and evaluate Long Short-Term Memory (LSTM) models for language modeling tasks. The project supports advanced techniques including weight tying, variational dropout, and NT-AvSGD optimization for flexible experimentation.

## Project Structure

- `main.py`: The entry point for the command-line interface (CLI). It parses arguments and handles the logic for training or evaluation.
- `functions.py`: Contains the core logic for model definition, training loops (`training`), and testing loops (`testing`).
- `bin/`: Directory where trained model weights (`.pt` files) are saved and loaded from.

## Usage

The script is run via `main.py` using the `--mode` argument to switch between training and evaluation.

### 1. Training a Model

To train a model, you must set `--mode train` and provide the required learning rate and optimizer.

**Syntax:**
```bash
python main.py --mode train --lr <RATE> --optimizer <OPT> [options]
```

**Required Arguments:**

- `--lr`: Learning rate (float).
- `--optimizer`: The optimizer to use. Options: `SGD` or `NTAvSGD`.

**Optional Arguments:**

- `--weight_tying`: Enable weight tying between embedding and output layers (flag, default: `False`).
- `--var_dropout`: Enable variational dropout (flag, default: `False`).
- `--emb_dropout`: Dropout probability for the embedding layer (default: `0.0`). Only used if `--var_dropout` is enabled.
- `--out_dropout`: Dropout probability for the output layer (default: `0.0`). Only used if `--var_dropout` is enabled.
- `--nt_interval`: NT-AvSGD interval for non-monotonicity check (required if using `NTAvSGD` optimizer).

**Examples:**

Train with SGD and default settings:
```bash
python main.py --mode train --lr 0.5 --optimizer SGD
```

Train with weight tying enabled:
```bash
python main.py --mode train --lr 0.5 --optimizer SGD --weight_tying
```

Train with variational dropout:
```bash
python main.py --mode train --lr 0.5 --optimizer SGD --var_dropout --emb_dropout 0.2 --out_dropout 0.2
```

Train with NT-AvSGD optimizer:
```bash
python main.py --mode train --lr 0.5 --optimizer NTAvSGD --nt_interval 3 
```

Train with all advanced features:
```bash
python main.py --mode train --lr 0.5 --optimizer NTAvSGD --nt_interval 3 --weight_tying --var_dropout --emb_dropout 0.2 --out_dropout 0.2
```

### 2. Evaluating a Model

To evaluate a pre-trained model on the test set, set `--mode evaluate` and specify the `model_number`.

**Syntax:**
```bash
python main.py --mode evaluate --model_number <INT>
```

**Model Number Mapping:**

The script expects specific pre-trained files in the `bin/` folder based on the ID provided:

| Model Number | Corresponding File |
|--------------|-------------------|
| `1` | `bin/exp_LSTM.pt` |
| `2` | `bin/exp_LSTM_WT.pt` |
| `3` | `bin/exp_LSTM_WT_VD.pt` |
| `4` | `bin/exp_LSTM_WT_VD_NTAvSGD.pt` |

**Example:**

Evaluate the model stored in `bin/exp_LSTM.pt`:
```bash
python main.py --mode evaluate --model_number 1
```

## Default Configuration

Training loop uses these default parameters defined in `main.py`:

- Architecture: LSTM
- Embedding Size: 350
- Hidden Size: 350
- Embedding Dropout: 0.0
- Output Dropout: 0.0
- Learning Rate: 0.5
- Gradient Clipping: 5
- Epochs: 100
- Patience (Early Stopping): 3 epochs (or 2× `nt_interval` for NT-AvSGD)

## Notes

- When using variational dropout (`--var_dropout`), you must specify the dropout probabilities via `--emb_dropout` and `--out_dropout`.
- When using NT-AvSGD optimizer, the `--nt_interval` parameter is required and controls the non-monotonicity check interval. The patience value is automatically set to twice this interval.
- Weight tying and variational dropout can be combined for improved regularization.