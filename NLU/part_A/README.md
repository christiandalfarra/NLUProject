# NLU Project: LSTM-based Intent and Slot Classification

This repository contains scripts to train and evaluate Long Short-Term Memory (LSTM) models for intent classification and slot filling tasks. The project supports both unidirectional and bidirectional architectures with configurable dropout and optimization strategies.

## Project Structure

- `main.py`: The entry point for the command-line interface (CLI). It parses arguments and handles the logic for training or evaluation.
- `functions.py`: Contains the core logic for model definition, training loops (`training`), and testing loops (`testing`).
- `utils.py`: Utility functions for data processing and helper methods.
- `bin/`: Directory where trained model weights (`.pt` files) are saved and loaded from.

## Usage

The script is run via `main.py` using the `--mode` argument to switch between training and evaluation.

### 1. Training a Model

To train a model, you must set `--mode train` and provide the required learning rate.

**Syntax:**
```bash
python main.py --mode train --lr <RATE> [options]
```

**Required Arguments:**

- `--lr`: Learning rate (float).

**Optional Arguments:**

- `--model_arch`: Model architecture to use (default: `LSTM`). Currently supports: `LSTM`.
- `--bidirectional`: Enable bidirectional LSTM (flag, default: `False`).
- `--optimizer`: The optimizer to use (default: `Adam`). Options: `SGD`, `AdamW`, or `Adam`.
- `--dropout_prob`: Dropout probability (default: `0.0`).
- `--emb_size`: Embedding layer size (default: `300`).
- `--hidden_size`: Hidden layer size (default: `200`).
- `--multiple_runs`: Number of training runs with different random initializations (default: `5`).

**Examples:**

Train with default settings (unidirectional LSTM, no dropout, Adam optimizer):
```bash
python main.py --mode train --lr 0.0001
```

Train with bidirectional LSTM:
```bash
python main.py --mode train --lr 0.0001 --bidirectional
```

Train with dropout enabled:
```bash
python main.py --mode train --lr 0.0001 --dropout_prob 0.3
```

Train with custom architecture sizes:
```bash
python main.py --mode train --lr 0.0001 --emb_size 400 --hidden_size 256
```

Train with AdamW optimizer and multiple runs:
```bash
python main.py --mode train --lr 0.0001 --optimizer AdamW --multiple_runs 10
```

Train with all advanced features:
```bash
python main.py --mode train --lr 0.0001 --bidirectional --dropout_prob 0.3 --optimizer AdamW --emb_size 400 --hidden_size 256
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
| `1` | `bin/LSTM_UniDir_NoDO_lr0.0001_Adam.pt` |
| `2` | `bin/LSTM_BiDir_NoDO_lr0.0001_Adam.pt` |
| `3` | `bin/LSTM_BiDir_DO_lr0.0001_Adam.pt` |

**Example:**

Evaluate the unidirectional model without dropout:
```bash
python main.py --mode evaluate --model_number 1
```

Evaluate the bidirectional model with dropout:
```bash
python main.py --mode evaluate --model_number 3
```

## Default Configuration

Training loop uses these default parameters defined in `main.py`:

- Model Architecture: LSTM
- Bidirectional: False (unidirectional)
- Embedding Size: 300
- Hidden Size: 200
- Dropout Probability: 0.0
- Learning Rate: 0.0001
- Optimizer: Adam
- Gradient Clipping: 5
- Epochs: 200
- Patience (Early Stopping): 3 epochs
- Multiple Runs: 5

## Notes

- The `--multiple_runs` parameter allows training multiple models with different random seeds to assess model stability and variance.
- Bidirectional LSTMs typically provide better performance but require more computational resources.
- When using dropout (`--dropout_prob > 0`), the model applies dropout to prevent overfitting.
- Model files are automatically named based on the configuration used during training, making it easy to identify different experimental setups.