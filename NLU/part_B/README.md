# NLU Project: BERT-based Intent and Slot Classification

This repository contains scripts to train and evaluate BERT models for intent classification and slot filling tasks. The project leverages pre-trained BERT embeddings with configurable dropout and optimization strategies.

## Project Structure

- `main.py`: The entry point for the command-line interface (CLI). It parses arguments and handles the logic for training or evaluation.
- `functions.py`: Contains the core logic for model definition, training loops (`training`), and testing loops (`testing`).
- `utils.py`: Utility functions for data processing and helper methods.
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
- `--optimizer`: The optimizer to use. Options: `SGD`, `AdamW`, or `Adam`.

**Optional Arguments:**

- `--dropout_prob`: Dropout probability (default: `0.0`).

**Examples:**

Train with default settings (no dropout, Adam optimizer):
```bash
python main.py --mode train --lr 0.00005 --optimizer Adam
```

Train with dropout enabled:
```bash
python main.py --mode train --lr 0.00005 --optimizer Adam --dropout_prob 0.3
```

Train with AdamW optimizer:
```bash
python main.py --mode train --lr 0.00005 --optimizer AdamW
```

Train with SGD optimizer and dropout:
```bash
python main.py --mode train --lr 0.0001 --optimizer SGD --dropout_prob 0.2
```

### 2. Evaluating a Model

To evaluate a pre-trained model on the test set, set `--mode evaluate`.

**Syntax:**
```bash
python main.py --mode evaluate
```

**Note:** The evaluation mode uses a hardcoded model path: `bin/bert_lr5e-05_AdamNoDO.pt`. To evaluate a different model, modify the `path` variable in the `main.py` script.

**Example:**

Evaluate the trained model:
```bash
python main.py --mode evaluate
```

## Default Configuration

Training loop uses these default parameters defined in `main.py`:

- Learning Rate: 0.0001
- Optimizer: Adam
- Dropout Probability: 0.0
- Gradient Clipping: 5
- Epochs: 50
- Patience (Early Stopping): 3 epochs

## Notes

- The model uses pre-trained BERT embeddings as the base architecture.
- Learning rates for BERT-based models are typically much smaller than traditional RNNs (e.g., 5e-5 to 1e-4).
- When using dropout (`--dropout_prob > 0`), the model applies dropout to prevent overfitting.
- Model files are automatically named based on the configuration used during training (e.g., `bert_lr5e-05_AdamNoDO.pt`).
- The optimizer must be one of `SGD`, `AdamW`, or `Adam`. The script will exit with an error if an invalid optimizer is provided.