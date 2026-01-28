from transformers import BertTokenizer

def tokenize_with_bert(sentence):
    # 1. Load the pre-trained tokenizer
    # We use 'bert-base-uncased' as requested
    print(f"Loading tokenizer for 'bert-base-uncased'...\n")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # --- Method A: View the raw tokens ---
    # This shows how BERT breaks down words (WordPiece algorithm)
    raw_tokens = tokenizer.tokenize(sentence)
    
    print(f"Original Sentence: \"{sentence}\"")
    print(f"Raw Tokens: {raw_tokens}")
    print("-" * 40)

    # --- Method B: Prepare input for the model ---
    # This adds special tokens ([CLS], [SEP]), pads/truncates, 
    # and converts tokens to their integer IDs.
    inputs = tokenizer(
        sentence,
        padding=True,       # Pad to the max length (if processing batches)
        truncation=True,    # Truncate if sentence is longer than model max length
        return_tensors="pt" # Return PyTorch tensors (use "tf" for TensorFlow)
    )

    print("Model Inputs (encoded):")
    for key, value in inputs.items():
        print(f"{key}: {value}")

    # --- Verification: Decode back to text ---
    # We take the input_ids and convert them back to a string to see 
    # what the model actually 'sees' (notice the added special tokens)
    decoded_text = tokenizer.decode(inputs['input_ids'][0])
    print("-" * 40)
    print(f"Decoded (what the model sees): {decoded_text}")

if __name__ == "__main__":
    # Example sentence with a slightly complex word ('tokenization') 
    # to demonstrate subword splitting
    sample_sentence = "Tokenization is essential for BERT models."
    
    tokenize_with_bert(sample_sentence)