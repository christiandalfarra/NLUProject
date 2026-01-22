from transformers import BertTokenizerFast

# 1. Initialize the tokenizer (use the "Fast" version for offset mapping)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# 2. Your original text and split
text = "verylongword"
# We split by space to get the "original split" you asked for
original_words = text.split(" ")

# 3. Tokenize with offset mapping
# is_split_into_words=True tells the tokenizer we are providing a list of words,
# preventing it from messing up our specific whitespace alignment.
encoding = tokenizer(
    original_words, 
    is_split_into_words=True, 
    return_offsets_mapping=True
)

# 4. Extract data
tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
word_ids = encoding.word_ids()  # This provides the direct mapping!

print(f"{'TOKEN':<12} | {'ORIGINAL WORD INDEX':<20} | {'ORIGINAL WORD'}")
print("-" * 50)

for token, word_idx in zip(tokens, word_ids):
    # Special tokens (CLS, SEP) return None for word_id
    if word_idx is None:
        original_word = "[Special Token]"
    else:
        original_word = original_words[word_idx]
        
    print(f"{token:<12} | {str(word_idx):<20} | {original_word}")