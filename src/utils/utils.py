import json
import re
import os
from collections import Counter

from src.utils.path_utils import get_splinter_dir, get_logs_dir


def add_static_result_to_file(result):
    """Add a static check result to the results file."""
    logs_dir = get_logs_dir()
    # Ensure the directory exists
    os.makedirs(logs_dir, exist_ok=True)
    
    results_file = os.path.join(logs_dir, 'static_checks_results.json')
    
    with open(results_file, 'a', encoding='utf-8') as file:
        file.write('\n')
        json.dump(result, file, indent='\t')


def get_reductions_map_from_file():
    """Load reductions map from file."""
    splinter_dir = get_splinter_dir()
    file_path = os.path.join(splinter_dir, 'reductions_map.json')
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # Convert string keys to int where possible
    data = {int(key) if key.isdigit() else key: value for key, value in data.items()}
    return data


def get_new_unicode_chars_map_from_file():
    """Load new unicode chars map from file."""
    splinter_dir = get_splinter_dir()
    file_path = os.path.join(splinter_dir, 'new_unicode_chars.json')
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def get_new_unicode_chars_inverted_map_from_file():
    """Load inverted new unicode chars map from file."""
    splinter_dir = get_splinter_dir()
    file_path = os.path.join(splinter_dir, 'new_unicode_chars_inverted.json')
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def get_permutation(word, position, word_length=None):
    """
    Get word after removing character at given position.
    
    Args:
        word: Original word
        position: Position to remove (can be negative)
        word_length: Length of word (optional)
    """
    if word_length is None:
        word_length = len(word)
        
    if position < 0:
        # Handle negative indexing (from the end)
        permutation = word[:word_length + position] + word[(word_length + position + 1):]
    else:
        permutation = word[:position] + word[(position + 1):]
    return permutation


def get_words_dict_by_length(words):
    """
    Group words dictionary by length.
    
    Args:
        words: Dictionary mapping words to frequencies (values can be int or float)
        
    Returns:
        Dictionary mapping length -> {word: frequency}
    """
    words_dict_by_length = {}
    
    if not words:
        return words_dict_by_length
    
    # Find maximum word length
    max_length = 0
    for word in words.keys():
        if len(word) > max_length:
            max_length = len(word)
    
    # Initialize dictionary for each length from 2 to max_length
    for i in range(2, max_length + 1):
        words_dict_by_length[i] = {}
    
    # Fill with words
    for word, freq in words.items():
        word_len = len(word)
        if word_len >= 2:  # Only include words of length 2 or more
            words_dict_by_length[word_len][word] = freq
    
    # Remove empty lengths
    words_dict_by_length = {k: v for k, v in words_dict_by_length.items() if v}
    
    return words_dict_by_length


def get_letters_frequency(words):
    """
    Calculate frequency of each letter in the words dictionary.
    
    Args:
        words: Dictionary mapping words to frequencies
        
    Returns:
        Dictionary mapping letters to frequencies
    """
    letters_frequency = {}
    
    for word, freq in words.items():
        for char in word:
            letters_frequency[char] = letters_frequency.get(char, 0) + freq
    
    # Sort by frequency descending
    letters_frequency = dict(sorted(letters_frequency.items(), key=lambda item: item[1], reverse=True))
    return letters_frequency


def decode_tokens_vocab_file(tokens_vocab_file):
    """Decode token vocabulary file using splinter maps."""
    try:
        vocab_file = f'{tokens_vocab_file}.vocab'
        with open(vocab_file, 'r', encoding='utf-8') as file:
            encoded_text = file.read()
    except FileNotFoundError:
        print(f"Warning: {tokens_vocab_file}.vocab not found")
        return

    new_unicode_chars_inverted_map = get_new_unicode_chars_inverted_map_from_file()
    encoded_tokens = encoded_text.splitlines()
    decoded_tokens = []
    
    for encoded_token in encoded_tokens:
        if not encoded_token.strip():
            continue
        decoded_words_list = decode_sentence(encoded_token, new_unicode_chars_inverted_map)
        decoded_tokens.append("\t".join(decoded_words_list))
    
    if decoded_tokens:
        output_file = f'{tokens_vocab_file}_decoded.vocab'
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write("\n".join(decoded_tokens))


def decode_sentence(text, new_unicode_chars_inverted_map):
    """
    Decode a sentence using the inverted map.
    
    Args:
        text: Encoded text
        new_unicode_chars_inverted_map: Map from encoded chars to reductions
        
    Returns:
        List of decoded words
    """
    decoded_words_list = []
    encoded_words = text.split()
    
    for encoded_word in encoded_words:
        if not encoded_word:
            continue
        decoded_word = []
        for char in encoded_word:
            decoded_char = new_unicode_chars_inverted_map.get(char, char)
            decoded_word.append(decoded_char)
        decoded_words_list.append("".join(decoded_word))
    
    return decoded_words_list


def get_corpus_name(dataset_path, dataset_name):
    """Generate a corpus name from dataset path and name."""
    # Clean the string to be filesystem-friendly
    combined = f'{dataset_path}_{dataset_name}'
    # Replace non-word characters with underscore
    return re.sub(r'\W+', '_', combined)


def timer_func(func):
    """Decorator to time functions."""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f} seconds")
        return result
    return wrapper