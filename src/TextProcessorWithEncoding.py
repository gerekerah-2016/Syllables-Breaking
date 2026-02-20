"""
Text processor with encoding following the Syllable Breaking document.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import re
from src.TextProcessorInterface import TextProcessorInterface
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface


class TextProcessorWithEncoding(TextProcessorInterface):
    """
    Encodes Ethiopic text following the Syllable Breaking algorithm.
    
    This processor takes original Ge'ez text and converts it to a clean format
    with consonant skeletons and reduction tags (e.g., ⟨1⟩, ⟨2⟩) that can
    be tokenized by standard tokenizers. No extra punctuation is added.
    
    Example: ባሱማ → በሰመ⟨1⟩⟨2⟩⟨3⟩
    """
    
    def __init__(self, language_utils: LanguageUtilsInterface, reductions_map, new_unicode_chars_map, new_unicode_chars_inverted_map=None):
        super().__init__(language_utils)
        self.reductions_map = reductions_map
        self.new_unicode_chars_map = new_unicode_chars_map  # Maps reduction keys to tags (e.g., "0:[a]" → "⟨1⟩")
        self.new_unicode_chars_inverted_map = new_unicode_chars_inverted_map  # Maps tags to reduction keys (e.g., "⟨1⟩" → "0:[a]")
        self.word_cache = {}

    def process(self, text):
        """
        Process text with splinter encoding.
        
        Args:
            text: Input text (can be multiple sentences)
            
        Returns:
            Clean encoded text with consonant skeletons and reduction tags
        """
        if text is None:
            return ''

        # Split into sentences
        sentences = re.split(r'[.\n]', text)
        encoded_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            encoded_words = []
            # Split into words - keep only actual words, not punctuation
            # Ge'ez punctuation marks: ፡ ። ፣ ፤ ፥ ፦ ፧ ፨
            words = re.split(r'[\s፡።፣፤፥፦፧፨]+', sentence)
            
            for word in words:
                word = word.strip()
                if not word:
                    continue
                
                # Skip if it's just punctuation or numbers
                if all(not self.language_utils.is_letter_in_language(c) for c in word):
                    continue
                    
                # Use cache for efficiency
                if word in self.word_cache:
                    encoded_words.append(self.word_cache[word])
                else:
                    encoded_word = self.encode_word(word)
                    self.word_cache[word] = encoded_word
                    encoded_words.append(encoded_word)
            
            if encoded_words:
                # Join words with spaces (no punctuation)
                encoded_sentences.append(" ".join(encoded_words))
        
        return "\n".join(encoded_sentences)

    def encode_word(self, word):
        """
        Encode a single word following the splinter algorithm.
        
        This method processes a single word, extracting only the consonants
        and their vowel tags. No punctuation is included.
        
        Args:
            word: Original Ge'ez word (e.g., "ባሱማ")
            
        Returns:
            Clean encoded word: consonant skeleton + reduction tags (e.g., "በሰመ⟨1⟩⟨2⟩⟨3⟩")
        """
        # Handle words with foreign characters
        if hasattr(self.language_utils, 'is_word_contains_letters_from_other_languages') and \
           self.language_utils.is_word_contains_letters_from_other_languages(word):
            return self.handle_mixed_word(word)
        
        # Step 1: Syllable breaking
        # This returns a string like "በ[a]ሰ[u]መ[a]"
        broken = self.language_utils.syllable_break(word)
        
        # Step 2: Extract consonant skeleton and collect reduction keys
        consonants = []
        reduction_keys = []
        
        i = 0
        length = len(broken)
        while i < length:
            char = broken[i]
            
            # Check if this is a consonant (Ethiopic letter)
            if self.language_utils.is_letter_in_language(char):
                consonants.append(char)
                curr_cons_idx = len(consonants) - 1
                
                # Look ahead to see if there's a vowel marker
                # Vowel markers start with '[' and end with ']'
                if i + 1 < length and broken[i+1] == '[':
                    # Find the closing bracket
                    end_idx = broken.find(']', i+1)
                    if end_idx != -1:
                        # Extract the FULL vowel marker including brackets
                        # This gives "[a]" not just "["
                        vowel_marker = broken[i+1:end_idx+1]
                        
                        # Create reduction key: "consonant_index:vowel_marker"
                        # Example: "0:[a]" for first consonant with [a] vowel
                        key = f"{curr_cons_idx}:{vowel_marker}"
                        
                        # Use the encode map to get the corresponding tag (e.g., "⟨1⟩")
                        if key in self.new_unicode_chars_map:
                            reduction_keys.append(self.new_unicode_chars_map[key])
                        # If no mapping, we skip this vowel (rare patterns)
                        
                        # Skip ahead past the vowel marker
                        i = end_idx + 1
                    else:
                        i += 1
                else:
                    i += 1
            else:
                # Skip non-letter characters (should not happen in a clean word)
                i += 1
        
        # Step 3: Combine clean consonant skeleton and tags
        skeleton = "".join(consonants)
        tags = "".join(reduction_keys)
        
        return skeleton + tags

    def decode_word(self, encoded_word):
        """
        Decode an encoded word back to original Ge'ez.
        
        This reverses the encoding process:
        1. Extracts consonant skeleton and tags
        2. Uses decode map to convert tags back to reduction keys
        3. Applies vowels to consonants to reconstruct original word
        
        Args:
            encoded_word: Encoded word (e.g., "በሰመ⟨1⟩⟨2⟩⟨3⟩")
            
        Returns:
            Original Ge'ez word (e.g., "ባሱማ")
        """
        import re
        
        if not self.new_unicode_chars_inverted_map:
            return encoded_word
        
        # Extract consonant skeleton (all Ethiopic letters)
        skeleton_chars = []
        for char in encoded_word:
            if self.language_utils.is_letter_in_language(char):
                skeleton_chars.append(char)
        skeleton = ''.join(skeleton_chars)
        
        # Extract all tags (patterns like ⟨1⟩, ⟨2⟩, etc.)
        tags = re.findall(r'⟨\d+⟩', encoded_word)
        
        # Use language utils to decode
        return self.language_utils.decode_from_splinter(
            skeleton, tags, self.new_unicode_chars_inverted_map
        )

    def handle_mixed_word(self, word):
        """
        Handle words containing characters from other languages.
        
        For mixed words, we try to encode Ethiopic letters and leave
        foreign characters as is.
        
        Args:
            word: Word containing mixed language characters
            
        Returns:
            Encoded version with Ethiopic parts processed
        """
        encoded = []
        for char in word:
            if self.language_utils.is_letter_in_language(char):
                # This is an Ethiopic character, try to map it
                # For single characters, we might have a direct mapping
                if char in self.new_unicode_chars_map:
                    encoded.append(self.new_unicode_chars_map[char])
                else:
                    encoded.append(char)
            else:
                # Foreign character - leave as is
                encoded.append(char)
        return ''.join(encoded)

    def get_skeleton(self, word):
        """
        Get just the consonant skeleton of a word (without tags).
        
        Args:
            word: Original Ge'ez word
            
        Returns:
            Consonant skeleton (e.g., "ባሱማ" → "በሰመ")
        """
        encoded = self.encode_word(word)
        # Extract only the consonant part (before any ⟨tag⟩)
        import re
        match = re.match(r'([ሀ-ፐ]+)', encoded)
        if match:
            return match.group(1)
        return encoded

    def get_tags(self, word):
        """
        Get just the tags for a word (without skeleton).
        
        Args:
            word: Original Ge'ez word
            
        Returns:
            Tags string (e.g., "⟨1⟩⟨2⟩⟨3⟩")
        """
        encoded = self.encode_word(word)
        # Extract only the tags
        import re
        tags = re.findall(r'⟨\d+⟩', encoded)
        return ''.join(tags)