import re
from src.TextProcessorInterface import TextProcessorInterface
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface


class TextProcessorBaseline(TextProcessorInterface):
    """Baseline processor that only normalizes text."""

    def __init__(self, language_utils: LanguageUtilsInterface):
        super().__init__(language_utils)

    def process(self, text):
        if text is None:
            return ''

        clean_text = self.language_utils.remove_diacritics(text)
        sentences = re.split(r'[.\n]', clean_text)
        processed_sentences = []
        
        for sentence in sentences:
            words = re.split(r'\s|-|,|:|"|\(|\)', sentence)
            filtered_words = [w for w in words if w]
            processed_words = [self.language_utils.replace_final_letters(w) for w in filtered_words]
            if processed_words:
                processed_sentences.append(" ".join(processed_words))
        
        return "\n".join(processed_sentences)