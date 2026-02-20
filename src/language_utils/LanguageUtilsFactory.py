from src.language_utils.EthiopicUtils import EthiopicUtils

class LanguageUtilsFactory:
    @staticmethod
    def get_by_language(language: str):
        languages = {
            
            
            'ge': EthiopicUtils(),   # Ge'ez
            'am': EthiopicUtils(),   # Amharic (can have subclass if needed)
            'ti': EthiopicUtils(),   # Tigrinya (can have subclass if needed)
        }
        return languages.get(language.lower())