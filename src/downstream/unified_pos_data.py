"""
Unified POS Data Loader for Amharic/Ge'ez
"""

import os
import requests
import conllu
import random
from pathlib import Path
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedPOSDataLoader:
    """Load POS data from multiple sources"""
    
    def __init__(self):
        self.logger = logger
        self.all_sentences = []
        
    def load_ud_amharic(self):
        """Load UD Amharic dataset"""
        self.logger.info("Loading UD Amharic...")
        base_url = "https://raw.githubusercontent.com/UniversalDependencies/UD_Amharic-ATT/master"
        
        files = {
            'train': f"{base_url}/am_att-ud-train.conllu",
            'dev': f"{base_url}/am_att-ud-dev.conllu",
            'test': f"{base_url}/am_att-ud-test.conllu"
        }
        
        sentences = []
        for split, url in files.items():
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    content = response.text
                    if content.strip():
                        parsed = list(conllu.parse(content))
                        sentences.extend(parsed)
                        self.logger.info(f"  Added {len(parsed)} from UD {split}")
            except Exception as e:
                self.logger.warning(f"  Failed to load {url}: {e}")
        
        return sentences
    
    def load_masakhaner_as_pos(self):
        """Convert MasakhaNER to POS-like format"""
        self.logger.info("Loading MasakhaNER...")
        
        urls = {
            "train": "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/amh/train.txt",
            "dev": "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/amh/dev.txt",
            "test": "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/amh/test.txt"
        }
        
        sentences = []
        for split, url in urls.items():
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    lines = response.text.splitlines()
                    tokens = []
                    for line in lines:
                        line = line.strip()
                        if line == "":
                            if tokens:
                                # Convert NER to POS
                                pos_tags = []
                                for token, tag in tokens:
                                    if tag == "O":
                                        pos_tags.append("NOUN")
                                    elif "PER" in tag:
                                        pos_tags.append("PROPN")
                                    elif "LOC" in tag:
                                        pos_tags.append("PROPN")
                                    elif "ORG" in tag:
                                        pos_tags.append("PROPN")
                                    elif "DATE" in tag:
                                        pos_tags.append("NUM")
                                    else:
                                        pos_tags.append("NOUN")
                                
                                # Create CONLL-U format
                                conllu_lines = []
                                for i, (token, _) in enumerate(tokens, 1):
                                    conllu_lines.append(f"{i}\t{token}\t_\t{pos_tags[i-1]}\t_\t_\t_\t_\t_\t_")
                                conllu_text = "\n".join(conllu_lines) + "\n\n"
                                
                                try:
                                    parsed = list(conllu.parse(conllu_text))
                                    sentences.extend(parsed)
                                except:
                                    pass
                                
                                tokens = []
                        else:
                            parts = line.split()
                            if len(parts) >= 2:
                                tokens.append((parts[0], parts[-1]))
            except Exception as e:
                self.logger.warning(f"  Failed to load {url}: {e}")
        
        self.logger.info(f"  Added {len(sentences)} from MasakhaNER")
        return sentences
    
    def load_geez_corpus(self, corpus_path="./Geez-Dataset-Clean"):
        """Load your Ge'ez corpus for unsupervised pre-training"""
        self.logger.info(f"Loading Ge'ez corpus from {corpus_path}...")
        
        texts = []
        if not os.path.exists(corpus_path):
            self.logger.warning(f"  Corpus path not found: {corpus_path}")
            return texts
            
        for root, dirs, files in os.walk(corpus_path):
            for file in files:
                if file.endswith('.txt'):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            texts.append(f.read())
                    except Exception as e:
                        self.logger.warning(f"  Error reading {file}: {e}")
        
        self.logger.info(f"  Added {len(texts)} files from Ge'ez corpus")
        return texts
    
    def create_pos_dataset(self, output_dir="./combined_pos_data"):
        """Create combined POS dataset from all sources"""
        
        # Load all POS-annotated data
        ud_sentences = self.load_ud_amharic()
        masakhaner_sentences = self.load_masakhaner_as_pos()
        
        all_pos_sentences = ud_sentences + masakhaner_sentences
        
        if not all_pos_sentences:
            self.logger.warning("No POS sentences loaded! Creating synthetic data...")
            all_pos_sentences = self.create_synthetic_pos_data(100)
        
        # Shuffle
        random.shuffle(all_pos_sentences)
        
        # Split into train/dev/test (80/10/10)
        total = len(all_pos_sentences)
        train_end = int(total * 0.8)
        dev_end = int(total * 0.9)
        
        train = all_pos_sentences[:train_end]
        dev = all_pos_sentences[train_end:dev_end]
        test = all_pos_sentences[dev_end:]
        
        self.logger.info(f"Final POS dataset: {len(train)} train, {len(dev)} dev, {len(test)} test")
        
        # Save to files
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, split_data in [('train', train), ('dev', dev), ('test', test)]:
            output_file = Path(output_dir) / f"{split_name}.conllu"
            with open(output_file, 'w', encoding='utf-8') as f:
                for sent in split_data:
                    f.write(sent.serialize())
            self.logger.info(f"  Saved {len(split_data)} to {output_file}")
        
        return train, dev, test
    
    def create_synthetic_pos_data(self, num_sentences=100):
        """Create synthetic POS data for testing"""
        from conllu import Sentence, TokenList
        
        sentences = []
        
        # Simple templates
        templates = [
            [("ሰላም", "NOUN"), ("ዓለም", "NOUN")],
            [("አዲስ", "ADJ"), ("አበባ", "PROPN")],
            [("ኢትዮጵያ", "PROPN"), ("ቆንጆ", "ADJ"), ("ናት", "VERB")],
            [("ብዙ", "ADV"), ("ሰዎች", "NOUN"), ("መጡ", "VERB")],
            [("እሱ", "PRON"), ("መምህር", "NOUN"), ("ነው", "VERB")],
        ]
        
        for i in range(num_sentences):
            template = random.choice(templates)
            token_list = []
            for j, (form, upos) in enumerate(template, 1):
                token_data = {
                    'id': j,
                    'form': form,
                    'lemma': '_',
                    'upos': upos,
                    'xpos': '_',
                    'feats': '_',
                    'head': 0,
                    'deprel': 'root' if j == len(template) else 'dep',
                    'deps': '_',
                    'misc': '_'
                }
                token_list.append(token_data)
            
            sent = TokenList(token_list)
            sentences.append(sent)
        
        return sentences
    
    def create_pretraining_corpus(self, output_file="./pretrain_corpus.txt"):
        """Create unsupervised pre-training corpus from all sources"""
        
        # Load all text sources
        geez_texts = self.load_geez_corpus()
        
        all_texts = []
        all_texts.extend(geez_texts)
        
        # Also load UD and MasakhaNER for text
        ud_sentences = self.load_ud_amharic()
        for sent in ud_sentences:
            words = [token['form'] for token in sent if hasattr(token, 'form')]
            if words:
                all_texts.append(' '.join(words))
        
        # Write to file
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in all_texts:
                if text and len(text.strip()) > 0:
                    f.write(text.strip() + '\n')
        
        self.logger.info(f"Created pre-training corpus with {len(all_texts)} lines in {output_file}")
        return output_file


if __name__ == "__main__":
    loader = UnifiedPOSDataLoader()
    
    # Create POS dataset
    print("\n" + "="*60)
    print("Creating POS dataset...")
    print("="*60)
    train, dev, test = loader.create_pos_dataset()
    
    # Create pre-training corpus
    print("\n" + "="*60)
    print("Creating pre-training corpus...")
    print("="*60)
    pretrain_file = loader.create_pretraining_corpus()
    
    print(f"\n✅ Done! Check ./combined_pos_data/ and {pretrain_file}")