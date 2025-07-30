"""
Compatibility module for old Keras tokenizer pickle files.
This module provides a mock implementation of the deprecated keras.preprocessing.text module.
"""

import collections
import json
import numpy as np
from typing import List, Dict, Any

class MockTokenizer:
    """Mock tokenizer class to replace the deprecated keras.preprocessing.text.Tokenizer"""
    
    def __init__(self, **kwargs):
        self.word_counts = collections.OrderedDict()
        self.word_docs = collections.defaultdict(int)
        self.filters = kwargs.get('filters', '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        self.split = kwargs.get('split', ' ')
        self.lower = kwargs.get('lower', True)
        self.num_words = kwargs.get('num_words', None)
        self.document_count = kwargs.get('document_count', 0)
        self.char_level = kwargs.get('char_level', False)
        self.oov_token = kwargs.get('oov_token', None)
        self.index_docs = collections.defaultdict(int)
        self.word_index = {}
        self.index_word = {}
        self.analyzer = kwargs.get('analyzer', None)
    
    def fit_on_texts(self, texts):
        for text in texts:
            self.document_count += 1
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = self._text_to_word_sequence(text)
            
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            
            for w in set(seq):
                self.word_docs[w] += 1
        
        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)
        
        self.word_index = dict(zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))
        self.index_word = {c: w for w, c in self.word_index.items()}
        
        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c
    
    def _text_to_word_sequence(self, text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" "):
        if lower:
            text = text.lower()
        
        translate_dict = {c: split for c in filters}
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)
        
        seq = text.split(split)
        return [i for i in seq if i]
    
    def texts_to_sequences(self, texts):
        return list(self.texts_to_sequences_generator(texts))
    
    def texts_to_sequences_generator(self, texts):
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        
        for text in texts:
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = self._text_to_word_sequence(text, self.filters, self.lower, self.split)
            
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect
    
    def sequences_to_texts(self, sequences):
        return list(self.sequences_to_texts_generator(sequences))
    
    def sequences_to_texts_generator(self, sequences):
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        
        for seq in sequences:
            vect = []
            for num in seq:
                word = self.index_word.get(num)
                if word is not None:
                    if num_words and num >= num_words:
                        if oov_token_index is not None:
                            vect.append(self.index_word[oov_token_index])
                    else:
                        vect.append(word)
                elif self.oov_token is not None:
                    vect.append(self.index_word[oov_token_index])
            vect = " ".join(vect)
            yield vect
    
    def get_config(self):
        json_word_counts = json.dumps(self.word_counts)
        json_word_docs = json.dumps(self.word_docs)
        json_index_docs = json.dumps(self.index_docs)
        json_word_index = json.dumps(self.word_index)
        json_index_word = json.dumps(self.index_word)
        
        return {
            "num_words": self.num_words,
            "filters": self.filters,
            "lower": self.lower,
            "split": self.split,
            "char_level": self.char_level,
            "oov_token": self.oov_token,
            "document_count": self.document_count,
            "word_counts": json_word_counts,
            "word_docs": json_word_docs,
            "index_docs": json_index_docs,
            "index_word": json_index_word,
            "word_index": json_word_index,
        }
    
    def to_json(self, **kwargs):
        config = self.get_config()
        tokenizer_config = {
            "class_name": self.__class__.__name__,
            "config": config,
        }
        return json.dumps(tokenizer_config, **kwargs)

# Mock module to replace keras.preprocessing.text
class MockKerasPreprocessingText:
    """Mock module to replace keras.preprocessing.text"""
    
    @staticmethod
    def text_to_word_sequence(input_text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" "):
        if lower:
            input_text = input_text.lower()
        
        translate_dict = {c: split for c in filters}
        translate_map = str.maketrans(translate_dict)
        input_text = input_text.translate(translate_map)
        
        seq = input_text.split(split)
        return [i for i in seq if i]
    
    @staticmethod
    def one_hot(input_text, n, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ", analyzer=None):
        return MockKerasPreprocessingText.hashing_trick(input_text, n, hash_function=hash, filters=filters, lower=lower, split=split, analyzer=analyzer)
    
    @staticmethod
    def hashing_trick(text, n, hash_function=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ", analyzer=None):
        if hash_function is None:
            hash_function = hash
        elif hash_function == "md5":
            import hashlib
            def hash_function(w):
                return int(hashlib.md5(w.encode()).hexdigest(), 16)
        
        if analyzer is None:
            seq = MockKerasPreprocessingText.text_to_word_sequence(text, filters=filters, lower=lower, split=split)
        else:
            seq = analyzer(text)
        
        return [(hash_function(w) % (n - 1) + 1) for w in seq]
    
    Tokenizer = MockTokenizer
    
    @staticmethod
    def tokenizer_from_json(json_string):
        import json
        tokenizer_config = json.loads(json_string)
        config = tokenizer_config.get("config")
        
        word_counts = json.loads(config.pop("word_counts"))
        word_docs = json.loads(config.pop("word_docs"))
        index_docs = json.loads(config.pop("index_docs"))
        index_docs = {int(k): v for k, v in index_docs.items()}
        index_word = json.loads(config.pop("index_word"))
        index_word = {int(k): v for k, v in index_word.items()}
        word_index = json.loads(config.pop("word_index"))
        
        tokenizer = MockTokenizer(**config)
        tokenizer.word_counts = word_counts
        tokenizer.word_docs = word_docs
        tokenizer.index_docs = index_docs
        tokenizer.word_index = word_index
        tokenizer.index_word = index_word
        return tokenizer

# Mock sequence module
class MockKerasPreprocessingSequence:
    """Mock module to replace keras.preprocessing.sequence"""
    
    @staticmethod
    def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
        if not sequences:
            return np.array([])
        
        if not isinstance(sequences, (list, tuple)):
            sequences = [sequences]
        
        lengths = [len(x) for x in sequences]
        
        if maxlen is None:
            maxlen = max(lengths)
        
        # Create the padded array
        padded_sequences = np.full((len(sequences), maxlen), value, dtype=dtype)
        
        for idx, seq in enumerate(sequences):
            if len(seq) == 0:
                continue
            
            if truncating == 'pre':
                trunc = seq[-maxlen:]
            elif truncating == 'post':
                trunc = seq[:maxlen]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)
            
            if padding == 'post':
                padded_sequences[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                padded_sequences[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)
        
        return padded_sequences

# Patch the sys.modules to mock the deprecated modules
import sys

# Mock the deprecated modules
sys.modules['keras.preprocessing.text'] = MockKerasPreprocessingText()
sys.modules['keras.preprocessing.sequence'] = MockKerasPreprocessingSequence() 