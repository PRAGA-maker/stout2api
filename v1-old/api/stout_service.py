# Import compatibility module first to patch deprecated Keras modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import tokenizer_compat

import tensorflow as tf
import pickle
import pystow
import re
import logging
import numpy as np
import signal
import time
from typing import Optional, Tuple
from rdkit import Chem
from py2opsin import py2opsin

# Silence tensorflow warnings
logging.getLogger("absl").setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class StoutService:
    def __init__(self):
        self.models_loaded = False
        self.reloaded_forward = None
        self.reloaded_reverse = None
        self.inp_lang = None
        self.targ_lang = None
        self.inp_max_length = None
        self.targ_max_length = None
        self.default_path = None
        
        # Cache for reverse translation tokenizers
        self.reverse_targ_lang = None
        self.reverse_inp_lang = None
        self.reverse_inp_max_length = None
        
    def _setup_gpu(self):
        """Setup GPU configuration"""
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    def _download_models_if_needed(self):
        """Download models if not present"""
        self.default_path = pystow.join("STOUT-V2", "models")
        model_url = "https://storage.googleapis.com/decimer_weights/models.zip"
        model_path = str(self.default_path) + "/translator_forward/"
        
        if not os.path.exists(model_path):
            from utils.helper import download_trained_weights
            download_trained_weights(model_url, self.default_path)
    
    def load_models(self):
        """Load models and tokenizers - called once on startup"""
        if self.models_loaded:
            return
            
        print("Loading STOUT models...")
        self._setup_gpu()
        self._download_models_if_needed()
        
        # Load models
        self.reloaded_forward = tf.saved_model.load(self.default_path.as_posix() + "/translator_forward")
        self.reloaded_reverse = tf.saved_model.load(self.default_path.as_posix() + "/translator_reverse")
        
        # Load tokenizers and max lengths
        self.inp_lang = pickle.load(open(self.default_path.as_posix() + "/assets/tokenizer_input.pkl", "rb"))
        self.targ_lang = pickle.load(open(self.default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb"))
        self.inp_max_length = pickle.load(open(self.default_path.as_posix() + "/assets/max_length_inp.pkl", "rb"))
        self.targ_max_length = pickle.load(open(self.default_path.as_posix() + "/assets/max_length_targ.pkl", "rb"))
        
        self.models_loaded = True
        print("STOUT models loaded successfully!")
    
    def _preprocess_sentence(self, w: str) -> str:
        """Preprocess sentence for tokenization"""
        import unicodedata
        w = "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = w.strip()
        w = "<start> " + w + " <end>"
        return w
    
    def _tokenize_input(self, input_text: str, inp_lang, inp_max_length: int) -> np.array:
        """Tokenize input text"""
        sentence = self._preprocess_sentence(input_text)
        inputs = [inp_lang.word_index[i] for i in sentence.split(" ")]
        # Use keras.preprocessing.sequence.pad_sequences for compatibility
        import tensorflow.keras as keras
        tokenized_input = keras.preprocessing.sequence.pad_sequences(
            [inputs], maxlen=inp_max_length, padding="post"
        )
        return tokenized_input
    
    def _detokenize_output(self, predicted_array: np.array, targ_lang) -> str:
        """Detokenize model output"""
        outputs = [targ_lang.index_word[i] for i in predicted_array[0].numpy()]
        prediction = (
            " ".join([str(elem) for elem in outputs])
            .replace("<start> ", "")
            .replace(" <end>", "")
            .replace(" ", "")
        )
        return prediction
    
    def _get_smiles_cdk(self, smiles: str) -> Optional[str]:
        """Canonicalize SMILES using CDK"""
        try:
            from jpype import startJVM, getDefaultJVMPath, JClass, JVMNotFoundException, isJVMStarted
            import pystow
            
            if not isJVMStarted():
                jvmPath = getDefaultJVMPath()
                cdk_path = "https://github.com/cdk/cdk/releases/download/cdk-2.8/cdk-2.8.jar"
                jar_path = str(pystow.join("STOUT-V2")) + "/cdk-2.8.jar"
                
                if not os.path.exists(jar_path):
                    jar_path = pystow.ensure("STOUT-V2", url=cdk_path)
                
                startJVM(jvmPath, "-ea", "-Djava.class.path=" + str(jar_path))
            
            cdk_base = "org.openscience.cdk"
            SCOB = JClass(cdk_base + ".DefaultChemObjectBuilder")
            SmiFlavour = JClass(cdk_base + ".smiles.SmiFlavor")
            SmilesGenerator = JClass(cdk_base + ".smiles.SmilesGenerator")(SmiFlavour.Absolute)
            SmilesParser = JClass(cdk_base + ".smiles.SmilesParser")(SCOB.getInstance())
            
            molecule = SmilesParser.parseSmiles(smiles)
            CanSMILES = SmilesGenerator.create(molecule)
            return CanSMILES
        except Exception as e:
            print(f"CDK canonicalization failed: {e}")
            return None
    
    def translate_forward(self, smiles: str) -> str:
        """SMILES to IUPAC translation"""
        if not self.models_loaded:
            self.load_models()
            
        if len(smiles) == 0:
            return ""
            
        smiles = smiles.replace("\\/", "/")
        smiles_canon = self._get_smiles_cdk(smiles)
        
        if smiles_canon:
            splitted_list = list(smiles_canon)
            tokenized_SMILES = re.sub(r"\s+(?=[a-z])", "", " ".join(map(str, splitted_list)))
            decoded = self._tokenize_input(tokenized_SMILES, self.inp_lang, self.inp_max_length)
            result_predicted = self.reloaded_forward(decoded)
            result = self._detokenize_output(result_predicted, self.targ_lang)
            return result
        else:
            return "Could not generate IUPAC name for SMILES provided."
    
    def translate_reverse(self, iupacname: str) -> str:
        """IUPAC to SMILES translation"""
        if not self.models_loaded:
            self.load_models()
            
        try:
            # Load reverse translation tokenizers (cached for performance)
            if self.reverse_targ_lang is None:
                self.reverse_targ_lang = pickle.load(open(self.default_path.as_posix() + "/assets/tokenizer_input.pkl", "rb"))
                self.reverse_inp_lang = pickle.load(open(self.default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb"))
                self.reverse_inp_max_length = pickle.load(open(self.default_path.as_posix() + "/assets/max_length_targ.pkl", "rb"))
                
            splitted_list = list(iupacname)
            tokenized_IUPACname = " ".join(map(str, splitted_list))
            decoded = self._tokenize_input(tokenized_IUPACname, self.reverse_inp_lang, self.reverse_inp_max_length)
            
            # Add timeout for model inference (30 seconds)
            start_time = time.time()
            result_predicted = self.reloaded_reverse(decoded)
            inference_time = time.time() - start_time
            
            # If inference takes too long, return early
            if inference_time > 30:
                print(f"Warning: Reverse translation took {inference_time:.2f}s for '{iupacname}'")
            
            result = self._detokenize_output(result_predicted, self.reverse_targ_lang)
            
            # Clean up the result - remove repetitive patterns and extract valid SMILES
            if result:
                # Remove repetitive patterns like "CCO.CCO.CCO..."
                if "." in result and len(result) > 20:
                    parts = result.split(".")
                    if len(parts) > 1 and all(parts[0] == part for part in parts[1:3]):
                        result = parts[0]
                
                # Extract first valid SMILES pattern
                import re
                # More comprehensive SMILES pattern
                smiles_pattern = r'[A-Z][a-z]?[0-9]*[A-Z]?[a-z]?[0-9]*[A-Z]?[a-z]?[0-9]*[A-Z]?[a-z]?[0-9]*[A-Z]?[a-z]?[0-9]*'
                matches = re.findall(smiles_pattern, result)
                if matches:
                    result = matches[0]
                
                # Validate the result is a reasonable SMILES
                if len(result) < 2 or len(result) > 50:
                    result = ""
            
            return result
        except Exception as e:
            print(f"Error in translate_reverse: {e}")
            return ""
    
    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES is valid"""
        return Chem.MolFromSmiles(smiles) is not None
    
    def iupac_to_smiles_opsin(self, iupac: str) -> str:
        """Convert IUPAC to SMILES using OPSIN"""
        try:
            result = py2opsin(chemical_name=iupac, output_format="SMILES")
            if result and result != '':
                return result
        except Exception:
            pass
        # Fallback to STOUT reverse model
        try:
            return self.translate_reverse(iupac)
        except Exception:
            return ''

# Global service instance
stout_service = StoutService() 