# Import compatibility module first to patch deprecated Keras modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api.utils import tokenizer_compat

import tensorflow as tf
import pickle
import pystow
import re
import logging
import numpy as np
import signal
import time
from typing import Optional, Tuple, List
from rdkit import Chem
from py2opsin import py2opsin
from api.utils.validation import validate_smiles_format, validate_iupac_format, sanitize_smiles, sanitize_iupac

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        
        # CDK JVM status
        self.cdk_available = False
        self.cdk_initialized = False
        
    def _setup_gpu(self):
        """Setup GPU configuration"""
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU setup completed. Found {len(gpus)} GPU(s)")
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}. Continuing with CPU.")
    
    def _download_models_if_needed(self):
        """Download models if not present"""
        try:
            self.default_path = pystow.join("STOUT-V2", "models")
            model_url = "https://storage.googleapis.com/decimer_weights/models.zip"
            model_path = str(self.default_path) + "/translator_forward/"
            
            if not os.path.exists(model_path):
                logger.info("Downloading STOUT models...")
                from api.utils.helper import download_trained_weights
                download_trained_weights(model_url, self.default_path)
                logger.info("Models downloaded successfully")
            else:
                logger.info("Models already present")
        except Exception as e:
            logger.error(f"Failed to download models: {e}")
            raise
    
    def _initialize_cdk(self):
        """Initialize CDK for SMILES canonicalization"""
        try:
            from jpype import startJVM, getDefaultJVMPath, JClass, JVMNotFoundException, isJVMStarted
            import pystow
            
            if isJVMStarted():
                self.cdk_initialized = True
                self.cdk_available = True
                logger.info("CDK JVM already started")
                return
            
            jvmPath = getDefaultJVMPath()
            cdk_path = "https://github.com/cdk/cdk/releases/download/cdk-2.8/cdk-2.8.jar"
            jar_path = str(pystow.join("STOUT-V2")) + "/cdk-2.8.jar"
            
            if not os.path.exists(jar_path):
                jar_path = pystow.ensure("STOUT-V2", url=cdk_path)
            
            startJVM(jvmPath, "-ea", "-Djava.class.path=" + str(jar_path))
            self.cdk_initialized = True
            self.cdk_available = True
            logger.info("CDK initialized successfully")
        except Exception as e:
            logger.warning(f"CDK initialization failed: {e}. Will use RDKit fallback.")
            self.cdk_available = False
            self.cdk_initialized = False
    
    def load_models(self):
        """Load models and tokenizers - called once on startup"""
        if self.models_loaded:
            return
            
        logger.info("Loading STOUT models...")
        try:
            self._setup_gpu()
            self._download_models_if_needed()
            self._initialize_cdk()
            
            # Load models
            self.reloaded_forward = tf.saved_model.load(self.default_path.as_posix() + "/translator_forward")
            self.reloaded_reverse = tf.saved_model.load(self.default_path.as_posix() + "/translator_reverse")
            
            # Load tokenizers and max lengths
            self.inp_lang = pickle.load(open(self.default_path.as_posix() + "/assets/tokenizer_input.pkl", "rb"))
            self.targ_lang = pickle.load(open(self.default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb"))
            self.inp_max_length = pickle.load(open(self.default_path.as_posix() + "/assets/max_length_inp.pkl", "rb"))
            self.targ_max_length = pickle.load(open(self.default_path.as_posix() + "/assets/max_length_targ.pkl", "rb"))
            
            self.models_loaded = True
            logger.info("STOUT models loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _preprocess_sentence(self, w: str) -> str:
        """Preprocess sentence for tokenization"""
        try:
            import unicodedata
            w = "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")
            w = re.sub(r"([?.!,¿])", r" \1 ", w)
            w = re.sub(r'[" "]+', " ", w)
            w = w.strip()
            w = "<start> " + w + " <end>"
            return w
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def _tokenize_input(self, input_text: str, inp_lang, inp_max_length: int) -> np.array:
        """Tokenize input text"""
        try:
            sentence = self._preprocess_sentence(input_text)
            try:
                inputs = [inp_lang.word_index[i] for i in sentence.split(" ")]
            except Exception as e:
                logger.error(f"Tokenization mapping failed: {e}")
                raise ValueError(f"Tokenization mapping failed: {e}")
            # Use keras.preprocessing.sequence.pad_sequences for compatibility
            import tensorflow.keras as keras
            tokenized_input = keras.preprocessing.sequence.pad_sequences(
                [inputs], maxlen=inp_max_length, padding="post"
            )
            return tokenized_input
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise ValueError(f"Tokenization failed: {e}")
    
    def _detokenize_output(self, predicted_array: np.array, targ_lang) -> str:
        """Detokenize model output"""
        try:
            try:
                outputs = [targ_lang.index_word[i] for i in predicted_array[0].numpy()]
            except Exception as e:
                logger.error(f"Detokenization mapping failed: {e}")
                raise ValueError(f"Detokenization mapping failed: {e}")
            prediction = (
                " ".join([str(elem) for elem in outputs])
                .replace("<start> ", "")
                .replace(" <end>", "")
                .replace(" ", "")
            )
            return prediction
        except Exception as e:
            logger.error(f"Detokenization failed: {e}")
            raise ValueError(f"Detokenization failed: {e}")
    
    def _get_smiles_rdkit(self, smiles: str) -> Optional[str]:
        """Canonicalize SMILES using RDKit as fallback"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                return canonical_smiles
            return None
        except Exception as e:
            logger.warning(f"RDKit canonicalization failed: {e}")
            return None
    
    def _get_smiles_cdk(self, smiles: str) -> Optional[str]:
        """Canonicalize SMILES using CDK with comprehensive error handling"""
        if not self.cdk_available:
            logger.debug("CDK not available, using RDKit fallback")
            return self._get_smiles_rdkit(smiles)
        
        try:
            from jpype import JClass, isJVMStarted
            
            if not isJVMStarted():
                logger.warning("JVM not started, attempting to initialize CDK")
                self._initialize_cdk()
                if not self.cdk_available:
                    return self._get_smiles_rdkit(smiles)
            
            cdk_base = "org.openscience.cdk"
            SCOB = JClass(cdk_base + ".DefaultChemObjectBuilder")
            SmiFlavour = JClass(cdk_base + ".smiles.SmiFlavor")
            SmilesGenerator = JClass(cdk_base + ".smiles.SmilesGenerator")(SmiFlavour.Absolute)
            SmilesParser = JClass(cdk_base + ".smiles.SmilesParser")(SCOB.getInstance())
            
            # Validate input SMILES
            if not smiles or not smiles.strip():
                logger.warning("Empty SMILES input")
                return None
            
            # Try to parse and canonicalize
            molecule = SmilesParser.parseSmiles(smiles)
            if molecule is None:
                logger.warning(f"CDK failed to parse SMILES: {smiles}")
                return self._get_smiles_rdkit(smiles)
            
            CanSMILES = SmilesGenerator.create(molecule)
            if CanSMILES and CanSMILES.strip():
                logger.debug(f"CDK canonicalization successful: {smiles} -> {CanSMILES}")
                return CanSMILES
            else:
                logger.warning(f"CDK returned empty canonical SMILES for: {smiles}")
                return self._get_smiles_rdkit(smiles)
                
        except Exception as e:
            logger.warning(f"CDK canonicalization failed: {e}")
            # Fallback to RDKit
            return self._get_smiles_rdkit(smiles)
    
    def _validate_smiles_input(self, smiles: str) -> Tuple[bool, List[str]]:
        """Validate SMILES input and return validation status and warnings"""
        return validate_smiles_format(smiles)
    
    def translate_forward(self, smiles: str) -> str:
        """SMILES to IUPAC translation with comprehensive error handling"""
        if not self.models_loaded:
            self.load_models()
        is_valid, warnings = self._validate_smiles_input(smiles)
        if not is_valid:
            logger.warning(f"Invalid SMILES input: {smiles}")
            return "Invalid SMILES input provided."
        if warnings:
            logger.info(f"SMILES validation warnings for '{smiles}': {warnings}")
        try:
            smiles = sanitize_smiles(smiles)
            smiles_canon = self._get_smiles_cdk(smiles)
            if not smiles_canon:
                logger.warning(f"Failed to canonicalize SMILES: {smiles}")
                return "Could not canonicalize SMILES input."
            splitted_list = list(smiles_canon)
            tokenized_SMILES = re.sub(r"\s+(?=[a-z])", "", " ".join(map(str, splitted_list)))
            try:
                decoded = self._tokenize_input(tokenized_SMILES, self.inp_lang, self.inp_max_length)
            except Exception as e:
                logger.error(f"Tokenization error in forward translation: {e}")
                return f"Tokenization error: {e}"
            result_predicted = self.reloaded_forward(decoded)
            try:
                result = self._detokenize_output(result_predicted, self.targ_lang)
            except Exception as e:
                logger.error(f"Detokenization error in forward translation: {e}")
                return f"Detokenization error: {e}"
            if not result or not result.strip():
                logger.warning(f"Empty IUPAC output for SMILES: {smiles}")
                return "Translation produced empty result."
            logger.info(f"Successful forward translation: {smiles} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Forward translation failed for '{smiles}': {e}")
            return f"Translation failed: {str(e)}"
    
    def translate_reverse(self, iupacname: str) -> str:
        """IUPAC to SMILES translation with comprehensive error handling"""
        if not self.models_loaded:
            self.load_models()
        is_valid, warnings = validate_iupac_format(iupacname)
        if not is_valid:
            logger.warning(f"Invalid IUPAC input: {iupacname}")
            return "Invalid IUPAC input provided."
        if warnings:
            logger.info(f"IUPAC validation warnings for '{iupacname}': {warnings}")
        iupacname = sanitize_iupac(iupacname)
        try:
            if self.reverse_targ_lang is None:
                self.reverse_targ_lang = pickle.load(open(self.default_path.as_posix() + "/assets/tokenizer_input.pkl", "rb"))
                self.reverse_inp_lang = pickle.load(open(self.default_path.as_posix() + "/assets/tokenizer_target.pkl", "rb"))
                self.reverse_inp_max_length = pickle.load(open(self.default_path.as_posix() + "/assets/max_length_targ.pkl", "rb"))
            splitted_list = list(iupacname)
            tokenized_IUPACname = " ".join(map(str, splitted_list))
            try:
                decoded = self._tokenize_input(tokenized_IUPACname, self.reverse_inp_lang, self.reverse_inp_max_length)
            except Exception as e:
                logger.error(f"Tokenization error in reverse translation: {e}")
                return f"Tokenization error: {e}"
            result_predicted = self.reloaded_reverse(decoded)
            try:
                result = self._detokenize_output(result_predicted, self.reverse_targ_lang)
            except Exception as e:
                logger.error(f"Detokenization error in reverse translation: {e}")
                return f"Detokenization error: {e}"
            if result:
                if "." in result and len(result) > 20:
                    parts = result.split(".")
                    if len(parts) > 1 and all(parts[0] == part for part in parts[1:3]):
                        result = parts[0]
                        logger.debug(f"Removed repetitive patterns from result: {result}")
                smiles_pattern = r'[A-Z][a-z]?[0-9]*[A-Z]?[a-z]?[0-9]*[A-Z]?[a-z]?[0-9]*[A-Z]?[a-z]?[0-9]*[A-Z]?[a-z]?[0-9]*'
                matches = re.findall(smiles_pattern, result)
                if matches:
                    result = matches[0]
                if len(result) < 2 or len(result) > 50:
                    logger.warning(f"Generated SMILES length suspicious: {len(result)} characters")
                    result = ""
                if result and not self.is_valid_smiles(result):
                    logger.warning(f"Generated SMILES failed RDKit validation: {result}")
                    result = ""
            if result:
                logger.info(f"Successful reverse translation: {iupacname} -> {result}")
            else:
                logger.warning(f"Empty or invalid SMILES result for IUPAC: {iupacname}")
            return result
        except Exception as e:
            logger.error(f"Reverse translation failed for '{iupacname}': {e}")
            return f"Translation failed: {str(e)}"
    
    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES is valid using RDKit""" 
        try:
            if not smiles or not smiles.strip():
                return False
            mol = Chem.MolFromSmiles(smiles.strip())
            return mol is not None
        except Exception as e:
            logger.debug(f"SMILES validation error: {e}")
            return False
    
    def iupac_to_smiles_opsin(self, iupac: str) -> str:
        """Convert IUPAC to SMILES using OPSIN with error handling"""
        if not iupac or not iupac.strip():
            return ""
        
        try:
            result = py2opsin(chemical_name=iupac.strip(), output_format="SMILES")
            if result and result.strip():
                logger.info(f"OPSIN successful: {iupac} -> {result}")
                return result.strip()
        except Exception as e:
            logger.warning(f"OPSIN failed for '{iupac}': {e}")
        
        # Fallback to STOUT reverse model
        try:
            result = self.translate_reverse(iupac)
            if result:
                logger.info(f"STOUT fallback successful: {iupac} -> {result}")
            return result
        except Exception as e:
            logger.error(f"STOUT fallback failed for '{iupac}': {e}")
            return ''

# Global service instance
stout_service = StoutService() 