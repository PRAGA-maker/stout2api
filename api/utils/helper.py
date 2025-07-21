import os
import tensorflow as tf
import re
import unicodedata
import numpy as np
import pystow
import zipfile
from jpype import startJVM, getDefaultJVMPath
from jpype import JClass, JVMNotFoundException, isJVMStarted

# Start JVM to use CDK in python
try:
    jvmPath = getDefaultJVMPath()
except JVMNotFoundException:
    print(
        "If you see this message, for some reason JPype",
        "cannot find jvm.dll.",
        "This indicates that the environment varibale JAVA_HOME",
        "is not set properly.",
        "You can set it or set it manually in the code",
    )
    jvmPath = "Define/path/or/set/JAVA_HOME/variable/properly"

if not isJVMStarted():
    cdk_path = "https://github.com/cdk/cdk/releases/download/cdk-2.8/cdk-2.8.jar"
    jar_path = str(pystow.join("STOUT-V2")) + "/cdk-2.8.jar"

    if not os.path.exists(jar_path):
        jar_path = pystow.ensure("STOUT-V2", url=cdk_path)

    startJVM(jvmPath, "-ea", "-Djava.class.path=" + str(jar_path))

cdk_base = "org.openscience.cdk"

def unicode_to_ascii(s: str) -> str:
    """Converts a unicode string to an ASCII string"""
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )

def preprocess_sentence(w: str) -> str:
    """Takes in a sentence, removes white spaces and generates a clean sentence."""
    w = unicode_to_ascii(w.strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.strip()
    w = "<start> " + w + " <end>"
    return w

def get_smiles_cdk(smiles: str) -> str:
    """This function takes the user input SMILES and Canonicalize it using the CDK Canonicalisation algorthim."""
    cdk_base = "org.openscience.cdk"
    SCOB = JClass(cdk_base + ".DefaultChemObjectBuilder")
    SmiFlavour = JClass(cdk_base + ".smiles.SmiFlavor")
    SmilesGenerator = JClass(cdk_base + ".smiles.SmilesGenerator")(SmiFlavour.Absolute)
    SmilesParser = JClass(cdk_base + ".smiles.SmilesParser")(SCOB.getInstance())
    molecule = SmilesParser.parseSmiles(smiles)
    CanSMILES = SmilesGenerator.create(molecule)
    return CanSMILES

def tokenize_input(input_SMILES: str, inp_lang, inp_max_length: int) -> np.array:
    """This function takes a user input SMILES and tokenizes it to feed it to the model."""
    sentence = preprocess_sentence(input_SMILES)
    inputs = [inp_lang.word_index[i] for i in sentence.split(" ")]
    # Use tf.keras.utils.pad_sequences instead of keras.preprocessing.sequence.pad_sequences
    tokenized_input = tf.keras.utils.pad_sequences(
        [inputs], maxlen=inp_max_length, padding="post"
    )
    return tokenized_input

def detokenize_output(predicted_array: np.array, targ_lang) -> str:
    """This function takes a predited input array and returns a IUPAC name by detokenizing the input."""
    outputs = [targ_lang.index_word[i] for i in predicted_array[0].numpy()]
    prediction = (
        " ".join([str(elem) for elem in outputs])
        .replace("<start> ", "")
        .replace(" <end>", "")
        .replace(" ", "")
    )
    return prediction

def download_trained_weights(model_url: str, model_path: str, verbose=1):
    """This function downloads the trained models and tokenizers to a default location."""
    if verbose > 0:
        print("Downloading trained model to " + str(model_path))
        model_path = pystow.ensure("STOUT-V2", url=model_url)
        print(model_path)
    if verbose > 0:
        print("... done downloading trained model!")
        with zipfile.ZipFile(model_path.as_posix(), "r") as zip_ref:
            zip_ref.extractall(model_path.parent.as_posix()) 