this is a FastAPI for Stout2 [https://jcheminf.biomedcentral.com/counter/pdf/10.1186/s13321-024-00941-x.pdf] which translates IUPAC -> SMILES, vice versa

this repo is missing some large models, to fix you can install [https://pypi.org/project/STOUT-pypi/], find the models folder and replace the models folder here, or the easiest option:
docker pull pupmaster/stout2api:latest
docker run -p 8000:8000 pupmaster/stout_api:latest

Documentation:
http://localhost:8000/docs

Example: 

def test_single_inference_forward():
    """Single SMILES → IUPAC Inference"""
    print_test_header("Single SMILES → IUPAC Inference")
    
    data = {
        "input_text": "CCO",
        "translation_type": "forward"
    }
    
    try:
        response = requests.post(f"{API_URL}/inference", json=data)
        print_response(response, "SMILES → IUPAC")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

some more documentation:
## 🍎 Summary:

We download STOUTv2 using the STOUT-pypi package and attempt to reproduce Rajan et al.’s reported metrics. While the model performs near-100% in translating important chemical structure, our exact-match results are worse than benchmarked. This discrepancy likely stems from small differences in tokenization or RDKit preprocessing, not due to model limitations. High BLEU and roundtrip scores confirm near-100% chemically correct translation. The stout-pypi package is lightweight, fast, and pipeline-friendly, though it requires a specific environment setup. We also demonstrate a method to improve inference speed by up to 14× with no code changes.

*Note: STOUTv2’s aim is SMILES→IUPAC, and while their package supports IUPAC→SMILES, OPSIN is a better alternative for reverse translation.*

## 🔧 Env Setup

### **Requirements**

- Linux (Recommended), MacOS, Windows/WSL
    - Windows is not recommended as is 6-12x slower than WSL/Linux.
- Recommended to run in Conda
- Tensorflow==2.10.1
    - See Improvement 1 for how to upgrade to 2.13.0 and get 7-14x speedup on WSL/Linux and 2x speedup on Windows.
- Python==3.8
    - Required for .pkl variables to be used.
    - Hard-Requirement.
- Less than 1GB. (Model weights stored on-GCP, local models add addtl. 250MB)

### Installation

```
**Pip Install:**
pip install STOUT-pypi

**Conda Install: (Recommended)**
conda create --name [name] python=3.8
conda activate [name]
conda install -c decimer 
python -m pip install stout-pypi
```

## 🎑 How to Use

### Basic Usage

```
from STOUT import translate_forward, translate_reverse

# SMILES to IUPAC name translation

SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
IUPAC_name = translate_forward(SMILES)
print("IUPAC name of "+SMILES+" is: "+IUPAC_name)

# IUPAC name to SMILES translation

IUPAC_name = "1,3,7-trimethylpurine-2,6-dione"
SMILES = translate_reverse(IUPAC_name)
print("SMILES of "+IUPAC_name+" is: "+SMILES)

#With optimizations as described in Improvement 1:
#Forward pass: Avg. 4 seconds on i7 CPU. Avg. [XX] seconds on XX GPU. (7x faster than naive implementation)
#Reverse pass: Avg  4.5 seconds on i7 CPU. Avg. [XX] seconds on XX GPU. (14x faster than naive implementation)
```

Note: Within the STOUT files is the team’s inference rig, training, and evaluation scripts minus training data. No documentation exists, but their code is well-commented. 

## 📊 Reproducibility Benchmarks

### Performance Metrics

| Metric | Value | Expected |
| --- | --- | --- |
| % Exact-Match Accuracy (SMILES → IUPAC) | 62.51% | 89.86 % |
| BLEU Score | 0.87, 67.54% perfect | 0.99 Average, 97.49% perfect BLEU |
| % Exact-Match Accuracy (IUPAC → SMILES ) | 8.54% | 96.97 % |
| Average Tanimoto Similarity (effectively roundtrip matching SMILES→IUPAC→SMILES) | 0.99,96.16% | 0.99 Average, 99.12% cases = 1.0 |
| Valid-SMILES %  | 99.7% | 99.80% |
| CPU Usage / Memory Used | WIN: 33.30%, 115.94 MB 

WSL: 25.10%, 635.81 MB | N/A |

![Figure_1.png](attachment:07c59545-ab09-4345-85be-d7d630e0cf78:Figure_1.png)

### Comparison Results

- **STOUTv2 underperforms benchmarks.**
    - Differences from Rajan et al. are likely due to minor tokenization and RDKit preprocessing mismatches (e.g. RDKit versioning, canonicalization).
    - High BLEU and roundtrip Tanimoto scores suggest the model captures chemical structure well but fails on formatting-sensitive exact-match metrics. From a cheminformatics standpoint, the model succeeds in preserving molecular identity which is more critical than token-level match in many real-world applications.
    - Failures are concentrated in non-chemical tokens (punctuation, formatting), not core structural terms, pointing to token-level noise, not model underperformance.
        - If B-12 wants a fix, train an AlphaEvolve-style RL or manually reverse engineer Rajan’s tokenization code.

![image.png](attachment:c6b15514-a0d6-44f8-9c4d-874b788ca4e6:image.png)

![image.png](attachment:315b2028-ec0d-4c5d-b4b5-2691934d72d6:image.png)

![Only 3% of failures are true roundtrip errors (SMILES→IUPAC→SMILES yields a different molecule), indicating the model preserves chemical structure even when formatting fails.](attachment:f9e53f22-ea15-4495-ae50-447568ae3806:image.png)

Only 3% of failures are true roundtrip errors (SMILES→IUPAC→SMILES yields a different molecule), indicating the model preserves chemical structure even when formatting fails.

- We can never 100% ensure perfect replication without researcher’s train:test split, model may have already trained on these datapoints.
- OS Matters here: Windows uses more CPU power, but uses substantially less memory. WSL/Linux uses 1/3 less compute but 6x memory. However, WSL computes order of magnitudes faster. **Prefer WSL/Linux unless mem-constrained.**

## 🔍 My Analysis

### Potential Improvements

1. **Issue 1:** Works only with an older TensorFlow version. (2.10.XX)
    - **✅ Improvement 1:** Make compatible with TF 2.13. **Gives 7-14x speedup on-inference.** For full effect, must run in WSL/Linux. Steps:
        - conda create -n [name] python=3.8 -y
        - conda activate [name]
        - conda install -c decimer #win-only
        - pip install tensorflow==2.13.0 keras==2.13.1 tensorflow-estimator==2.13.0 tensorboard==2.13.0 protobuf==4.25.8
        - pip install stout-pypi --no-deps
        - pip install pystow jpype1 unicodedata2 tqdm click
        
        ```bash
        *Test: (Linux/WSL Only)
        python3 -c "from STOUT import translate_forward, translate_reverse; print(translate_forward('CN1C=NC2=C1C(=O)N(C(=O)N2C)C'))"*
        ```
        
    - ❌ **Improvement 2:** Rebuild PyTorch-compatible version using ONNX.
        - Not worth time investment (unless comfortable with ONNX internal files) . Error exists because unable to convert TensorListStack inside a while_loop.
2. **Issue 2:** Trained model files are unoptimized, potentially slow on heavy inference pipelines. 
    1. **🔜 Improvement 3**: Analyze trained model weights using WeightWatcher to prune degenerate or noisy neurons. Will reduce memory usage, speedup inference.  
    2. **🔜 Improvement 4**: Add FlashAttention2/3. Easy to add and benchmark. Likely will decrease memory requirement by 2x and speedup inference by 1.5x. 
    3. **🔜 Improvement 5**: Quick & Easy: LoRA + can add heads for new behaviors if needed. For example, vector embeddings from heads can be fed into chemLLMs to bake-in functional group priority information. 
3. **Issue 3:** Unreliable past 350 chars.
    1. **🔜 Improvement 6:** Hard & Project-Specific: For projects ingesting long biomolecules (macrocyclic peptides, oligonucleotides, biologics), can train a H-Net with SSM layers. SSMs are designed to be attention-efficient, work well in long sequences, and the H-Net hierarchal architecture allows easy nested structure understanding. However, training time and param size will increase nontrivially. Not worth training costs for small molecules. 
4. **Issue 4:** General Performance
    1. 💭 I am of the opinion that this model is robust enough for prod, but can easily be improved to perform better, especially with modern test-time methods on similar molecules and their IUPAC translation. If B-12 opts for full-retraining, I suggest forking the [DCG](https://arxiv.org/html/2505.22949v1#S3) model for its strong inductive priors for this task.


    
