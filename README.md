# Arabic-Scam-and-Legitimate-Call-Conversation-Dataset-ASLC-448-

## Description

This dataset presents a novel multi-dialect Arabic scam and legitimate telephone call conversation corpus designed for training and evaluating scam detection models. The dataset addresses a critical gap in Arabic-language fraud detection research, where no publicly available scam call datasets currently exist.

The dataset contains 448 annotated conversations covering nine Arabic dialects: Modern Standard Arabic (MSA), Egyptian, Gulf, Jordanian, Saudi, Yemeni, Sudanese, Iraqi, and Syrian. Each conversation simulates a realistic telephone interaction structured as a multi-turn dialogue between a caller and a receiver over five utterance turns (three caller turns and two receiver turns).

The conversations are labeled as either scam (246 samples, 54.9%) or not_scam (202 samples, 45.1%) and span 23 fine-grained categories. The scam class includes 11 categories representing prevalent fraud tactics: bank fraud, prize/lottery scams, government impersonation, investment/crypto fraud, tech support scams, charity/emergency exploitation, romance/social engineering, fraudulent job offers, SIM/WhatsApp hijacking, rental/real estate fraud, and delivery/shipping scams. The non-scam class covers 12 categories of common legitimate calls: banking services, delivery notifications, utility services, personal/family calls, medical appointments, workplace communications, telecom offers, government services, restaurant orders, educational institutions, insurance services, and real estate inquiries.

Each conversation is additionally annotated with five numerical risk indicator scores that quantify the intensity of social engineering tactics: urgency score (0–5), sensitive information requests (0–2), financial pressure score (0–5), threat score (0–3), and impersonation score (0–2). These annotations provide granular insight into the manipulation strategies employed in scam calls and can serve as auxiliary features for classification models.

The audio component of the dataset consists of synthesized speech files generated from the text conversations using the Microsoft Edge Neural Text-to-Speech (TTS) engine. Dialect-specific voice pairs were assigned to each conversation (distinct male and female voices for caller and receiver), and post-processing was applied to simulate realistic telephone call conditions, including a bandpass filter (300–3400 Hz), light background noise, speed variation, and amplitude normalization. All audio files are in 16-bit PCM WAV format at 16 kHz mono, compatible with speech processing models such as OpenAI Whisper.

---

## Keywords

Arabic NLP; Scam Detection; Fraud Detection; Phone Call Classification; Multi-Dialect Arabic; Social Engineering; Text Classification; Speech Classification; Cybersecurity Dataset; AraBERT

---

## Files Included

| File | Description |
|------|-------------|
| arabic_scam_dataset_complete.xlsx | Complete text dataset with 448 conversations, labels, categories, dialects, risk scores, and metadata (18 columns) |
| audio_dataset/scam/*.wav | Synthesized audio files for scam conversations (16 kHz, mono, WAV) |
| audio_dataset/not_scam/*.wav | Synthesized audio files for legitimate conversations (16 kHz, mono, WAV) |

---

## Data Structure

The Excel file contains 18 columns per conversation:

| Column | Type | Description |
|--------|------|-------------|
| conversation_id | String | Unique identifier (CONV_0001 to CONV_0448) |
| full_conversation | String | Complete conversation text with speaker labels |
| caller_turn_1 | String | First caller utterance |
| receiver_turn_1 | String | First receiver response |
| caller_turn_2 | String | Second caller utterance |
| receiver_turn_2 | String | Second receiver response |
| caller_turn_3 | String | Third caller utterance |
| label | String | Binary class label: scam or not_scam |
| category | String | Fine-grained category (23 categories) |
| dialect | String | Arabic dialect (9 dialects) |
| urgency_score | Integer | Time pressure intensity (0–5) |
| sensitive_info_requests | Integer | Confidential data solicitation (0–2) |
| financial_pressure_score | Integer | Monetary demands intensity (0–5) |
| threat_score | Integer | Threat/intimidation level (0–3) |
| impersonation_score | Integer | Identity deception level (0–2) |
| conversation_length | Integer | Total characters in conversation |
| word_count | Integer | Total words in conversation |
| label_binary | Integer | Binary encoding: 1 = scam, 0 = not_scam |

---

## Categories

### Scam (11 categories, 246 conversations)
bank_fraud, prize_lottery, government_impersonation, investment_crypto, tech_support, charity_emergency, romance_social, job_offer, sim_whatsapp, rental_realestate, delivery_shipping

### Legitimate (12 categories, 202 conversations)
bank_legitimate, delivery_legitimate, service_legitimate, personal_family, medical_legitimate, work_legitimate, telecom_legitimate, government_legitimate, restaurant_legitimate, education_legitimate, insurance_legitimate, realestate_legitimate

---

## Suggested Uses

- Arabic text-based scam call classification using transformer models (e.g., AraBERT, CAMeLBERT)
- Audio-based scam detection using speech features (MFCC, mel-spectrograms)
- Multimodal scam detection combining text and audio modalities
- Cross-dialect Arabic NLP evaluation
- Social engineering tactic analysis and visualization
- Benchmarking Arabic language understanding models


Arabic Scam Call Detection System
A comprehensive AI-powered system for detecting scam phone calls in Arabic using multiple strategies including fine-tuned BERT, zero-shot learning, few-shot learning, and ensemble methods.
📁 Project Structure
arabic_scam_detection/
├── train_arabert_classifier.py      # Main AraBERT training script
├── inference_wav_classifier.py      # WAV file classification (Whisper + AraBERT)
├── dataset_tts_conversion.py        # Text-to-Speech dataset creation
├── dataset_builder_complete.py      # Complete dataset builder with audio features
├── damse_v2_enhanced.py            # DAMSE v2: Enhanced ensemble with ZSL & Few-Shot
├── damse_v3_ieee.py                # DAMSE v3: IEEE quality with K-Fold CV & GZSL
├── explainability_lime_shap.py     # Model explainability (LIME & SHAP)
└── README.md                        # This file
🎯 Features
Core Capabilities

Multi-Strategy Detection: Combines 4-5 different approaches for robust scam detection
Dialect-Aware: Supports 8+ Arabic dialects (Egyptian, Gulf, Saudi, Iraqi, Syrian, etc.)
Multi-Modal: Text + Audio classification
Explainable AI: LIME & SHAP visualizations

Detection Strategies

S1: Fine-tuned AraBERT (Supervised Learning)

Model: aubmindlab/bert-base-arabertv2
100 epochs with early stopping
Achieves ~95%+ F1-score


S2: Enhanced Zero-Shot Learning (No Training Required)

Model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
6 Arabic + English hypothesis templates
Scam signal extraction + Platt calibration
Achieves ~85-90% F1-score


S3: Few-Shot Learning (5-20 examples per class)

SetFit or sentence-transformer + kNN
Learns from minimal labeled data
Achieves ~80-85% F1-score with 20 shots


S4: Risk-Score Gradient Boosting (Feature-based ML)

Uses conversation features: urgency, financial pressure, threats, etc.
Traditional ML approach for interpretability


S5: DAMSE Ensemble (Dialect-Adaptive Multi-Strategy Ensemble)

Adaptive weighted fusion of S1-S4
Optimized weights per dialect
Best overall performance: ~96-98% F1-score



# Arabic Scam Call Detection System

A comprehensive AI-powered system for detecting scam phone calls in Arabic using multiple strategies including fine-tuned BERT, zero-shot learning, few-shot learning, and ensemble methods.

## 📁 Project Structure

```
arabic_scam_detection/
├── train_arabert_classifier.py      # Main AraBERT training script
├── inference_wav_classifier.py      # WAV file classification (Whisper + AraBERT)
├── dataset_tts_conversion.py        # Text-to-Speech dataset creation
├── dataset_builder_complete.py      # Complete dataset builder with audio features
├── damse_v2_enhanced.py            # DAMSE v2: Enhanced ensemble with ZSL & Few-Shot
├── damse_v3_ieee.py                # DAMSE v3: IEEE quality with K-Fold CV & GZSL
├── explainability_lime_shap.py     # Model explainability (LIME & SHAP)
└── README.md                        # This file
```

## 🎯 Features

### Core Capabilities
- **Multi-Strategy Detection**: Combines 4-5 different approaches for robust scam detection
- **Dialect-Aware**: Supports 8+ Arabic dialects (Egyptian, Gulf, Saudi, Iraqi, Syrian, etc.)
- **Multi-Modal**: Text + Audio classification
- **Explainable AI**: LIME & SHAP visualizations

### Detection Strategies

1. **S1: Fine-tuned AraBERT** (Supervised Learning)
   - Model: `aubmindlab/bert-base-arabertv2`
   - 100 epochs with early stopping
   - Achieves ~95%+ F1-score

2. **S2: Enhanced Zero-Shot Learning** (No Training Required)
   - Model: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
   - 6 Arabic + English hypothesis templates
   - Scam signal extraction + Platt calibration
   - Achieves ~85-92% F1-score

3. **S3: Few-Shot Learning** (5-20 examples per class)
   - SetFit or sentence-transformer + kNN
   - Learns from minimal labeled data
   - Achieves ~89-93% F1-score with 20 shots

4. **S4: Risk-Score Gradient Boosting** (Feature-based ML)
   - Uses conversation features: urgency, financial pressure, threats, etc.
   - Traditional ML approach for interpretability

5. **S5: DAMSE Ensemble** (Dialect-Adaptive Multi-Strategy Ensemble)
   - Adaptive weighted fusion of S1-S4
   - Optimized weights per dialect
   - Best overall performance: ~99-99.9% F1-score

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers
pip install pandas numpy scikit-learn
pip install librosa soundfile  # For audio processing
pip install openai-whisper     # For speech recognition
pip install edge-tts pydub     # For TTS conversion
pip install lime shap          # For explainability
pip install sentence-transformers
pip install openpyxl matplotlib seaborn
```

### Basic Usage

#### 1. Train AraBERT Classifier

```bash
python train_arabert_classifier.py
```

**Expected Output:**
- `outputs/best_model.pt` - Fine-tuned model weights
- `outputs/results.json` - Performance metrics
- `outputs/*.png` - Visualizations

**Performance:** ~95% accuracy, ~94% F1-score on test set

#### 2. Classify Audio File

```bash
python inference_wav_classifier.py
```

**Configuration:**
```python
WAV_FILE = "333.wav"              # Your audio file
MODEL_WEIGHTS = "outputs/best_model.pt"
WHISPER_SIZE = "large-v3"         # or "base" for faster
```

**Output:**
```
🚨 SCAM DETECTED!
  Label: SCAM
  Scam Probability: 0.9234 (92.3%)
  Risk Level: 🔴 HIGH
```

#### 3. Create Audio Dataset

```bash
python dataset_tts_conversion.py
```

Converts text conversations to realistic WAV files with:
- Dialect-specific voices (Edge TTS)
- Telephone band-pass filtering
- Background noise simulation
- Speed variation

#### 4. Run DAMSE v2 Ensemble

```bash
python damse_v2_enhanced.py
```

**Features:**
- Enhanced Zero-Shot with multi-template averaging
- Few-Shot learning with 20 shots/class
- Dialect-adaptive ensemble weighting
- Improved performance: 96-97% F1-score

#### 5. Run DAMSE v3 (Research Quality)

```bash
python damse_v3_ieee.py
```

**Features:**
- 5-Fold Stratified Cross-Validation
- GZSL: Generalized Zero-Shot Learning
- Statistical significance testing
- Few-shot learning curves (5/10/20 shots)

**Output:**
```
5-FOLD CV RESULTS (mean ± std)
  S5: DAMSE | F1 | 0.9678 ± 0.0123  [0.9512, 0.9844]
  
GZSL: Unseen Dialects
  H-Score: 0.8934  (Harmonic mean: Seen vs Unseen)
```

#### 6. Model Explainability

```bash
python explainability_lime_shap.py
```

**Generates:**
- LIME HTML visualization
- SHAP bar charts
- Top contributing words for each prediction

## 📊 Dataset Format

### Required Excel/CSV Structure

```
conversation_id,full_conversation,label,dialect,category,urgency_score,...
CONV_001,"السلام عليكم...",scam,Egyptian,bank_impersonation,5,...
CONV_002,"مرحبا...",not_scam,Saudi,family_call,0,...
```

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `conversation_id` | str | Unique identifier |
| `full_conversation` | str | Complete conversation text |
| `label` | str | "scam" or "not_scam" |
| `label_binary` | int | 1 (scam) or 0 (not_scam) |
| `dialect` | str | Arabic dialect |
| `category` | str | Conversation category |
| `urgency_score` | int | 0-5 urgency level |
| `financial_pressure_score` | int | 0-5 financial pressure |
| `threat_score` | int | 0-5 threat level |
| `impersonation_score` | int | 0-5 impersonation |

## 🎓 Research Contributions

### Novel Aspects

1. **DAMSE Framework**: First dialect-aware ensemble for Arabic scam detection
2. **Zero-Shot Arabic NLI**: Multi-template averaging with signal extraction
3. **GZSL Evaluation**: Harmonic mean on unseen dialects
4. **Few-Shot Analysis**: Learning curves with 5/10/20 examples
5. **Cross-Dialect Robustness**: Performance across 8+ Arabic dialects


## 🔧 Advanced Configuration

### AraBERT Training Parameters

```python
class Config:
    MODEL_NAME = "aubmindlab/bert-base-arabertv2"
    MAX_LEN = 256           # Sequence length
    BATCH_SIZE = 16         # Adjust based on GPU
    EPOCHS = 100            # With early stopping
    LEARNING_RATE = 2e-5
    PATIENCE = 3            # Early stopping patience
    DROPOUT = 0.3
```

### Zero-Shot Templates

Customize in `damse_v2_enhanced.py`:

```python
zs_templates = {
    "ar_scam_direct": [
        "هذه مكالمة احتيال",      # Scam call
        "هذه مكالمة عادية"        # Normal call
    ],
    # Add custom templates...
}
```

### DAMSE Ensemble Weights

Weights are automatically optimized per dialect. Manual override:

```python
# In damse_v2_enhanced.py
dialect_weights = {
    "Egyptian": [0.45, 0.12, 0.18, 0.25],  # [AraBERT, ZS, FS, GBM]
    "Gulf": [0.40, 0.15, 0.20, 0.25],
    # ...
}
```

## 📈 Evaluation Metrics

All scripts output comprehensive metrics:

- **Classification**: Accuracy, Precision, Recall, F1 (weighted/macro/micro)
- **Advanced**: MCC, Cohen's Kappa, Balanced Accuracy
- **Probabilistic**: AUC-ROC, AUC-PR, Log Loss
- **Per-Class**: Sensitivity (TPR), Specificity (TNR)
- **GZSL**: Harmonic Mean (H-score)

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size in Config
BATCH_SIZE = 8  # or 4
```

**2. Whisper Installation Issues**
```bash
# Use specific version
pip install openai-whisper==20230314
```

**3. Edge TTS Errors**
```bash
# Update edge-tts
pip install --upgrade edge-tts
```

**4. Arabic Text Display Issues**
```python
# In matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{tawfik2024damse,
  title={DAMSE: Dialect-Aware Multi-Strategy Ensemble for Arabic Scam Call Detection},
  author={Tawfik, Mohammed},
  journal={IEEE Transactions on...},
  year={2024}
}
```

## 📧 Contact

**Dr. Mohammed Tawfik**
- Institution: Ajloun National University, Jordan
- Email: KMKHOL01@GMAIL.COM
- Research: AI, Cybersecurity, Natural Language Processing

## 📜 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **AraBERT**: aubmindlab team for Arabic BERT models
- **Whisper**: OpenAI for multilingual ASR
- **Transformers**: HuggingFace for the framework
- **Edge TTS**: Microsoft for Arabic voices

## 🔄 Version History

- **v3.0** (Current): K-Fold CV, GZSL, Statistical testing
- **v2.0**: Enhanced ZSL, Few-Shot learning
- **v1.0**: Initial DAMSE framework

---

**Note**: This is research code. For production deployment, add additional validation, security measures, and ethical considerations.
---

## License

CC BY 4.0 (Creative Commons Attribution 4.0 International) To access to this dataset email me :kmkhol01@gmail.com
