#!/usr/bin/env python3
"""
=============================================================================
 Model Explainability: LIME & SHAP for Arabic Scam Detection
 
 Provides interpretable explanations for model predictions using:
   - LIME: Local Interpretable Model-agnostic Explanations
   - SHAP: SHapley Additive exPlanations
=============================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    SEED            = 42
    ARABERT_MODEL   = "aubmindlab/bert-base-arabertv2"
    MAX_LEN         = 256
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH      = "outputs/best_model.pt"
    OUTPUT_DIR      = "outputs"

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  LIME & SHAP Explainability")
print("=" * 70)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(Config.ARABERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(Config.ARABERT_MODEL, num_labels=2)

if os.path.exists(Config.MODEL_PATH):
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    print(f"✓ Loaded fine-tuned weights from {Config.MODEL_PATH}")
else:
    print("⚠ Using base model (no fine-tuned weights found)")

model.to(Config.DEVICE)
model.eval()

# Load sample data
try:
    df = pd.read_excel("arabic_scam_dataset_complete.xlsx")
    df['label_binary'] = df['label'].apply(lambda x: 1 if x == 'scam' else 0)
except FileNotFoundError:
    print("⚠ Dataset not found. Creating dummy data...")
    df = pd.DataFrame({
        'full_conversation': [
            "مبروك ربحت جائزة كبيرة",
            "كيف حالك اليوم",
            "تم حظر حسابك البنكي فوراً",
            "مرحبا بك في الخدمة"
        ],
        'label_binary': [1, 0, 1, 0]
    })

print(f"✓ Loaded {len(df)} samples")

# ═════════════════════════════════════════════════════════════════════════════
# 1. LIME EXPLANATION
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  LIME: Local Interpretable Model-agnostic Explanations")
print(f"{'=' * 70}")

import lime
import lime.lime_text

def predictor_for_lime(texts):
    """Predict probabilities for LIME."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, 
                       truncation=True, max_length=Config.MAX_LEN).to(Config.DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=1).cpu().numpy()

# Initialize LIME explainer
explainer_lime = lime.lime_text.LimeTextExplainer(
    class_names=["not_scam", "scam"],
    split_expression=r'\s+',
    random_state=Config.SEED
)

# Explain a scam example
scam_examples = df[df['label_binary'] == 1]
if len(scam_examples) > 0:
    text_to_explain = str(scam_examples.iloc[0]['full_conversation'])
    print(f"\n✓ Explaining SCAM example:")
    print(f"  Text: {text_to_explain[:80]}...")
    
    explanation = explainer_lime.explain_instance(
        text_to_explain,
        predictor_for_lime,
        num_features=10,
        num_samples=100
    )
    
    print(f"\n  Top 10 Features (LIME):")
    for word, weight in sorted(explanation.as_list(), key=lambda x: abs(x[1]), reverse=True):
        direction = "→ SCAM" if weight > 0 else "→ NOT_SCAM"
        print(f"    {word:20s} : {weight:+.4f}  {direction}")
    
    # Save HTML visualization
    explanation.save_to_file(os.path.join(Config.OUTPUT_DIR, "lime_explanation.html"))
    print(f"\n  ✓ Saved LIME visualization to {Config.OUTPUT_DIR}/lime_explanation.html")

# ═════════════════════════════════════════════════════════════════════════════
# 2. SHAP EXPLANATION (CORRECTED)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  SHAP: SHapley Additive exPlanations")
print(f"{'=' * 70}")

import shap

def predictor_for_shap(texts):
    """Predict logits for SHAP (handles NumPy arrays and strings)."""
    # Handle NumPy arrays
    if hasattr(texts, 'tolist'):
        texts = texts.tolist()
    
    # Handle single string
    if isinstance(texts, str):
        texts = [texts]
    
    # Tokenize
    inputs = tokenizer(texts, return_tensors="pt", padding=True, 
                       truncation=True, max_length=Config.MAX_LEN).to(Config.DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.logits.cpu().numpy()

# Initialize masker and explainer
masker = shap.maskers.Text(tokenizer)
explainer_shap = shap.Explainer(predictor_for_shap, masker, 
                                 output_names=["not_scam", "scam"])

# Explain a not_scam example
not_scam_examples = df[df['label_binary'] == 0]
if len(not_scam_examples) > 0:
    text_shap = str(not_scam_examples.iloc[0]['full_conversation'])
    print(f"\n✓ Explaining NOT_SCAM example:")
    print(f"  Text: {text_shap[:80]}...")
    
    # Calculate SHAP values
    shap_values = explainer_shap([text_shap])
    
    # Get prediction
    prediction_logits = predictor_for_shap([text_shap])[0]
    predicted_idx = np.argmax(prediction_logits)
    predicted_label = ["not_scam", "scam"][predicted_idx]
    
    # Extract top contributing words
    values = shap_values.values[0][:, predicted_idx]
    tokens = shap_values.data[0]
    
    # Zip and sort
    word_shap = list(zip(tokens, values))
    word_shap_sorted = sorted(word_shap, key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n  Predicted Class: {predicted_label}")
    print(f"  Top 10 Contributing Words (SHAP):")
    for token, val in word_shap_sorted[:10]:
        clean_token = token.replace("##", "")
        direction = f"→ {predicted_label.upper()}" if val > 0 else "→ OTHER"
        print(f"    {clean_token:20s} : {val:+.4f}  {direction}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  Generating Visualizations")
print(f"{'=' * 70}")

# Manual Bar Chart for Top Words (SHAP)
if len(not_scam_examples) > 0:
    vals = shap_values.values[0][:, predicted_idx]
    feature_names = shap_values.data[0]
    
    # Sort and take top 15
    combined = list(zip(feature_names, vals))
    combined_sorted = sorted(combined, key=lambda x: abs(x[1]), reverse=False)
    top_features = combined_sorted[-15:]
    
    features_plot, values_plot = zip(*top_features)
    clean_features = [f.replace("##", "") for f in features_plot]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in values_plot]
    
    ax.barh(range(len(values_plot)), values_plot, color=colors, 
            edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(values_plot)))
    ax.set_yticklabels(clean_features, fontsize=11)
    ax.set_xlabel(f"SHAP Value (Impact on '{predicted_label}')", fontsize=12)
    ax.set_title("Top 15 Words - SHAP Explanation", fontsize=14, weight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "shap_bar_chart.png"), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved SHAP bar chart to {Config.OUTPUT_DIR}/shap_bar_chart.png")
    plt.close()

# Comparison: LIME vs SHAP
if len(scam_examples) > 0 and len(not_scam_examples) > 0:
    print(f"\n{'─' * 70}")
    print(f"  LIME vs SHAP Comparison:")
    print(f"  - LIME: Model-agnostic, uses local linear approximation")
    print(f"  - SHAP: Based on game theory, provides additive feature attribution")
    print(f"  - Both highlight important words, but may differ in exact rankings")
    print(f"{'─' * 70}")

print(f"""
{'=' * 70}
  ✅ EXPLAINABILITY ANALYSIS COMPLETE
{'=' * 70}

  Outputs saved to {Config.OUTPUT_DIR}/:
    - lime_explanation.html (interactive visualization)
    - shap_bar_chart.png (top contributing words)

  Key Insights:
    - LIME identifies local word importance for specific predictions
    - SHAP provides global feature attribution across all predictions
    - Both methods help understand model decision-making process
    
  Usage in Jupyter/Colab:
    - LIME: Use explanation.show_in_notebook(text=True)
    - SHAP: Use shap.initjs(); shap.plots.text(shap_values[0])
{'=' * 70}
""")
