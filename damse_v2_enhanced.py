#!/usr/bin/env python3
"""
=============================================================================
 DAMSE v2: Dialect-Aware Multi-Strategy Ensemble (Enhanced)
=============================================================================
 
 Improvements over v1:
   ✓ Enhanced Zero-Shot: mDeBERTa + 6 templates + signal extraction + calibration
   ✓ Few-Shot Learning: SetFit with 5/10/20 shots
   ✓ Improved ensemble weights optimization
 
 Strategies:
   S1: Fine-tuned AraBERT (supervised)
   S2: Enhanced Zero-Shot (multi-template + calibration)
   S3: Few-Shot Learning (SetFit or kNN)
   S4: Risk-Score Gradient Boosting
   S5: DAMSE Ensemble (dialect-adaptive)
=============================================================================
"""

import os, warnings, random, json, time, re
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup, pipeline
)
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    SEED            = 42
    ARABERT_MODEL   = "aubmindlab/bert-base-arabertv2"
    ZS_MODEL        = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    SETFIT_MODEL    = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    FEW_SHOT_N      = 20
    MAX_LEN         = 256
    BATCH_SIZE      = 16
    EPOCHS          = 10
    LEARNING_RATE   = 2e-5
    WEIGHT_DECAY    = 0.01
    WARMUP_RATIO    = 0.1
    PATIENCE        = 3
    DROPOUT         = 0.3
    TEST_SIZE       = 0.20
    VAL_SIZE        = 0.10
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH       = "arabic_scam_dataset_complete.xlsx"
    OUTPUT_DIR      = "outputs_v2"

def seed_everything(seed=Config.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Extract scam signals for ZSL
# ─────────────────────────────────────────────────────────────────────────────
def extract_scam_signals(text):
    """Extract key scam indicators to prepend to conversation."""
    signals = []
    urgency = ['بسرعة', 'فوراً', 'حالاً', 'عاجل', 'آخر موعد', 'ضروري']
    money = ['حوّل', 'ادفع', 'رسوم', 'مبلغ', 'تحويل', 'جائزة', 'ربحت']
    info = ['رقم البطاقة', 'الرقم السري', 'رمز التحقق', 'بياناتك']
    threat = ['إجراءات', 'تجميد', 'إيقاف', 'غرامة']
    impersonate = ['معك من البنك', 'شركة الاتصالات', 'الدعم الفني']

    if any(w in text for w in urgency):   signals.append("ضغط زمني")
    if any(w in text for w in money):     signals.append("طلب مالي")
    if any(w in text for w in info):      signals.append("طلب بيانات سرية")
    if any(w in text for w in threat):    signals.append("تهديد")
    if any(w in text for w in impersonate): signals.append("انتحال صفة")

    return (" | ".join(signals) + " || " + text) if signals else text

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  PHASE 1: Data Loading")
print("=" * 70)

df = pd.read_excel(Config.DATA_PATH)
print(f"✓ Loaded: {len(df)} conversations")

# Split
df_trainval, df_test = train_test_split(
    df, test_size=Config.TEST_SIZE, random_state=Config.SEED, stratify=df['label']
)
df_train, df_val = train_test_split(
    df_trainval, test_size=Config.VAL_SIZE / (1 - Config.TEST_SIZE),
    random_state=Config.SEED, stratify=df_trainval['label']
)

df_train = df_train.reset_index(drop=True)
df_val   = df_val.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)

print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

feature_cols = ['urgency_score', 'sensitive_info_requests', 'financial_pressure_score',
                'threat_score', 'impersonation_score', 'conversation_length', 'word_count']

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1: AraBERT (abbreviated - load from checkpoint)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  STRATEGY 1: Fine-tuned AraBERT (loading checkpoint)")
print(f"{'=' * 70}")

# Assume model already trained - load it
tokenizer = AutoTokenizer.from_pretrained(Config.ARABERT_MODEL)
model_s1 = AutoModelForSequenceClassification.from_pretrained(Config.ARABERT_MODEL, num_labels=2)

checkpoint_path = os.path.join(Config.OUTPUT_DIR.replace("_v2", ""), "best_model.pt")
if os.path.exists(checkpoint_path):
    model_s1.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    print("✓ Loaded checkpoint")
else:
    print("⚠ No checkpoint found - using base model")

model_s1.to(Config.DEVICE)
model_s1.eval()

# Get predictions (simplified)
class ScamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

test_ds = ScamDataset(df_test['full_conversation'].tolist(),
                       df_test['label_binary'].tolist(), tokenizer, Config.MAX_LEN)
test_dl = DataLoader(test_ds, batch_size=Config.BATCH_SIZE)

s1_preds, s1_probs = [], []
with torch.no_grad():
    for batch in test_dl:
        out = model_s1(input_ids=batch['input_ids'].to(Config.DEVICE),
                      attention_mask=batch['attention_mask'].to(Config.DEVICE))
        probs = torch.softmax(out.logits, dim=1)
        s1_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
        s1_probs.extend(probs[:, 1].cpu().numpy())

s1_preds = np.array(s1_preds)
s1_probs = np.array(s1_probs)
s1_labels = df_test['label_binary'].values

print(f"  Test Acc: {accuracy_score(s1_labels, s1_preds):.4f}  F1: {f1_score(s1_labels, s1_preds, average='weighted'):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2: Enhanced Zero-Shot
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  STRATEGY 2: Enhanced Zero-Shot (mDeBERTa + calibration)")
print(f"{'=' * 70}")

zs_classifier = pipeline("zero-shot-classification", model=Config.ZS_MODEL,
                          device=0 if torch.cuda.is_available() else -1)

# 6 Arabic + English templates
zs_templates = {
    "ar_scam_direct": ["هذه مكالمة احتيال", "هذه مكالمة عادية"],
    "ar_social_eng": ["المتصل يستخدم الضغط والتهديد", "المتصل يتواصل بشكل طبيعي"],
    "ar_bank_fraud": ["محاولة احتيال مالي", "مكالمة خدمة عملاء"],
    "en_detailed": ["scam phone call fraud", "legitimate phone call"],
    "en_indicators": ["caller stealing money information", "normal conversation"],
    "mixed_strong": ["احتيال - scam - خداع", "مكالمة شرعية - legitimate"],
}
template_weights = {"ar_scam_direct": 1.5, "ar_social_eng": 1.3, "ar_bank_fraud": 1.2,
                    "en_detailed": 0.8, "en_indicators": 0.7, "mixed_strong": 1.0}
total_tw = sum(template_weights.values())

def run_zsl(texts_list, zs_clf, templates, tw, total_w, batch_size=8):
    all_scores = np.zeros(len(texts_list))
    for tname, tlabels in templates.items():
        w = tw.get(tname, 1.0)
        scores = []
        for i in range(0, len(texts_list), batch_size):
            batch = [extract_scam_signals(t) for t in texts_list[i:i+batch_size]]
            res = zs_clf(batch, tlabels, multi_label=False)
            if not isinstance(res, list): res = [res]
            for r in res:
                scores.append(r['scores'][r['labels'].index(tlabels[0])])
        all_scores += w * np.array(scores)
    return all_scores / total_w

print("  Running ZSL on test set...")
s2_probs_raw = run_zsl(df_test['full_conversation'].tolist(), zs_classifier,
                        zs_templates, template_weights, total_tw)

# Calibration
s2_val_raw = run_zsl(df_val['full_conversation'].tolist(), zs_classifier,
                      zs_templates, template_weights, total_tw)
platt = LogisticRegression(C=1.0, random_state=Config.SEED)
platt.fit(s2_val_raw.reshape(-1, 1), df_val['label_binary'].values)

s2_probs = platt.predict_proba(s2_probs_raw.reshape(-1, 1))[:, 1]
s2_val_cal = platt.predict_proba(s2_val_raw.reshape(-1, 1))[:, 1]

# Optimize threshold
best_thr = 0.5
best_f1 = 0
for thr in np.arange(0.3, 0.7, 0.01):
    f1_v = f1_score(df_val['label_binary'].values, (s2_val_cal >= thr).astype(int), average='weighted')
    if f1_v > best_f1: best_f1 = f1_v; best_thr = thr

s2_preds = (s2_probs >= best_thr).astype(int)
s2_labels = df_test['label_binary'].values

print(f"  Test Acc: {accuracy_score(s2_labels, s2_preds):.4f}  F1: {f1_score(s2_labels, s2_preds, average='weighted'):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3: Few-Shot (kNN on sentence embeddings)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  STRATEGY 3: Few-Shot ({Config.FEW_SHOT_N}-shot kNN)")
print(f"{'=' * 70}")

st_model = SentenceTransformer(Config.SETFIT_MODEL)

# Encode all
print("  Encoding texts...")
train_emb = st_model.encode(df_train['full_conversation'].tolist(), batch_size=64, show_progress_bar=False)
test_emb = st_model.encode(df_test['full_conversation'].tolist(), batch_size=64, show_progress_bar=False)

# Sample N shots per class
fs_idx = []
for lbl in [0, 1]:
    lbl_m = np.where(df_train['label_binary'].values == lbl)[0]
    chosen = np.random.choice(lbl_m, min(Config.FEW_SHOT_N, len(lbl_m)), replace=False)
    fs_idx.extend(chosen)

knn = KNeighborsClassifier(n_neighbors=7, metric='cosine', weights='distance')
knn.fit(train_emb[fs_idx], df_train['label_binary'].values[fs_idx])

s3_probs = knn.predict_proba(test_emb)[:, 1]
s3_preds = knn.predict(test_emb)
s3_labels = df_test['label_binary'].values

print(f"  Test Acc: {accuracy_score(s3_labels, s3_preds):.4f}  F1: {f1_score(s3_labels, s3_preds, average='weighted'):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 4: Gradient Boosting
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  STRATEGY 4: Gradient Boosting")
print(f"{'=' * 70}")

scaler = StandardScaler()
X_train_gb = scaler.fit_transform(df_train[feature_cols].values)
X_test_gb = scaler.transform(df_test[feature_cols].values)

gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                 subsample=0.8, random_state=Config.SEED)
gb.fit(X_train_gb, df_train['label_binary'].values)

s4_probs = gb.predict_proba(X_test_gb)[:, 1]
s4_preds = gb.predict(X_test_gb)
s4_labels = df_test['label_binary'].values

print(f"  Test Acc: {accuracy_score(s4_labels, s4_preds):.4f}  F1: {f1_score(s4_labels, s4_preds, average='weighted'):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 5: DAMSE Ensemble
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  STRATEGY 5: DAMSE v2 Ensemble")
print(f"{'=' * 70}")

# Get val predictions
val_ds = ScamDataset(df_val['full_conversation'].tolist(),
                      df_val['label_binary'].tolist(), tokenizer, Config.MAX_LEN)
val_dl = DataLoader(val_ds, batch_size=Config.BATCH_SIZE)

s1_val_p = []
with torch.no_grad():
    for batch in val_dl:
        out = model_s1(input_ids=batch['input_ids'].to(Config.DEVICE),
                      attention_mask=batch['attention_mask'].to(Config.DEVICE))
        s1_val_p.extend(torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy())
s1_val_p = np.array(s1_val_p)

val_emb = st_model.encode(df_val['full_conversation'].tolist(), batch_size=64, show_progress_bar=False)
s3_val_p = knn.predict_proba(val_emb)[:, 1]

X_val_gb = scaler.transform(df_val[feature_cols].values)
s4_val_p = gb.predict_proba(X_val_gb)[:, 1]

# Optimize weights per dialect
d_val = df_val['dialect'].values
d_test = df_test['dialect'].values
dw = {}

for d in np.unique(d_test):
    m = d_val == d
    if m.sum() < 2:
        dw[d] = [0.40, 0.10, 0.15, 0.35]
        continue
    
    best_w, best_fd = None, 0
    for w1 in np.arange(0.2, 0.7, 0.1):
        for w2 in np.arange(0.05, 0.3, 0.05):
            for w3 in np.arange(0.05, 0.3, 0.05):
                w4 = round(1.0 - w1 - w2 - w3, 2)
                if w4 < 0.05 or w4 > 0.6: continue
                fused = w1*s1_val_p[m] + w2*s2_val_cal[m] + w3*s3_val_p[m] + w4*s4_val_p[m]
                fd = f1_score(df_val['label_binary'].values[m], (fused>=0.5).astype(int),
                              average='weighted', zero_division=0)
                if fd > best_fd: best_fd = fd; best_w = [w1, w2, w3, w4]
    
    dw[d] = best_w if best_w else [0.40, 0.10, 0.15, 0.35]
    print(f"  {d:12s}: w=[{dw[d][0]:.2f}, {dw[d][1]:.2f}, {dw[d][2]:.2f}, {dw[d][3]:.2f}]")

# Apply DAMSE
damse_probs = np.zeros(len(df_test))
for i in range(len(df_test)):
    w = dw.get(d_test[i], [0.40, 0.10, 0.15, 0.35])
    damse_probs[i] = w[0]*s1_probs[i] + w[1]*s2_probs[i] + w[2]*s3_probs[i] + w[3]*s4_probs[i]

damse_preds = (damse_probs >= 0.5).astype(int)
damse_labels = df_test['label_binary'].values

print(f"\n  DAMSE Test Acc: {accuracy_score(damse_labels, damse_preds):.4f}  "
      f"F1: {f1_score(damse_labels, damse_preds, average='weighted'):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "S1: AraBERT": {"acc": float(accuracy_score(s1_labels, s1_preds)),
                     "f1": float(f1_score(s1_labels, s1_preds, average='weighted'))},
    "S2: Zero-Shot": {"acc": float(accuracy_score(s2_labels, s2_preds)),
                       "f1": float(f1_score(s2_labels, s2_preds, average='weighted'))},
    "S3: Few-Shot": {"acc": float(accuracy_score(s3_labels, s3_preds)),
                      "f1": float(f1_score(s3_labels, s3_preds, average='weighted'))},
    "S4: GBM": {"acc": float(accuracy_score(s4_labels, s4_preds)),
                 "f1": float(f1_score(s4_labels, s4_preds, average='weighted'))},
    "S5: DAMSE": {"acc": float(accuracy_score(damse_labels, damse_preds)),
                   "f1": float(f1_score(damse_labels, damse_preds, average='weighted'))}
}

with open(os.path.join(Config.OUTPUT_DIR, "results_v2.json"), 'w') as f:
    json.dump(results, f, indent=2)

with open(os.path.join(Config.OUTPUT_DIR, "dialect_weights_v2.json"), 'w') as f:
    json.dump({k: [round(v, 3) for v in vals] for k, vals in dw.items()}, f, indent=2)

print(f"""
{'=' * 70}
  ✅ DAMSE v2 COMPLETE
{'=' * 70}

  Results saved to {Config.OUTPUT_DIR}/
  - results_v2.json
  - dialect_weights_v2.json
{'=' * 70}
""")
