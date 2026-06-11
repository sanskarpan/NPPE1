#!/usr/bin/env python3
"""Generate the final submission notebook as .ipynb"""
import json

cells = []

def code_cell(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": source
    })

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 0: Environment detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# ENVIRONMENT DETECTION
# ============================================================
import os, sys

IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    DATA_DIR   = '/kaggle/input/competitions/26-t-1-dl-gen-ainppe-1'
    IMAGE_DIR  = os.path.join(DATA_DIR, 'images')
    TRAIN_CSV  = os.path.join(DATA_DIR, 'train.csv')
    TEST_CSV   = os.path.join(DATA_DIR, 'test.csv')
    SAMPLE_CSV = os.path.join(DATA_DIR, 'sample_submission.csv')
    OUTPUT_DIR = '/kaggle/working'
else:
    DATA_DIR   = '/Users/sanskar/dev/NPPE1'
    IMAGE_DIR  = os.path.join(DATA_DIR, 'images')
    TRAIN_CSV  = os.path.join(DATA_DIR, 'train.csv')
    TEST_CSV   = os.path.join(DATA_DIR, 'test.csv')
    SAMPLE_CSV = os.path.join(DATA_DIR, 'sample_submission.csv')
    OUTPUT_DIR = DATA_DIR

print(f'Environment: {"Kaggle" if IS_KAGGLE else "Local"}')
print(f'Data dir: {DATA_DIR}')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1: Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# IMPORTS
# ============================================================
if IS_KAGGLE:
    os.system('pip install timm -q')

import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as T
import timm
from torch.optim.lr_scheduler import OneCycleLR
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f'PyTorch {torch.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print('Imports done.')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2: Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# CONFIG — optimized for score, not overfitting
# ============================================================
CFG = {
    'img_size'       : 384,
    'train_img_size' : 320,
    'val_img_size'   : 384,
    'num_classes'    : 20,
    'seed'           : 42,

    # Model
    'backbone'       : 'tf_efficientnet_b4_ns',
    'pretrained'     : True,
    'dropout'        : 0.3,

    # Training — 15 epochs saves ~2.5 hrs vs 30
    'epochs'         : 15,
    'batch_size'     : 32,
    'val_batch_size' : 64,
    'num_workers'    : 4,
    'lr'             : 3e-4,
    'weight_decay'   : 1e-4,
    'label_smoothing': 0.1,

    # NO class weights in loss — let the model learn true distribution
    # The optimal decision rule at inference handles the asymmetric scoring
    'use_class_weights': False,

    # NO mixup — harmful with extreme class imbalance
    'mixup_alpha'    : 0.0,

    # TTA
    'use_tta'        : True,
    'tta_n'          : 4,

    # Val split
    'val_fold'       : 0,
    'n_folds'        : 5,
}

# Column order MUST match CSV columns exactly
CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
    'Pneumoperitoneum', 'Pneumomediastinum', 'Subcutaneous Emphysema',
    'Tortuous Aorta', 'Calcification of the Aorta', 'No Finding'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
print(f'Epochs: {CFG["epochs"]} | Batch: {CFG["batch_size"]} | LR: {CFG["lr"]}')
print(f'Class weights: {CFG["use_class_weights"]} | Mixup: {CFG["mixup_alpha"]}')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3: Load data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# LOAD DATA
# ============================================================
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f'Train: {train_df.shape}  |  Test: {test_df.shape}')

# Verify single-label
row_sums = train_df[CLASSES].sum(axis=1)
assert (row_sums == 1).all(), f'NOT single-label! Found {(row_sums != 1).sum()} bad rows'
print('Verified: single-label (every row sums to 1)')

# Class distribution
class_counts = train_df[CLASSES].sum().sort_values(ascending=False)
class_counts_arr = np.array([class_counts[cls] for cls in CLASSES])
print(f'\\nClass distribution:')
for cls in class_counts.index:
    n = int(class_counts[cls])
    pct = n / len(train_df) * 100
    print(f'  {cls:<35} {n:>6}  ({pct:>6.2f}%)')
print(f'\\nImbalance ratio: {class_counts.max()}/{class_counts.min()} = {class_counts.max()/class_counts.min():.0f}x')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4: Scoring functions + strategy finder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# SCORING FUNCTIONS + AUTOMATED STRATEGY SELECTION
# ============================================================
def competition_score(y_true, y_pred, classes=CLASSES, verbose=False):
    C = len(classes)
    scores = []
    for c_idx, cls in enumerate(classes):
        N_c = (y_true == c_idx).sum()
        if N_c == 0:
            continue
        TP = ((y_true == c_idx) & (y_pred == c_idx)).sum()
        FP = ((y_true != c_idx) & (y_pred == c_idx)).sum()
        FN = ((y_true == c_idx) & (y_pred != c_idx)).sum()
        score_c = (TP - FP - 5 * FN) / N_c
        scores.append(score_c)
        if verbose:
            print(f'{cls:<35} N={N_c:5d} | TP={TP:5d} FP={FP:5d} FN={FN:5d} | Score={score_c:+8.4f}')
    macro_avg = np.mean(scores)
    if verbose:
        print('-' * 80)
        print(f'{\"MACRO AVERAGE\":<35}                                | Score={macro_avg:+8.4f}')
    return macro_avg


def optimal_predict(probs, class_counts, min_train_samples=30):
    \"\"\"
    Bayes-optimal: argmax_c [(7*P(c|x) - 1) / N_c]
    Ultra-rare classes (N < min_train_samples) have N_c clamped to prevent
    runaway FP from the 1/N_c boost.
    \"\"\"
    effective_counts = np.maximum(class_counts, min_train_samples).astype(float)
    decision_scores = (7 * probs - 1) / effective_counts[np.newaxis, :]
    return np.argmax(decision_scores, axis=1)


def find_best_strategy(val_probs, val_labels, class_counts):
    \"\"\"Try multiple prediction strategies and return the best one.\"\"\"
    strategies = {}
    nf_idx = CLASSES.index('No Finding')

    # 1. Argmax
    preds = val_probs.argmax(axis=1)
    strategies['argmax'] = competition_score(val_labels, preds)

    # 2. Optimal rule with various min_train thresholds
    for min_t in [20, 30, 50, 100, 200]:
        preds = optimal_predict(val_probs, class_counts, min_train_samples=min_t)
        strategies[f'optimal_min{min_t}'] = competition_score(val_labels, preds)

    # 3. Confidence threshold: only predict disease if max_prob > threshold
    for thresh in [0.3, 0.4, 0.5, 0.6]:
        preds = val_probs.argmax(axis=1)
        max_p = val_probs.max(axis=1)
        preds[max_p < thresh] = nf_idx  # not confident → No Finding
        strategies[f'thresh_{thresh}'] = competition_score(val_labels, preds)

    # 4. Always No Finding (safety baseline)
    preds = np.full(len(val_labels), nf_idx)
    strategies['always_NF'] = competition_score(val_labels, preds)

    # Sort and print
    ranked = sorted(strategies.items(), key=lambda x: x[1], reverse=True)
    print('=== Strategy Comparison ===')
    for name, score in ranked:
        marker = ' ← BEST' if name == ranked[0][0] else ''
        print(f'  {name:<25} {score:+8.4f}{marker}')

    best_name = ranked[0][0]
    print(f'\\nUsing strategy: {best_name} (score={ranked[0][1]:+.4f})')
    return best_name, strategies


print('Scoring functions defined.')
# Quick sanity test
y_t = np.array([0, 0, 1, 1, 2])
y_p = np.array([0, 1, 1, 1, 2])
s = competition_score(y_t, y_p, classes=['A','B','C'])
print(f'Sanity check score: {s:.4f}')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5: Train/val split
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# TRAIN / VALIDATION SPLIT (Stratified K-Fold)
# ============================================================
from sklearn.model_selection import StratifiedKFold

y_labels = train_df[CLASSES].values.argmax(axis=1)
train_df['label'] = y_labels

skf = StratifiedKFold(n_splits=CFG['n_folds'], shuffle=True, random_state=CFG['seed'])
train_df['fold'] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_df, y_labels)):
    train_df.loc[val_idx, 'fold'] = fold

df_trn = train_df[train_df['fold'] != CFG['val_fold']].reset_index(drop=True)
df_val = train_df[train_df['fold'] == CFG['val_fold']].reset_index(drop=True)

print(f'Train: {len(df_trn):,} | Val: {len(df_val):,}')
print(f'Val NF%: {(df_val[\"label\"]==CLASSES.index(\"No Finding\")).mean()*100:.1f}%')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6: Transforms
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# TRANSFORMS
# ============================================================
NORM_MEAN = [0.485, 0.456, 0.406]  # ImageNet stats for pretrained
NORM_STD  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((CFG['train_img_size'], CFG['train_img_size'])),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    T.RandomErasing(p=0.15, scale=(0.02, 0.08)),
])

val_transform = T.Compose([
    T.Resize((CFG['val_img_size'], CFG['val_img_size'])),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

tta_transforms = [
    val_transform,
    T.Compose([T.Resize((CFG['val_img_size'], CFG['val_img_size'])),
               T.RandomHorizontalFlip(p=1.0),
               T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD)]),
    T.Compose([T.Resize((int(CFG['val_img_size']*1.1), int(CFG['val_img_size']*1.1))),
               T.CenterCrop(CFG['val_img_size']),
               T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD)]),
    T.Compose([T.Resize((CFG['val_img_size'], CFG['val_img_size'])),
               T.RandomRotation(degrees=(10, 10)),
               T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD)]),
]

print(f'Train: {CFG[\"train_img_size\"]}px + aug | Val: {CFG[\"val_img_size\"]}px | TTA: {len(tta_transforms)}')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7: Dataset + Dataloaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# DATASET & DATALOADERS
# ============================================================
class ChestXRayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.image_dir, row['id'])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, row['id']
        return img, int(row['label'])

# Natural sampling — NO WeightedRandomSampler
# Class imbalance is handled by the optimal decision rule at inference
train_ds = ChestXRayDataset(df_trn, IMAGE_DIR, train_transform)
val_ds   = ChestXRayDataset(df_val, IMAGE_DIR, val_transform)

_pin = (DEVICE.type == 'cuda')

train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                          shuffle=True, num_workers=CFG['num_workers'],
                          pin_memory=_pin, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=CFG['val_batch_size'],
                          shuffle=False, num_workers=CFG['num_workers'],
                          pin_memory=_pin)

print(f'Train: {len(train_loader)} batches × {CFG[\"batch_size\"]}')
print(f'Val:   {len(val_loader)} batches × {CFG[\"val_batch_size\"]}')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8: Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# MODEL
# ============================================================
class ChestXRayModel(nn.Module):
    def __init__(self, backbone_name, num_classes, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            num_classes=0, global_pool='avg',
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        return self.head(self.backbone(x))

model = ChestXRayModel(CFG['backbone'], CFG['num_classes'],
                        CFG['pretrained'], CFG['dropout']).to(DEVICE)

total_p = sum(p.numel() for p in model.parameters())
print(f'Model: {CFG[\"backbone\"]} | Params: {total_p:,}')

# Test forward
with torch.no_grad():
    out = model(torch.randn(2, 3, CFG['train_img_size'], CFG['train_img_size']).to(DEVICE))
print(f'Output: {out.shape}')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 9: Loss + Optimizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# LOSS + OPTIMIZER + SCHEDULER
# ============================================================
# Plain CrossEntropyLoss — NO class weights, NO focal loss
# Rationale: class weights destroyed calibration in the -5 scoring run.
# A well-calibrated model + optimal decision rule at inference is the
# theoretically correct approach for this asymmetric scoring function.
criterion = nn.CrossEntropyLoss(label_smoothing=CFG['label_smoothing'])
print(f'Loss: CrossEntropyLoss(label_smoothing={CFG[\"label_smoothing\"]})')

# Differential learning rates
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': CFG['lr'] * 0.1},
    {'params': model.head.parameters(),     'lr': CFG['lr']},
], weight_decay=CFG['weight_decay'])

scheduler = OneCycleLR(
    optimizer,
    max_lr=[CFG['lr'] * 0.1, CFG['lr']],
    epochs=CFG['epochs'],
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1e4,
)

_amp_device = DEVICE.type if DEVICE.type in ('cuda', 'cpu') else 'cpu'
scaler = GradScaler(_amp_device, enabled=(DEVICE.type == 'cuda'))

print(f'Optimizer: AdamW | Backbone LR: {CFG[\"lr\"]*0.1:.1e} | Head LR: {CFG[\"lr\"]:.1e}')
print(f'Scheduler: OneCycleLR ({CFG[\"epochs\"]} epochs)')
print(f'AMP: {DEVICE.type == \"cuda\"}')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 10: Train/Val functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# TRAINING & VALIDATION FUNCTIONS (no mixup)
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, epoch, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    _amp = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(_amp, enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

        if batch_idx % 200 == 0:
            lr = scheduler.get_last_lr()[-1]
            print(f'  [{batch_idx:4d}/{len(loader)}] Loss:{total_loss/total:.4f} '
                  f'Acc:{100*correct/total:.1f}% LR:{lr:.2e}')

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    _amp = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(_amp, enabled=(device.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return (total_loss / len(loader.dataset),
            np.concatenate(all_probs),
            np.concatenate(all_labels))

print('Training functions defined.')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 11: Training loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# MAIN TRAINING LOOP
# ============================================================
best_score = -float('inf')
best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_score': [], 'lr': []}

print(f'Training {CFG[\"epochs\"]} epochs...')
print('=' * 70)

for epoch in range(1, CFG['epochs'] + 1):
    print(f'\\nEpoch {epoch}/{CFG[\"epochs\"]}')
    print('-' * 40)

    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, scheduler, scaler, criterion, epoch, DEVICE)

    val_loss, val_probs, val_labels = validate(model, val_loader, criterion, DEVICE)

    # Compute val score using argmax (most stable for model selection)
    val_preds = val_probs.argmax(axis=1)
    val_acc = (val_preds == val_labels).mean()
    val_score = competition_score(val_labels, val_preds)

    print(f'  Train Loss:{train_loss:.4f} Acc:{train_acc*100:.1f}%')
    print(f'  Val   Loss:{val_loss:.4f} Acc:{val_acc*100:.1f}% Score(argmax):{val_score:+.4f}')

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_score'].append(val_score)
    history['lr'].append(scheduler.get_last_lr()[-1])

    if val_score > best_score:
        best_score = val_score
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': best_score,
        }, best_model_path)
        print(f'  ★ New best! Score={best_score:+.4f}')

print(f'\\nTraining complete. Best score: {best_score:+.4f}')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 12: Training curves
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# TRAINING CURVES
# ============================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ep = range(1, len(history['train_loss']) + 1)

axes[0].plot(ep, history['train_loss'], 'b-o', label='Train', markersize=3)
axes[0].plot(ep, history['val_loss'],   'r-o', label='Val', markersize=3)
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, [s for s in history['val_score']], 'g-o', markersize=3)
axes[1].axhline(y=best_score, color='red', ls='--', label=f'Best: {best_score:+.4f}')
axes[1].axhline(y=-4.0, color='orange', ls='--', label='Cutoff: -4.0')
axes[1].set_title('Val Competition Score'); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(ep, [a*100 for a in history['val_acc']], 'm-o', markersize=3)
axes[2].set_title('Val Accuracy (%)'); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=100)
plt.show()
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 13: Load best model + strategy selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# LOAD BEST MODEL + FIND BEST STRATEGY
# ============================================================
checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f'Loaded best model from epoch {checkpoint[\"epoch\"]} (score={checkpoint[\"score\"]:+.4f})')

# Full validation
_, val_probs, val_labels = validate(model, val_loader, criterion, DEVICE)

# Detailed per-class analysis
print('\\n=== Per-class scores (argmax) ===')
_ = competition_score(val_labels, val_probs.argmax(axis=1), verbose=True)

# Find best strategy automatically
print()
best_strategy, all_strategies = find_best_strategy(val_probs, val_labels, class_counts_arr)
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 14: TTA Inference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# INFERENCE WITH TTA
# ============================================================
def predict_tta(model, image_ids, image_dir, transforms, batch_size, device, num_workers=4):
    model.eval()
    all_avg = None
    _amp = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    for t_idx, tfm in enumerate(transforms):
        print(f'  TTA {t_idx+1}/{len(transforms)}...')
        ds = ChestXRayDataset(
            pd.DataFrame({'id': image_ids, 'label': 0}),
            image_dir, tfm, is_test=False  # is_test=False so we get integer labels
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=(device.type=='cuda'))

        probs_list = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                with autocast(_amp, enabled=(device.type=='cuda')):
                    logits = model(images)
                probs_list.append(F.softmax(logits, dim=1).cpu().numpy())

        probs_arr = np.concatenate(probs_list)
        all_avg = probs_arr if all_avg is None else all_avg + probs_arr

    return all_avg / len(transforms)


print('Running TTA on test set...')
test_ids = test_df['id'].tolist()
test_probs = predict_tta(
    model, test_ids, IMAGE_DIR,
    tta_transforms[:CFG['tta_n']],
    CFG['val_batch_size'], DEVICE, CFG['num_workers']
)
print(f'Test probs shape: {test_probs.shape}')
print(f'Probs sum check: {test_probs.sum(axis=1).mean():.4f} (should be ~1.0)')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 15: Build submission
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# BUILD SUBMISSION — using best strategy from validation
# ============================================================
# CRITICAL: Build from test_df (17015 rows), NOT sample_submission.csv (10 rows)!

# Apply the best strategy found on validation
if best_strategy == 'argmax':
    test_preds = test_probs.argmax(axis=1)
elif best_strategy.startswith('optimal_min'):
    min_t = int(best_strategy.split('min')[1])
    test_preds = optimal_predict(test_probs, class_counts_arr, min_train_samples=min_t)
elif best_strategy.startswith('thresh_'):
    thresh = float(best_strategy.split('_')[1])
    test_preds = test_probs.argmax(axis=1)
    nf_idx = CLASSES.index('No Finding')
    test_preds[test_probs.max(axis=1) < thresh] = nf_idx
elif best_strategy == 'always_NF':
    test_preds = np.full(len(test_ids), CLASSES.index('No Finding'))

# Build one-hot submission dataframe
submission = pd.DataFrame({'id': test_ids})
for cls in CLASSES:
    submission[cls] = 0

for i, pred_idx in enumerate(test_preds):
    submission.iloc[i, 1 + pred_idx] = 1  # +1 because col 0 is 'id'

# Verify
assert len(submission) == len(test_df), f'Row count mismatch: {len(submission)} vs {len(test_df)}'
assert (submission[CLASSES].sum(axis=1) == 1).all(), 'Not all rows single-label!'

# Save
submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
submission.to_csv(submission_path, index=False)
print(f'Submission saved: {submission_path} ({len(submission)} rows)')

# Distribution check
print(f'\\nPrediction distribution:')
for cls in CLASSES:
    n = submission[cls].sum()
    if n > 0:
        print(f'  {cls:<35} {n:>5} ({n/len(submission)*100:.1f}%)')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 16: Also save argmax submission as backup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_cell("""\
# ============================================================
# BACKUP: Save argmax submission too (safest baseline)
# ============================================================
argmax_preds = test_probs.argmax(axis=1)
sub_argmax = pd.DataFrame({'id': test_ids})
for cls in CLASSES:
    sub_argmax[cls] = 0
for i, pred_idx in enumerate(argmax_preds):
    sub_argmax.iloc[i, 1 + pred_idx] = 1

sub_argmax_path = os.path.join(OUTPUT_DIR, 'submission_argmax.csv')
sub_argmax.to_csv(sub_argmax_path, index=False)
print(f'Backup argmax submission: {sub_argmax_path} ({len(sub_argmax)} rows)')

# Also save an optimal submission with a conservative min_train
opt_preds = optimal_predict(test_probs, class_counts_arr, min_train_samples=50)
sub_opt = pd.DataFrame({'id': test_ids})
for cls in CLASSES:
    sub_opt[cls] = 0
for i, pred_idx in enumerate(opt_preds):
    sub_opt.iloc[i, 1 + pred_idx] = 1

sub_opt_path = os.path.join(OUTPUT_DIR, 'submission_optimal.csv')
sub_opt.to_csv(sub_opt_path, index=False)
print(f'Optimal submission: {sub_opt_path} ({len(sub_opt)} rows)')

print('\\n=== ALL SUBMISSIONS READY ===')
print(f'  1. submission.csv         — best strategy ({best_strategy})')
print(f'  2. submission_argmax.csv  — safe argmax baseline')
print(f'  3. submission_optimal.csv — optimal rule (min_train=50)')
print(f'\\nTotal test rows: {len(test_df)} | All submissions verified.')
""")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Assemble notebook
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells
}

out_path = '/Users/sanskar/dev/NPPE1/final_submission_notebook.ipynb'
with open(out_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f'Notebook written: {out_path}')
print(f'Total cells: {len(cells)}')
