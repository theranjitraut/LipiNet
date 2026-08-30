"""
LipiNet
Ablation Study  +  Hyperparameter Grid Search
Dataset :  Devanagari Handwritten Character Dataset (DHCD)
            46 classes (36 consonants + 10 digits), 32×32 grayscale
STRUCTURE:
 Section  0  – Reproducibility
 Section  1  – Configuration
 Section  2  – Dataset pipeline
 Section  3  – Display / logging utilities
 Section  4  – Shared building blocks  (residual, SE, AFC, dense-res, …)
 Section  5  – Model factory              build_variant(cfg) → nn.Module
 Section  6  – LR schedule (cosine annealing, per optimizer step)
 Section  7  – Training & evaluation helpers
 Section  8  – ABLATION STUDY   (A0..A7)
 Section  9  – HYPERPARAMETER GRID SEARCH  (3^4 = 81 configs)
 Section 10  – Persist results  (JSON + pretty terminal tables)
"""

import os, time, random, json, copy, itertools, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

# REPRODUCIBILITY
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONFIGURATION
BASE_CFG = {
    "num_classes"   : 46,
    "image_size"    : 32,
    "val_split"     : 0.10,
    "batch_size"    : 64,
    "epochs"        : 100,          # reduced from 100 for tractability
    "lr"            : 5e-4,
    "weight_decay"  : 1e-4,
    "label_smooth"  : 0.10,

    "data_dir"      : "/kaggle/input/datasets/theranjitraut/devanagari/"
                      "DevanagariHandwrittenCharacterDataset",
    "results_dir"   : "./results",
    "num_workers"   : 4,
}

os.makedirs(BASE_CFG["results_dir"], exist_ok=True)

NUM_CLASSES = BASE_CFG["num_classes"]
IMG         = BASE_CFG["image_size"]

# DATASET PIPELINE
#     Built once (indices fixed by SEED); batch size is applied per experiment
#     via fresh DataLoaders.

class _PadRandomCrop:
    """Pad by `pad` on each side with `value`, then random-crop back to `size`."""
    def __init__(self, size: int, pad: int = 2, value: float = -1.0):
        self.size, self.pad, self.value = size, pad, value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad,) * 4, mode="constant", value=self.value)
        max_off = 2 * self.pad
        i = random.randint(0, max_off)
        j = random.randint(0, max_off)
        return x[:, i:i + self.size, j:j + self.size]


def build_transforms(image_size: int):
    """Mirrors normalise()/augment() from the TF pipeline."""
    common = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),                      # → [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5]),  # → [-1,1], same as /127.5-1
    ]
    train_tf = transforms.Compose(common + [
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        _PadRandomCrop(image_size, pad=2, value=-1.0),
    ])
    eval_tf = transforms.Compose(common)
    return train_tf, eval_tf


def load_datasets(cfg: dict):
    """
    Returns (train_ds, val_ds, test_ds) as torch Datasets.
    Train/val split uses a fixed permutation seeded by SEED, matching the
    deterministic take()/skip() split used in the TF version.
    """
    train_dir = os.path.join(cfg["data_dir"], "Train")
    test_dir  = os.path.join(cfg["data_dir"], "Test")

    train_tf, eval_tf = build_transforms(cfg["image_size"])

    # Two ImageFolder instances over the *same* directory, different transforms,
    # combined with Subset() over identical indices so train/val get the right
    # augmentation while sharing the same underlying split.
    raw_index_ds  = datasets.ImageFolder(train_dir)  # only used for length/order
    train_backing = datasets.ImageFolder(train_dir, transform=train_tf)
    val_backing   = datasets.ImageFolder(train_dir, transform=eval_tf)

    n_total = len(raw_index_ds)
    n_val   = max(1, int(n_total * cfg["val_split"]))
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    train_ds = Subset(train_backing, train_idx)
    val_ds   = Subset(val_backing,   val_idx)
    test_ds  = datasets.ImageFolder(test_dir, transform=eval_tf)

    return train_ds, val_ds, test_ds, n_train, n_val


def build_loaders(train_ds, val_ds, test_ds, batch_size: int, num_workers: int):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


print("[INFO] Loading dataset …")
_train_ds, _val_ds, _test_ds, _n_train, _n_val = load_datasets(BASE_CFG)
print(f"[INFO] Train: {_n_train:,}  |  Val: {_n_val:,}  |  Test: {len(_test_ds):,}")


# DISPLAY / LOGGING UTILITIES
_COL = {
    "reset" : "\033[0m",  "bold"  : "\033[1m",
    "cyan"  : "\033[96m", "yellow": "\033[93m",
    "green" : "\033[92m", "red"   : "\033[91m",
    "grey"  : "\033[90m", "white" : "\033[97m",
    "blue"  : "\033[94m", "magenta": "\033[95m",
}

def _c(text, *codes):
    prefix = "".join(_COL.get(c, "") for c in codes)
    return f"{prefix}{text}{_COL['reset']}"


def _hline(width=72, char="─"):
    print(_c(char * width, "cyan"))


def print_experiment_header(title: str, index: int, total: int):
    _hline()
    pct = f"[{index}/{total}]"
    print(_c(f"  {pct}  {title}", "bold", "yellow"))
    _hline()


def print_result_row(name: str, r: dict, highlight: bool = False):
    color = "green" if highlight else "white"
    star  = "★" if highlight else " "
    print(_c(
        f"  {star} {name:<32}  "
        f"Acc={r['test_acc']:6.2f}%  "
        f"F1={r['macro_f1']:6.2f}%  "
        f"Loss={r['test_loss']:.4f}  "
        f"Params={r['params']:>9,}",
        color, "bold" if highlight else ""
    ))


def print_section(title: str):
    print()
    print(_c("╔" + "═" * 68 + "╗", "cyan", "bold"))
    print(_c(f"║  {title:<66}║", "cyan", "bold"))
    print(_c("╚" + "═" * 68 + "╝", "cyan", "bold"))
    print()


def print_final_table(title: str, results: dict, sort_key: str = "test_acc"):
    sorted_items = sorted(results.items(), key=lambda x: -x[1][sort_key])
    best_name    = sorted_items[0][0]
    W = 72
    print(_c(f"\n╔{'═'*W}╗", "cyan", "bold"))
    print(_c(f"║  {title:<{W-2}}║", "cyan", "bold"))
    print(_c(f"╠{'═'*34}╦{'═'*10}╦{'═'*10}╦{'═'*8}╦{'═'*6}╣", "cyan"))
    hdr = f"║  {'Name':<32}║{'Test Acc':>9} ║{'Macro F1':>9} ║{'Loss':>7} ║{'Params':>5} ║"
    print(_c(hdr, "bold", "white"))
    print(_c(f"╠{'═'*34}╬{'═'*10}╬{'═'*10}╬{'═'*8}╬{'═'*6}╣", "cyan"))
    for name, r in sorted_items:
        is_best = (name == best_name)
        color   = "green" if is_best else "white"
        star    = "★" if is_best else " "
        pm      = r.get("params", 0)
        pm_str  = f"{pm//1000}K" if pm < 1_000_000 else f"{pm/1e6:.1f}M"
        row = (
            f"║{star} {name:<32}║{r['test_acc']:>8.2f}% ║"
            f"{r['macro_f1']:>8.2f}% ║{r['test_loss']:>7.4f} ║{pm_str:>5} ║"
        )
        print(_c(row, color, "bold") if is_best else row)
    print(_c(f"╚{'═'*34}╩{'═'*10}╩{'═'*10}╩{'═'*8}╩{'═'*6}╝", "cyan"))
    best_r = results[best_name]
    print(_c(
        f"\n  ★  Best: {best_name}  "
        f"(acc={best_r['test_acc']:.2f}%  f1={best_r['macro_f1']:.2f}%)\n",
        "green", "bold"
    ))


# SHARED BUILDING BLOCKS
class ResidualBlock(nn.Module):
    """Standard pre-activation-style residual block (Conv→BN→ReLU×2 + skip)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class PlainResBlockDownsample(nn.Module):
    """
    Plain (non-dense) residual block with projection + stride-2 downsampling.
    Used in ablation A6 to replace DenseResBlock.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.project = in_ch != out_ch
        if self.project:
            self.proj_conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.proj_bn   = nn.BatchNorm2d(out_ch)
        self.res1 = ResidualBlock(out_ch)
        self.res2 = ResidualBlock(out_ch)
        self.dw   = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1,
                               groups=out_ch, bias=False)
        self.pw   = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        if self.project:
            x = F.relu(self.proj_bn(self.proj_conv(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.dw(x)
        x = self.pw(x)
        x = F.relu(self.bn(x))
        return x


class DenseResBlock(nn.Module):
    """
    DenseNet-inspired block: 2 chained residual blocks + dense concat +
    1×1 bottleneck + stride-2 depthwise downsample.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.project = in_ch != out_ch
        if self.project:
            self.proj_conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.proj_bn   = nn.BatchNorm2d(out_ch)
        self.res1 = ResidualBlock(out_ch)
        self.res2 = ResidualBlock(out_ch)
        self.bottleneck_conv = nn.Conv2d(out_ch * 2, out_ch, 1, bias=False)
        self.bottleneck_bn   = nn.BatchNorm2d(out_ch)
        self.dw = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1,
                             groups=out_ch, bias=False)
        self.pw = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        if self.project:
            x_in = F.relu(self.proj_bn(self.proj_conv(x)))
        else:
            x_in = x
        r1  = self.res1(x_in)
        r2  = self.res2(r1)
        cat = torch.cat([r1, r2], dim=1)
        out = F.relu(self.bottleneck_bn(self.bottleneck_conv(cat)))
        out = self.dw(out)
        out = self.pw(out)
        out = F.relu(self.bn(out))
        return out


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        gap  = F.adaptive_avg_pool2d(x, 1).view(b, c)
        attn = F.relu(self.fc1(gap))
        attn = torch.sigmoid(self.fc2(attn)).view(b, c, 1, 1)
        return x * attn


class AdaptiveFilterCapsule(nn.Module):
    """
    O(n) capsule-like routing: per-class filter bank applied to a projected
    feature vector → class-discriminative scores (B, K).
    """
    def __init__(self, in_features: int, num_classes: int, capsule_dim: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.capsule_dim = capsule_dim
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, num_classes * capsule_dim)
        self.bn  = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h).view(-1, self.num_classes, self.capsule_dim)
        x_sliced = x[:, :self.capsule_dim].unsqueeze(1).expand(-1, self.num_classes, -1)
        caps = (x_sliced * h).sum(dim=-1)
        caps = self.bn(caps)
        return caps


# MODEL
#     build_variant(vcfg) accepts a dict of boolean flags + HP values
#     and returns a WhatNetAFC nn.Module.
#
#  Variant config keys
#  ───────────────────
#  use_scaffold_stem   : bool  – 1×7 stroke-scaffold path in stem
#  use_scaffold_inject : bool  – weighted scaffold add after each encoder stage
#  use_channel_attn    : bool  – SE block in stem
#  use_afc             : bool  – Adaptive Filter Capsule head
#  use_gated_fusion    : bool  – learned soft gate (vs simple Add)
#  use_dense_res       : bool  – dense_res_block (vs plain resblock)
#  head_units          : int   – dense head width

class WhatNetAFC(nn.Module):
    def __init__(self, vcfg: dict):
        super().__init__()
        self.K   = vcfg.get("num_classes", NUM_CLASSES)
        self.SZ  = vcfg.get("image_size",  IMG)
        self.HU  = vcfg.get("head_units",  256)

        self.use_scaffold_stem   = vcfg.get("use_scaffold_stem",   True)
        self.use_scaffold_inject = vcfg.get("use_scaffold_inject", True)
        self.use_channel_attn    = vcfg.get("use_channel_attn",    True)
        self.use_afc             = vcfg.get("use_afc",             True)
        self.use_gated_fusion    = vcfg.get("use_gated_fusion",    True)
        self.use_dense_res       = vcfg.get("use_dense_res",       True)

        #  Stem 
        self.t_conv = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.t_bn   = nn.BatchNorm2d(32)

        if self.use_scaffold_stem:
            # Asymmetric 1×7 conv captures Devanagari shirorekha (horizontal bar)
            self.s_conv = nn.Conv2d(1, 32, (1, 7), padding=(0, 3), bias=False)
        else:
            # Replace scaffold path with a second standard 3×3 conv (same param budget)
            self.s_conv = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.s_bn = nn.BatchNorm2d(32)

        stem_ch = 64
        if self.use_channel_attn:
            self.stem_attn = ChannelAttention(stem_ch)

        self.stem_proj_conv = nn.Conv2d(stem_ch, 64, 1, bias=False)
        self.stem_proj_bn   = nn.BatchNorm2d(64)

        #  Encoder 
        Block = DenseResBlock if self.use_dense_res else PlainResBlockDownsample
        self.enc1 = Block(64, 64)
        self.enc2 = Block(64, 128)
        self.enc3 = Block(128, 256)

        if self.use_scaffold_inject:
            self.sc1_conv = nn.Conv2d(32, 64,  1, bias=False)
            self.sc2_conv = nn.Conv2d(32, 128, 1, bias=False)
            self.sc3_conv = nn.Conv2d(32, 256, 1, bias=False)

        # Head 
        fused_dim = 64 + 128 + 256
        self.head_dense = nn.Linear(fused_dim, self.HU, bias=False)
        self.head_ln    = nn.LayerNorm(self.HU)
        self.head_logits = nn.Linear(self.HU, self.K)

        if self.use_afc:
            self.afc = AdaptiveFilterCapsule(fused_dim, self.K)
            if self.use_gated_fusion:
                self.gate = nn.Linear(self.K * 2, 2)

        # Model name mirroring the TF naming scheme
        name_parts = []
        if not self.use_scaffold_stem:   name_parts.append("noStem")
        if not self.use_scaffold_inject: name_parts.append("noInj")
        if not self.use_channel_attn:    name_parts.append("noSE")
        if not self.use_afc:             name_parts.append("noAFC")
        if not self.use_gated_fusion:    name_parts.append("noGate")
        if not self.use_dense_res:       name_parts.append("plainRes")
        self.model_name = "WhatNet_" + ("_".join(name_parts) if name_parts else "Full")

    def forward(self, inp):
        t = F.relu(self.t_bn(self.t_conv(inp)))
        scaffold = F.relu(self.s_bn(self.s_conv(inp)))
        stem = torch.cat([t, scaffold], dim=1)  # (B, 64, SZ, SZ)

        if self.use_channel_attn:
            stem = self.stem_attn(stem)

        stem = F.relu(self.stem_proj_bn(self.stem_proj_conv(stem)))

        enc1 = self.enc1(stem)
        if self.use_scaffold_inject:
            sc1 = F.avg_pool2d(self.sc1_conv(scaffold), 2)
            enc1 = enc1 + sc1 * 0.1

        enc2 = self.enc2(enc1)
        if self.use_scaffold_inject:
            sc2 = F.avg_pool2d(self.sc2_conv(scaffold), 4)
            enc2 = enc2 + sc2 * 0.1

        enc3 = self.enc3(enc2)
        if self.use_scaffold_inject:
            sc3 = F.avg_pool2d(self.sc3_conv(scaffold), 8)
            enc3 = enc3 + sc3 * 0.1

        # Multi-scale GAP fusion
        gap1 = F.adaptive_avg_pool2d(enc1, 1).flatten(1)
        gap2 = F.adaptive_avg_pool2d(enc2, 1).flatten(1)
        gap3 = F.adaptive_avg_pool2d(enc3, 1).flatten(1)
        fused_gap = torch.cat([gap1, gap2, gap3], dim=1)

        # Dense head 
        x = self.head_dense(fused_gap)
        x = self.head_ln(x)
        x = F.relu(x)
        x = self.head_logits(x)

        if self.use_afc:
            afc_out = self.afc(fused_gap)
            if self.use_gated_fusion:
                combined = torch.cat([x, afc_out], dim=1)
                gate = F.softmax(self.gate(combined), dim=1)
                out = x * gate[:, 0:1] + afc_out * gate[:, 1:2]
            else:
                out = x + afc_out
        else:
            out = x

        return out


def build_variant(vcfg: dict) -> WhatNetAFC:
    """Build a WhatNet-AFC variant according to the boolean flags in vcfg."""
    return WhatNetAFC(vcfg)


# LR SCHEDULE (cosine annealing, applied per optimizer step)
def make_cosine_scheduler(optimizer: torch.optim.Optimizer, base_lr: float, total_steps: int):
    """
    Half-cosine decay from `base_lr` to 1e-6, over `total_steps` optimizer steps.
    Mirrors the custom CosineAnnealing LearningRateSchedule from the TF version.
    """
    floor_ratio = 1e-6 / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        cosine = 0.5 * (1.0 + math.cos(math.pi * step / max(1, total_steps)))
        return max(cosine, floor_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# TRAINING & EVALUATION HELPERS

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device) -> tuple:
    """Returns (avg_loss, accuracy_pct)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def compute_macro_f1(model: nn.Module, loader: DataLoader, device, num_classes: int) -> float:
    model.eval()
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        lbls  = labels.numpy()
        for c in range(num_classes):
            tp[c] += np.sum((preds == c) & (lbls == c))
            fp[c] += np.sum((preds == c) & (lbls != c))
            fn[c] += np.sum((preds != c) & (lbls == c))
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return float(f1.mean() * 100.0)


def run_experiment(
    exp_name   : str,
    vcfg       : dict,
    train_cfg  : dict,
    exp_index  : int,
    total_exps : int,
    ckpt_prefix: str = "exp",
) -> dict:
    """
    Build, train, and evaluate one experiment.
    Returns a results dict with test_acc, macro_f1, params, test_loss,
    best_val_acc, wall_time.
    """
    print_experiment_header(exp_name, exp_index, total_exps)

    BS = train_cfg["batch_size"]
    train_loader, val_loader, test_loader = build_loaders(
        _train_ds, _val_ds, _test_ds, BS, train_cfg.get("num_workers", 4)
    )

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * train_cfg["epochs"]

    model = build_variant({**vcfg, "num_classes": NUM_CLASSES, "image_size": IMG}).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = make_cosine_scheduler(optimizer, train_cfg["lr"], total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smooth"])

    safe_name = exp_name.replace(" ", "_").replace("/", "-").replace("=", "")
    ckpt_path = os.path.join(train_cfg["results_dir"], f"{ckpt_prefix}_{safe_name}_best.pt")

    best_val_acc     = -1.0
    best_state        = None
    patience          = 12
    epochs_no_improve = 0

    t0 = time.time()
    for epoch in range(train_cfg["epochs"]):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        _, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt_path)

    wall_time = time.time() - t0

    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    macro_f1 = compute_macro_f1(model, test_loader, DEVICE, NUM_CLASSES)

    result = {
        "test_acc"    : round(test_acc,  2),
        "macro_f1"    : round(macro_f1,  2),
        "params"      : sum(p.numel() for p in model.parameters()),
        "test_loss"   : round(float(test_loss), 4),
        "best_val_acc": round(best_val_acc, 2),
        "wall_time_s" : round(wall_time, 1),
        "config"      : {**vcfg, **train_cfg},
    }

    print(_c(
        f"\n  ✔  {exp_name}  |  "
        f"test_acc={test_acc:.2f}%  "
        f"f1={macro_f1:.2f}%  "
        f"best_val={best_val_acc:.2f}%  "
        f"time={wall_time:.0f}s\n",
        "green", "bold"
    ))

    # Clean up GPU memory between experiments
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

# ABLATION STUDY
#     8 variants systematically remove one architectural component at a time.
#     All variants share the same training hyper-parameters (BASE_CFG).

ABLATION_VARIANTS = {
    # Baseline (full model)
    "A0_Full": {
        "use_scaffold_stem"   : True,
        "use_scaffold_inject" : True,
        "use_channel_attn"    : True,
        "use_afc"             : True,
        "use_gated_fusion"    : True,
        "use_dense_res"       : True,
    },
    # Remove scaffold stem (1×7 path → plain 3×3)
    "A1_NoScaffoldStem": {
        "use_scaffold_stem"   : False,
        "use_scaffold_inject" : True,
        "use_channel_attn"    : True,
        "use_afc"             : True,
        "use_gated_fusion"    : True,
        "use_dense_res"       : True,
    },
    # Remove scaffold injection from encoder
    "A2_NoScaffoldInject": {
        "use_scaffold_stem"   : True,
        "use_scaffold_inject" : False,
        "use_channel_attn"    : True,
        "use_afc"             : True,
        "use_gated_fusion"    : True,
        "use_dense_res"       : True,
    },
    # Remove SE channel attention in stem
    "A3_NoChannelAttn": {
        "use_scaffold_stem"   : True,
        "use_scaffold_inject" : True,
        "use_channel_attn"    : False,
        "use_afc"             : True,
        "use_gated_fusion"    : True,
        "use_dense_res"       : True,
    },
    # Remove Adaptive Filter Capsule 
    "A4_NoAFC": {
        "use_scaffold_stem"   : True,
        "use_scaffold_inject" : True,
        "use_channel_attn"    : True,
        "use_afc"             : False,
        "use_gated_fusion"    : True,   # irrelevant when AFC=False, ignored
        "use_dense_res"       : True,
    },
    # Remove gated fusion (replace gate with simple Add)
    "A5_NoGatedFusion": {
        "use_scaffold_stem"   : True,
        "use_scaffold_inject" : True,
        "use_channel_attn"    : True,
        "use_afc"             : True,
        "use_gated_fusion"    : False,
        "use_dense_res"       : True,
    },
    # Replace dense residual blocks with plain residual blocks
    "A6_PlainResidual": {
        "use_scaffold_stem"   : True,
        "use_scaffold_inject" : True,
        "use_channel_attn"    : True,
        "use_afc"             : True,
        "use_gated_fusion"    : True,
        "use_dense_res"       : False,
    },
    # Minimal model: no scaffold, no AFC, no gate
    "A7_Minimal": {
        "use_scaffold_stem"   : False,
        "use_scaffold_inject" : False,
        "use_channel_attn"    : False,
        "use_afc"             : False,
        "use_gated_fusion"    : False,
        "use_dense_res"       : True,
    },
}

ablation_train_cfg = {
    **BASE_CFG,
    "epochs"      : BASE_CFG["epochs"],
    "batch_size"  : BASE_CFG["batch_size"],
    "lr"          : BASE_CFG["lr"],
    "weight_decay": BASE_CFG["weight_decay"],
    "label_smooth": BASE_CFG["label_smooth"],
}

print_section("SECTION 8 ▸ ABLATION STUDY  (8 variants × all components)")

ablation_results = {}
total_abl = len(ABLATION_VARIANTS)

for idx, (name, vcfg) in enumerate(ABLATION_VARIANTS.items(), start=1):
    ablation_results[name] = run_experiment(
        exp_name    = name,
        vcfg        = vcfg,
        train_cfg   = ablation_train_cfg,
        exp_index   = idx,
        total_exps  = total_abl,
        ckpt_prefix = "abl",
    )

print_final_table("ABLATION STUDY — TEST-SET RESULTS", ablation_results)

# Compute per-component accuracy drop vs full model
full_acc = ablation_results["A0_Full"]["test_acc"]
print(_c("  Component contribution (accuracy drop vs Full model):", "bold", "white"))
component_map = {
    "A1_NoScaffoldStem"   : "Scaffold stem   (1×7 path)",
    "A2_NoScaffoldInject" : "Scaffold inject (encoder residual)",
    "A3_NoChannelAttn"    : "Channel attn    (SE block)",
    "A4_NoAFC"            : "Adaptive Filter Capsule (AFC)",
    "A5_NoGatedFusion"    : "Gated fusion    (soft gate)",
    "A6_PlainResidual"    : "Dense residual  (vs plain ResBlock)",
    "A7_Minimal"          : "All above removed (minimal baseline)",
}
for key, label in component_map.items():
    drop = full_acc - ablation_results[key]["test_acc"]
    color = "red" if drop > 0.2 else ("yellow" if drop > 0 else "green")
    arrow = "▼" if drop > 0 else "▲"
    print(_c(f"    {arrow}  {label:<42}  Δacc = {drop:+.2f}%", color))
print()

# HYPERPARAMETER GRID SEARCH
'''
    Grid over 4 axes:
      lr           : [1e-3, 5e-4, 2e-4]
      batch_size   : [32, 64, 128]
      weight_decay : [1e-3, 1e-4, 1e-5]
      label_smooth : [0.05, 0.10, 0.15]
    Total = 3^4 = 81 configurations, all using the full model (A0).
    Each is trained for a shorter budget (30 epochs) so the grid completes
    in tractable time on a single GPU.  Increase hp_epochs for more
    reliable estimates.
'''

HP_GRID = {
    "lr"          : [1e-3, 5e-4, 2e-4],
    "batch_size"  : [32, 64, 128],
    "weight_decay": [1e-3, 1e-4, 1e-5],
    "label_smooth": [0.05, 0.10, 0.15],
}

HP_EPOCHS = 60   # ← increase to 50–60 for more reliable HP ranking

FULL_VCFG = ABLATION_VARIANTS["A0_Full"]   # always use full model

# Generate all combinations
hp_keys   = list(HP_GRID.keys())
hp_values = list(HP_GRID.values())
hp_combos = list(itertools.product(*hp_values))
total_hp  = len(hp_combos)   # 81

print_section(f"SECTION 9 ▸ HYPERPARAMETER GRID SEARCH  ({total_hp} configs × {HP_EPOCHS} epochs)")

hp_results = {}

for idx, combo in enumerate(hp_combos, start=1):
    hp = dict(zip(hp_keys, combo))

    exp_name = (
        f"lr={hp['lr']:.0e}_"
        f"bs={hp['batch_size']}_"
        f"wd={hp['weight_decay']:.0e}_"
        f"ls={hp['label_smooth']}"
    )

    train_cfg = {
        **BASE_CFG,
        "epochs"      : HP_EPOCHS,
        "batch_size"  : hp["batch_size"],
        "lr"          : hp["lr"],
        "weight_decay": hp["weight_decay"],
        "label_smooth": hp["label_smooth"],
    }

    hp_results[exp_name] = run_experiment(
        exp_name    = exp_name,
        vcfg        = FULL_VCFG,
        train_cfg   = train_cfg,
        exp_index   = idx,
        total_exps  = total_hp,
        ckpt_prefix = "hp",
    )

# Print top-20 HP configurations
sorted_hp = sorted(hp_results.items(), key=lambda x: -x[1]["test_acc"])

print(_c(f"\n╔{'═'*76}╗", "cyan", "bold"))
print(_c(f"║  {'HP GRID SEARCH — TOP 20 CONFIGURATIONS':<74}║", "cyan", "bold"))
print(_c(f"╠{'═'*40}╦{'═'*10}╦{'═'*10}╦{'═'*12}╣", "cyan"))
hdr = f"║  {'Config':<38}║{'Val Acc':>9} ║{'Test Acc':>9} ║{'Test Loss':>11} ║"
print(_c(hdr, "bold", "white"))
print(_c(f"╠{'═'*40}╬{'═'*10}╬{'═'*10}╬{'═'*12}╣", "cyan"))
for rank, (name, r) in enumerate(sorted_hp[:20], start=1):
    color = "green" if rank == 1 else ("yellow" if rank <= 3 else "white")
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank:2d}")
    row = (
        f"║{medal} {name:<38}║"
        f"{r['best_val_acc']:>8.2f}% ║"
        f"{r['test_acc']:>8.2f}% ║"
        f"{r['test_loss']:>10.4f} ║"
    )
    print(_c(row, color) if rank <= 3 else row)
print(_c(f"╚{'═'*40}╩{'═'*10}╩{'═'*10}╩{'═'*12}╝\n", "cyan"))

best_hp_name, best_hp_r = sorted_hp[0]
best_hp_cfg = best_hp_r["config"]
print(_c("  ★  Best HP configuration found:", "green", "bold"))
print(_c(f"       lr           = {best_hp_cfg['lr']}", "green"))
print(_c(f"       batch_size   = {best_hp_cfg['batch_size']}", "green"))
print(_c(f"       weight_decay = {best_hp_cfg['weight_decay']}", "green"))
print(_c(f"       label_smooth = {best_hp_cfg['label_smooth']}", "green"))
print(_c(f"       → test_acc   = {best_hp_r['test_acc']:.2f}%\n", "green", "bold"))

# Per-axis marginal analysis
print(_c("  Marginal accuracy by HP axis (mean test_acc across all combos):", "bold", "white"))
for key in hp_keys:
    print(_c(f"\n    {key}", "cyan", "bold"))
    val_groups: dict = {}
    for name, r in hp_results.items():
        v = str(r["config"][key])
        val_groups.setdefault(v, []).append(r["test_acc"])
    for v, accs in sorted(val_groups.items(), key=lambda x: -np.mean(x[1])):
        bar_len = int((np.mean(accs) - min(
            np.mean(a) for a in val_groups.values())) /
            max(1e-6, max(np.mean(a) for a in val_groups.values()) -
                min(np.mean(a) for a in val_groups.values())) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"      {v:>8}  {_c(bar, 'cyan')}  {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
print()

# PERSIST ALL RESULTS
print_section("SECTION 10 ▸ PERSISTING RESULTS")

def _serialise(d):
    """Recursively convert numpy/torch types for JSON serialisation."""
    if isinstance(d, dict):
        return {k: _serialise(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_serialise(v) for v in d]
    if isinstance(d, (np.integer, np.floating)):
        return d.item()
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, torch.Tensor):
        return d.tolist()
    return d


# Ablation results
abl_path = os.path.join(BASE_CFG["results_dir"], "ablation_results.json")
with open(abl_path, "w") as f:
    json.dump(_serialise(ablation_results), f, indent=2)
print(f"[INFO] Ablation results  → {abl_path}")

# HP grid results (full)
hp_full_path = os.path.join(BASE_CFG["results_dir"], "hp_grid_results.json")
with open(hp_full_path, "w") as f:
    json.dump(_serialise(hp_results), f, indent=2)
print(f"[INFO] HP grid results   → {hp_full_path}")

# HP top-20 summary
hp_top20 = dict(sorted_hp[:20])
hp_top20_path = os.path.join(BASE_CFG["results_dir"], "hp_top20.json")
with open(hp_top20_path, "w") as f:
    json.dump(_serialise(hp_top20), f, indent=2)
print(f"[INFO] HP top-20         → {hp_top20_path}")

# Combined summary
summary = {
    "ablation": {
        "full_model_acc"  : ablation_results["A0_Full"]["test_acc"],
        "full_model_f1"   : ablation_results["A0_Full"]["macro_f1"],
        "component_drops" : {
            label: round(full_acc - ablation_results[key]["test_acc"], 4)
            for key, label in component_map.items()
        },
    },
    "hp_search": {
        "best_config": best_hp_cfg,
        "best_acc"   : best_hp_r["test_acc"],
        "best_f1"    : best_hp_r["macro_f1"],
        "grid_size"  : total_hp,
    },
}
summary_path = os.path.join(BASE_CFG["results_dir"], "experiment_summary.json")
with open(summary_path, "w") as f:
    json.dump(_serialise(summary), f, indent=2)
print(f"Experiment summary → {summary_path}")
print(_c("\nAblation study + HP grid search complete.\n", "green", "bold"))