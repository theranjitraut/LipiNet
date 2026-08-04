# importing necessary dependencies
import os, time, random, json
import numpy as np
import urllib.request, zipfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
# from tensorflow.keras.callbacks import ReduceLROnPlateau - initially i thought to use ReduceLROnPlateau but i have used learning rate scheduler such as cosine-annealing or such

# Reproducibility - Fix all random seeds so results are reproducible across runs.
# Need to run it 5 times with different seeds - A single run at 99.75% could be lucky initialization. Run with seeds 42, 123, 7, 2024, 0 and report mean ± std. If you get something like 99.75 ± 0.04% that's a bulletproof claim.
# SEEDS = [42, 123, 7, 2024, 0]
# seed_results = []

# for seed in SEEDS:
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     # rebuild, retrain, evaluate
#     seed_results.append(test_acc)

# print(f"Mean: {np.mean(seed_results):.4f}% ± {np.std(seed_results):.4f}%")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration
CFG = {
    "num_classes":      46,      # DHCD: 36 consonants + 10 digits
    "image_size":       32,      
    "batch_size":       64,
    "epochs":           100,
    "lr":               5e-4,    # peak learning rate for cosine schedule
    "weight_decay":     1e-4,    # AdamW L2 regularisation
    "label_smoothing":  0.1,     # prevents over-confident softmax outputs
    "val_split":        0.1,     # fraction of training data held for validation
    # "data_dir":         "./data/DHCD",
    "data_dir":         "/kaggle/input/datasets/theranjitraut/devanagari/DevanagariHandwrittenCharacterDataset",
    "results_dir":      "./results",
}

os.makedirs(CFG["results_dir"], exist_ok=True)
NUM_CLASSES = CFG["num_classes"]
IMG         = CFG["image_size"]
BS          = CFG["batch_size"]
AUTOTUNE    = tf.data.AUTOTUNE

# Dataset download & pipeline
zip_path = "/kaggle/input/datasets/theranjitraut/devanagari/DevanagariHandwrittenCharacterDataset"

if os.path.exists(CFG["data_dir"]):
    print("DHCD already present – skipping download.")
else:
    print("Downloading DHCD …")
    try:
        # urllib.request.urlretrieve(_DHCD_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("./data/")
        os.rename("./data/DevanagariHandwrittenCharacterDataset", CFG["data_dir"])
        print("[INFO] DHCD extracted successfully.")
    except Exception as exc:
        print(f"[WARN] Download failed: {exc}")

# Loading raw (unbatched) tf.data datasets
train_full = keras.utils.image_dataset_from_directory(
    os.path.join(CFG["data_dir"], "Train"),
    image_size=(IMG, IMG), batch_size=None,
    color_mode="grayscale", label_mode="int", seed=SEED,
)
test_ds_raw = keras.utils.image_dataset_from_directory(
    os.path.join(CFG["data_dir"], "Test"),
    image_size=(IMG, IMG), batch_size=None,
    color_mode="grayscale", label_mode="int", seed=SEED,
)

total   = tf.data.experimental.cardinality(train_full).numpy()
n_val   = max(1, int(total * CFG["val_split"]))
n_train = total - n_val

train_ds_raw = train_full.take(n_train)
val_ds_raw   = train_full.skip(n_train)

# Preprocessing helpers
def normalise(img, lbl):
    """Scale pixels from [0, 255] → [-1, 1]."""
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return img, lbl

def augment(img, lbl):
    """
    Light stochastic augmentation applied only during training.
    Pad-then-crop gives a random translation effect.
    """
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.pad(img, [[2, 2], [2, 2], [0, 0]], constant_values=-1.0)
    img = tf.image.random_crop(img, [IMG, IMG, 1])
    return img, lbl

def to_onehot(img, lbl):
    """
    Convert integer class index → one-hot vector.
    Required because keras.losses.CategoricalCrossentropy expects one-hot
    targets, while SparseCategoricalCrossentropy does NOT support the
    label_smoothing argument.
    """
    return img, tf.one_hot(lbl, NUM_CLASSES)

# Building tf.data pipelines
train_ds = (
    train_ds_raw
    .map(normalise,  num_parallel_calls=AUTOTUNE)
    .map(augment,    num_parallel_calls=AUTOTUNE)
    .map(to_onehot,  num_parallel_calls=AUTOTUNE)
    .shuffle(8192, seed=SEED)
    .batch(BS)
    .prefetch(AUTOTUNE)
)
val_ds = (
    val_ds_raw
    .map(normalise,  num_parallel_calls=AUTOTUNE)
    .map(to_onehot,  num_parallel_calls=AUTOTUNE)
    .batch(BS)
    .prefetch(AUTOTUNE)
)

# Integer-label test set -> used for manual macro-F1 calculation.
test_ds = (
    test_ds_raw
    .map(normalise, num_parallel_calls=AUTOTUNE)
    .batch(BS)
    .prefetch(AUTOTUNE)
)

# One-hot test set -> used for model.evaluate() (loss + accuracy).
test_ds_oh = (
    test_ds_raw
    .map(normalise,  num_parallel_calls=AUTOTUNE)
    .map(to_onehot,  num_parallel_calls=AUTOTUNE)
    .batch(BS)
    .prefetch(AUTOTUNE)
)

# Building blocks
# Shared sub-modules used by the modle

def gelu(x):
    """gelu activation – smoother than gelu, better gradients in deep nets."""
    return tf.nn.gelu(x)


def residual_block(x, channels: int):
    """
    Standard pre-activation residual block.
      Conv → BN → gelu → Conv → BN → Add(skip) → gelu
    No channel projection needed because in_channels == out_channels.
    """
    residual = x
    x = layers.Conv2D(channels, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(gelu)(x)
    x = layers.Conv2D(channels, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.Activation(gelu)(x)
    return x


def dense_res_block(x, in_channels: int, out_channels: int):
    """
    DenseNet-inspired residual block.
    Runs three sequential residual blocks and concatenates their outputs
    (dense connection), then projects back to out_channels via 1×1 conv.
    A strided depthwise conv at the end halves the spatial resolution
    (acts as a learned downsampling, replacing MaxPool).
    If in_channels ≠ out_channels, a learned projection aligns dimensions
    before the first residual block.
    """
    # Optional projection 
    if in_channels != out_channels:
        skip = layers.Conv2D(out_channels, 1, use_bias=False)(x)
        skip = layers.BatchNormalization()(skip)
        x_in = layers.Activation(gelu)(skip)
    else:
        x_in = x

    # Three chained residual blocks (dense connections
    r1  = residual_block(x_in, out_channels)
    r2  = residual_block(r1,   out_channels)
    # r3  = residual_block(r2,   out_channels)
    cat = layers.Concatenate()([r1, r2])       #r3 # dense concat

    # Bottleneck projection back to out_channels
    out = layers.Conv2D(out_channels, 1, use_bias=False)(cat)
    out = layers.BatchNormalization()(out)
    out = layers.Activation(gelu)(out)

    # Spatial downsampling via stride-2 depthwise conv
    out = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(out)
    out = layers.Conv2D(out_channels, 1, use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Activation(gelu)(out)
    return out


def channel_attention(x, channels: int, reduction: int = 8):
    """
    Squeeze-and-Excitation (SE) channel attention.
    Computes per-channel importance weights via global average pooling
    followed by a two-layer MLP, then re-scales each channel accordingly.
    reduction controls the bottleneck size in the MLP.
    """
    gap  = layers.GlobalAveragePooling2D(keepdims=True)(x)
    gap  = layers.Reshape((channels,))(gap)
    attn = layers.Dense(channels // reduction, activation="gelu")(gap)
    attn = layers.Dense(channels, activation="sigmoid")(attn)
    attn = layers.Reshape((1, 1, channels))(attn)
    return layers.Multiply()([x, attn])


def linear_capsule_routing(x, num_classes: int, capsule_dim: int = 16):
    """
    Lightweight capsule-like routing module.
    Projects the feature vector into a (num_classes × capsule_dim) tensor,
    then uses the original feature as a per-class filter and sums to produce
    class-discriminative logit-like scores.  No dynamic routing – O(n) cost.
    """
    h = layers.Dense(256, activation=gelu)(x)
    h = layers.Dense(num_classes * capsule_dim)(h)
    h = layers.Reshape((num_classes, capsule_dim))(h)

    # Broadcast original features across classes and slice to capsule_dim
    x_exp    = layers.RepeatVector(num_classes)(x)
    x_sliced = layers.Lambda(lambda t: t[:, :, :capsule_dim])(x_exp)

    # Element-wise filter + sum-pool over the capsule dimension
    caps = layers.Multiply()([x_sliced, h])
    caps = layers.Lambda(lambda t: tf.reduce_sum(t, axis=-1))(caps)
    caps = layers.BatchNormalization()(caps)
    return caps

# Model definitions
def build_our_model_net(num_classes: int = 46, image_size: int = 32,
    # drop_path_rate: float = 0.05,
    # dropout_rate: float = 0.3,
    weight_decay: float = 1e-4,
    head_units: int = 256,
    override_tier: int = None,) -> Model:
    """
    Stem (dual-path):
      • Standard 3×3 conv path
      • Horizontal stroke scaffold (1×5 conv)
      → Concatenated and refined with channel attention

    Encoder (3 stages, each halving spatial dims):
      enc1: 64→64    (32×32)
      enc2: 64→128   (16×16)
      enc3: 128→256  ( 8× 8)
      Each encoder stage adds a weighted scaffold residual for continuity.

    Decoder head:
      • Linear Capsule Routing (lcr) produces class-discriminative scores
      • Dense head (classification head)
      • Gated fusion: learnable soft gate blends STM and dense head streams
      • Final MLP + layer norm → logits
    """
    K = num_classes

    inp = Input(shape=(image_size, image_size, 1), name="input")

    # Stem
    # Texture path: standard 3×3 convolution
    t        = layers.Conv2D(32, 3, padding="same", use_bias=False)(inp)
    t        = layers.BatchNormalization()(t)
    t        = layers.Activation(gelu)(t)

    # Stroke scaffold: horizontal asymmetric convolution (captures Devanagari
    # top-bar "shirorekha" and horizontal stroke components)
    s        = layers.Conv2D(32, (1, 5), padding="same", use_bias=False)(inp)
    s        = layers.BatchNormalization()(s)
    scaffold  = layers.Activation(gelu)(s)

    stem = layers.Concatenate()([t, scaffold])
    stem = channel_attention(stem, 64)          # SE re-weighting
    stem = layers.Conv2D(64, 1, use_bias=False)(stem)
    stem = layers.BatchNormalization()(stem)
    stem = layers.Activation(gelu)(stem)

    # Encoder
    # Each encoder stage is a dense residual block + scaffold injection.
    # Scaffold is pooled to match the encoder's spatial resolution and
    # added with a small learnable weight (0.1) to preserve stroke structure.

    enc1 = dense_res_block(stem, 64, 64)
    sc1  = layers.AveragePooling2D(2)(layers.Conv2D(64, 1, use_bias=False)(scaffold))
    enc1 = layers.Add()([enc1, layers.Lambda(lambda t: t * 0.1)(sc1)])

    enc2 = dense_res_block(enc1, 64, 128)
    sc2  = layers.AveragePooling2D(4)(layers.Conv2D(128, 1, use_bias=False)(scaffold))
    enc2 = layers.Add()([enc2, layers.Lambda(lambda t: t * 0.1)(sc2)])

    enc3 = dense_res_block(enc2, 128, 256)
    sc3  = layers.AveragePooling2D(8)(layers.Conv2D(256, 1, use_bias=False)(scaffold))
    enc3 = layers.Add()([enc3, layers.Lambda(lambda t: t * 0.1)(sc3)])

    # Multi-scale GAP fusion
    gap1 = layers.GlobalAveragePooling2D(name="gap1")(enc1)
    gap2 = layers.GlobalAveragePooling2D(name="gap2")(enc2)
    gap3 = layers.GlobalAveragePooling2D(name="gap3")(enc3)
    fused_gap = layers.Concatenate(name="multiscale_fused")([gap1, gap2, gap3])

    # Linear Capsule Routing (lcr)
    # Projects the fused multi-scale vector into capsule space.
    # Each of the K capsules learns to respond to one character class.
    lcr_out = linear_capsule_routing(fused_gap, num_classes)   # (B, K)

    # Classification head
    # Dense projection of the raw GAP features (residual path alongside lcr)
    x = layers.Dense(head_units, use_bias=False, name="head_dense")(fused_gap)
    x = layers.LayerNormalization(name="head_ln")(x)
    x = layers.Activation("gelu", name="head_act")(x)
    x = layers.Dense(num_classes, name="head_logits")(x)

    # Gated fusion: lcr scores + dense-head logits
    # A learned scalar gate (per-sample softmax over 2 weights) blends the lcr capsule scores with the plain dense logits.  This lets the model learn how much to trust the capsule routing vs. the direct projection.
    combined = layers.Concatenate(name="gate_input")([x, lcr_out])
    gate     = layers.Dense(2, activation="softmax", name="gate")(combined)  # (B, 2)

    # gate[:,0] weights the dense head; gate[:,1] weights the lcr output
    x_scaled   = layers.Lambda(
        lambda t: t[0] * t[1][:, 0:1], name="gate_dense")([x,gate])
    lcr_scaled = layers.Lambda(
        lambda t: t[0] * t[1][:, 1:2], name="gate_lcr"  )([lcr_out,gate])

    outputs = layers.Add(name="logits")([x_scaled, lcr_scaled])

    model = keras.Model(inputs=inp, outputs=outputs, name="our_model-Net")
    return model

model.summary()
# Registry: model name → builder function (called lazily inside the training loop)
MODELS_TF = {
    "our_model-Net":    lambda: build_our_model_net(NUM_CLASSES, IMG)
}

# LR schedule
class CosineAnnealing(keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine-annealing schedule without restarts.
    LR decays from `base` to a floor of 1e-6 following a half-cosine curve
    over `steps` optimizer steps.
    Formula:  lr(t) = max(base * 0.5 * (1 + cos(π·t/T)), 1e-6)
    """

    def __init__(self, base: float, steps: int):
        self.base  = base
        self.steps = tf.cast(steps, tf.float32)

    def __call__(self, step):
        step   = tf.cast(step, tf.float32)
        cosine = 0.5 * (1.0 + tf.cos(np.pi * step / self.steps))
        return tf.maximum(self.base * cosine, 1e-6)

    def get_config(self):
        return {"base": self.base, "steps": int(self.steps)}

# Training and evaluation helpers
def compile_model(model: Model, steps_total: int) -> Model:
    """
    Attach optimiser, loss, and metrics to a model.
    Uses AdamW (L2-regularised Adam) with a cosine-annealing LR schedule.
    CategoricalCrossentropy (with from_logits=True) is paired with label
    smoothing to improve calibration and reduce overfitting.
    """
    lr_sch = CosineAnnealing(CFG["lr"], steps_total)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_sch,
            weight_decay=CFG["weight_decay"],
        ),
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=CFG["label_smoothing"],
        ),
        metrics=["accuracy"],
        jit_compile=True
    )
    return model


def compute_macro_f1(model: Model, dataset) -> float:
    """
    Compute macro-averaged F1 score over all NUM_CLASSES classes.
    dataset must yield (images, integer_labels) batches.
    Returns F1 as a percentage (0–100).
    """
    tp = np.zeros(NUM_CLASSES)
    fp = np.zeros(NUM_CLASSES)
    fn = np.zeros(NUM_CLASSES)

    for images, labels in dataset:
        preds = tf.argmax(model(images, training=False), axis=1).numpy()
        lbls  = labels.numpy()
        for c in range(NUM_CLASSES):
            tp[c] += np.sum((preds == c) & (lbls == c))
            fp[c] += np.sum((preds == c) & (lbls != c))
            fn[c] += np.sum((preds != c) & (lbls == c))

    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return float(f1.mean() * 100.0)

#  Train + evaluate all models
trained_models  = {}
all_histories   = {}
steps_per_epoch = sum(1 for _ in train_ds)   # number of batches per epoch
total_steps     = steps_per_epoch * CFG["epochs"]

print(_c(f"\n{'═'*70}", "cyan"))
print(_c(f"  Starting benchmark: {len(MODELS_TF)} models × {CFG['epochs']} epochs", "bold", "white"))
print(_c(f"{'═'*70}\n", "cyan"))

for name, model_fn in MODELS_TF.items():
    # Build a fresh model for each experiment
    model = model_fn()
    model = compile_model(model, total_steps)

    # Print the compact parameter table before training starts
    print_model_summary(model)

    # Checkpoint saves the best val_accuracy weights during training
    ckpt_path = os.path.join(CFG["results_dir"], f"{name}_best.keras")
    callbacks = [
        ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=0,
        ),
        # ReduceLROnPlateau(
        #     monitor="val_loss", factor=0.5,
        #     patience=5, min_lr=1e-6, verbose=0,
        # ),
        EarlyStopping(
            monitor="val_accuracy", patience=15,
            restore_best_weights=True, verbose=0,
        )
    ]

model.history()

# Final test-set evaluation
results = {}
for name, model in trained_models.items():
    test_loss, test_acc_raw = model.evaluate(test_ds_oh, verbose=0)
    test_acc = test_acc_raw * 100.0
    macro_f1 = compute_macro_f1(model, test_ds)   # integer-label dataset
    results[name] = {
        "test_acc":  round(test_acc, 2),
        "macro_f1":  round(macro_f1, 2),
        "params":    model.count_params(),
        "test_loss": round(float(test_loss), 4),
    }