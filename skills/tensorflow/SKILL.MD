---
name: tensorflow
description: Comprehensive deep learning framework for building, training, and deploying neural networks. TensorFlow provides tf.keras high-level API for model construction, tf.data for efficient data pipelines, and tf.function for graph-mode optimization. Use when working with: neural network training and inference, image classification/detection/segmentation, NLP/text processing with embeddings or transformers, time series forecasting, generative models (VAE, GAN), transfer learning with pretrained models, custom training loops with GradientTape, GPU/TPU accelerated computation, or any deep learning task.
version: 2.16.0
license: Apache-2.0
---

# TensorFlow - Deep Learning Framework

TensorFlow is the most widely-used deep learning framework. TF 2.x uses eager execution by default for intuitive debugging, while `@tf.function` enables graph-mode compilation for production performance. `tf.keras` is the high-level API for model construction across all paradigms.

## When to Use

- Building and training neural networks (classification, regression, generation).
- Image processing (CNN architectures, object detection, segmentation).
- Natural Language Processing (text classification, sequence-to-sequence, transformers).
- Time series forecasting with recurrent or convolutional architectures.
- Generative models (VAE, GAN, autoencoders).
- Transfer learning with pretrained models from `tf.keras.applications`.
- Custom training loops with `tf.GradientTape` for full control.
- Deploying models via TensorFlow Serving or TF Lite (mobile/edge).
- Multi-GPU / TPU distributed training.
- Any task requiring automatic differentiation and computational graphs.

## Reference Documentation

**Official docs**: https://www.tensorflow.org/api_docs/python  
**Keras API**: https://keras.io/api/  
**Tutorials**: https://www.tensorflow.org/tutorials  
**GitHub**: https://github.com/tensorflow/tensorflow  
**Search patterns**: `tf.keras.Model`, `tf.data.Dataset`, `tf.function`, `model.fit`, `tf.GradientTape`

## Core Principles

### Eager Execution (Default in TF2)
Operations execute immediately, returning concrete values. Debugging is straightforward — print tensors, use standard Python control flow, inspect intermediate values at any point.

### tf.function — Graph Mode Compilation
Decorating a function with `@tf.function` traces it into a computational graph, enabling 2–10x speedups. Use on training steps and inference in production. Never put Python side-effects (print, append) inside `@tf.function` — use `tf.print` instead.

### Keras: Three Model-Building Paradigms
Sequential (linear stack), Functional API (DAG with multiple I/O), Model Subclassing (full Python control). All share `compile → fit → predict`. Choose based on architecture complexity.

### tf.data for Data Pipelines
`tf.data.Dataset` is the standard for efficient input pipelines. Supports lazy evaluation, parallel preprocessing, and GPU prefetching — critical to prevent CPU bottlenecks during training.

### Automatic Differentiation
`tf.GradientTape` records operations for backpropagation. `model.fit()` handles this internally; custom training loops require explicit tape management.

## Quick Reference

### Installation

```bash
# Standard install (auto-detects GPU if available)
pip install tensorflow

# CPU only (lighter)
pip install tensorflow-cpu
```

### Standard Imports

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
```

### Basic Pattern — Sequential Classification

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# 1. Data
X_train = np.random.randn(1000, 28, 28, 1).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)

# 2. Model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 3. Compile + Train + Predict
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
predictions = model.predict(X_train[:5])
```

### Basic Pattern — Custom Training Loop

```python
import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()

X = tf.constant(np.random.randn(100, 10).astype(np.float32))
y = tf.constant(np.random.randn(100, 1).astype(np.float32))

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_fn(y, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for epoch in range(10):
    loss = train_step(X, y)
    tf.print("Epoch", epoch, "loss=", loss)
```

## Critical Rules

### ✅ DO

- **Use `@tf.function` on training steps** — Graph compilation gives 2–10x speedup over eager mode.
- **Use `tf.data.Dataset` for all data loading** — Preferred over raw numpy arrays for any dataset beyond toy size.
- **Always call `.prefetch(tf.data.AUTOTUNE)`** — Overlaps CPU preprocessing with GPU computation.
- **Specify `input_shape` in the first layer** — Builds weights immediately; otherwise deferred until first call.
- **Pass `validation_data` or `validation_split` to `model.fit()`** — Only way to detect overfitting.
- **Freeze base model with `base_model.trainable = False`** — Standard practice for transfer learning.
- **Cast inputs to `float32`** — GPU optimized for float32; dtype mismatches cause silent errors or crashes.
- **Use `tf.keras.callbacks`** — EarlyStopping + ModelCheckpoint is the minimum viable training setup.
- **Call `model.summary()`** — Always inspect architecture and parameter count before training.
- **Use `from_logits=True` in loss** — Numerically more stable than softmax + crossentropy separately.

### ❌ DON'T

- **Don't put Python print/append inside `@tf.function`** — Executes only during tracing, not per call. Use `tf.print`.
- **Don't use numpy ops inside `@tf.function`** — Use `tf.*` operations exclusively inside traced functions.
- **Don't skip `.cache()` before `.shuffle()`** — Without cache, data is re-read from disk every epoch.
- **Don't call `model.predict()` in a loop per sample** — Batch predictions; or use `model(x)` for single inference.
- **Don't ignore GPU memory errors** — Set `tf.config.set_memory_growth(gpu, True)` at startup.
- **Don't hardcode batch dimensions** — Use `None`; enables flexible batch sizes across train/eval/serve.
- **Don't use softmax activation + categorical_crossentropy together** — Use logits output + `from_logits=True`.
- **Don't forget `training=True/False` in custom loops** — Controls Dropout and BatchNormalization behavior.

## Anti-Patterns (NEVER)

```python
import tensorflow as tf
import numpy as np

# ❌ BAD: Python side-effect inside tf.function (only runs during tracing)
@tf.function
def bad_step(x, y):
    loss = compute_loss(x, y)
    print(f"Loss: {loss}")          # Prints ONCE at trace time, then never again
    return loss

# ✅ GOOD: tf.print executes every call
@tf.function
def good_step(x, y):
    loss = compute_loss(x, y)
    tf.print("Loss:", loss)         # Prints every call
    return loss

# ❌ BAD: Numpy inside tf.function (breaks graph)
@tf.function
def bad_predict(x):
    result = model(x)
    return np.array(result)         # Fails — numpy not available in graph mode

# ✅ GOOD: Stay in TF ops
@tf.function
def good_predict(x):
    return model(x)

# ❌ BAD: No prefetch — CPU blocks GPU
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(100).batch(32)  # GPU starves waiting for CPU

# ✅ GOOD: Prefetch overlaps CPU/GPU work
dataset = (dataset
    .cache()
    .shuffle(1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE))

# ❌ BAD: Per-sample prediction loop (massive overhead)
results = []
for sample in test_data:
    pred = model.predict(sample[np.newaxis])  # Kernel launch overhead per sample
    results.append(pred)

# ✅ GOOD: Single batched call
results = model.predict(test_data)

# ❌ BAD: No memory growth config → OOM on first large allocation
# TF reserves ALL GPU memory by default

# ✅ GOOD: Enable growth at startup (before any TF ops)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.set_memory_growth(gpu, True)

# ❌ BAD: softmax + crossentropy (double application, numerical instability)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='softmax')  # Applies softmax
])
model.compile(loss='categorical_crossentropy')       # Applies log internally — redundant

# ✅ GOOD: Logits output + from_logits (single stable operation)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10)                         # Raw logits, no activation
])
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
```

## Tensors and Variables

### Tensors — Immutable Data

```python
import tensorflow as tf

# Creation
t1 = tf.constant([1.0, 2.0, 3.0])                  # 1D
t2 = tf.constant([[1, 2], [3, 4]])                  # 2D
t3 = tf.zeros((3, 4))                               # Zeros
t4 = tf.ones((2, 3))                                # Ones
t5 = tf.random.normal((100, 50))                    # Normal dist
t6 = tf.random.uniform((10,), minval=0, maxval=1)   # Uniform

# Properties
print(t2.shape)    # TensorShape([2, 2])
print(t2.dtype)    # tf.int32
print(t2.numpy())  # Convert to numpy array

# Arithmetic (vectorized)
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])
c = a + b                      # [5, 7, 9]
d = a * b                      # [4, 10, 18]
e = tf.reduce_sum(a)           # 6.0
f = tf.reduce_mean(a)          # 2.0
g = tf.math.sqrt(a)            # Element-wise sqrt

# Matrix operations
A = tf.constant([[1.0, 2.0], [3.0, 4.0]])
B = tf.constant([[5.0, 6.0], [7.0, 8.0]])
C = tf.matmul(A, B)            # Matrix multiply
D = tf.linalg.inv(A)           # Inverse
det = tf.linalg.det(A)         # Determinant
eigs = tf.linalg.eigh(A)       # Eigenvalues + eigenvectors

# Reshaping
t = tf.constant([1, 2, 3, 4, 5, 6])
reshaped = tf.reshape(t, (2, 3))           # (2, 3)
expanded = tf.expand_dims(t, axis=0)       # (1, 6) — add batch dim
squeezed = tf.squeeze(expanded)            # (6,) — remove dim of size 1

# Slicing
t = tf.constant([10, 20, 30, 40, 50])
print(t[1:3])       # [20, 30]
print(t[-2:])       # [40, 50]
print(t[::2])       # [10, 30, 50]
```

### Variables — Mutable State (Model Weights)

```python
import tensorflow as tf

# Creation
w = tf.Variable(tf.random.normal((3, 2)), name='weights')
b = tf.Variable(tf.zeros((2,)), name='bias')

# Mutation methods
w.assign(tf.random.normal((3, 2)))           # Replace entirely
w.assign_add(tf.ones_like(w) * 0.01)         # w += 0.01
w.assign_sub(tf.ones_like(w) * 0.001)        # w -= 0.001

# In practice, model manages its own variables:
# model.trainable_variables     — weights updated by optimizer
# model.non_trainable_variables — e.g. BatchNorm running stats
# model.variables               — all variables
```

## tf.data.Dataset — Efficient Data Pipelines

```python
import tensorflow as tf
import numpy as np

# ─── From numpy arrays ───
X = np.random.randn(1000, 28, 28, 1).astype(np.float32)
y = np.random.randint(0, 10, 1000)
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# ─── Core pipeline (canonical order) ───
# load → cache → shuffle → batch → map(augment) → prefetch
train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .cache()                                                    # Cache after first read
    .shuffle(buffer_size=1000)                                  # Shuffle with buffer
    .batch(32)                                                  # Batch after shuffle
    .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)       # Parallel augmentation
    .prefetch(tf.data.AUTOTUNE)                                 # Overlap CPU/GPU
)

# ─── Map functions ───
def normalize(x, y):
    return x / 255.0, y

def augment(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, max_delta=0.1)
    x = tf.image.random_crop(x, size=tf.shape(x))  # Random crop to same size
    return x, y

dataset = dataset.map(normalize).map(augment)

# ─── Filter and slice ───
dataset = dataset.filter(lambda x, y: y != 5)     # Remove class 5
first_100 = dataset.take(100)                      # First 100 elements
after_100 = dataset.skip(100)                      # Everything after 100

# ─── Repeat ───
dataset = dataset.repeat(3)       # 3 epochs
dataset = dataset.repeat()        # Infinite — control stopping in training loop

# ─── Enumerate for inspection ───
for i, (x, y) in dataset.enumerate():
    if i >= 3: break
    print(f"Batch {i}: x={x.shape}, y={y.shape}")

# ─── From file patterns ───
dataset = tf.data.TFRecordDataset(
    tf.data.Dataset.list_files('data/*.tfrecord')
)

# ─── TFRecord: Write ───
writer = tf.io.TFRecordWriter('data.tfrecord')
feature = {
    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
}
example = tf.train.Example(features=tf.train.Features(feature=feature))
writer.write(example.SerializeToString())
writer.close()

# ─── TFRecord: Read + Parse ───
def parse_tfrecord(serialized):
    features = tf.io.parse_single_example(serialized, {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    })
    image = tf.io.decode_jpeg(features['image'], channels=3)
    image = tf.image.resize(image, [224, 224])
    return image, features['label']

dataset = (
    tf.data.TFRecordDataset('data.tfrecord')
    .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .shuffle(10000)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)
```

## Model Building

### Sequential API — Linear Stack

Best for: simple architectures where each layer has one input and one output.

```python
import tensorflow as tf
from tensorflow.keras import layers

# Pass list to constructor
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10)  # Logits — no activation here; use from_logits in loss
])

model.summary()
print(f"Parameters: {model.count_params():,}")
```

### Functional API — DAG of Layers

Best for: multiple inputs/outputs, skip connections, shared layers.

```python
import tensorflow as tf
from tensorflow.keras import layers

# ─── Single input/output ───
inputs = tf.keras.Input(shape=(224, 224, 3), name='image')
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(10, name='logits')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# ─── Multiple inputs ───
img_input = tf.keras.Input(shape=(224, 224, 3), name='image')
meta_input = tf.keras.Input(shape=(16,), name='metadata')

img_feat = layers.Conv2D(32, (3, 3), activation='relu')(img_input)
img_feat = layers.GlobalAveragePooling2D()(img_feat)

merged = layers.Concatenate()([img_feat, meta_input])
x = layers.Dense(64, activation='relu')(merged)
output = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=[img_input, meta_input], outputs=output)

# ─── Skip connection (ResNet block) ───
def residual_block(x, filters):
    residual = x
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])      # Skip connection
    return layers.Activation('relu')(x)
```

### Model Subclassing — Full Control

Best for: dynamic computation, research prototyping, conditional logic in forward pass.

```python
import tensorflow as tf
from tensorflow.keras import layers

class Classifier(tf.keras.Model):
    def __init__(self, num_classes, hidden_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.dense1   = layers.Dense(hidden_dim, activation='relu')
        self.dropout  = layers.Dropout(0.3)
        self.dense2   = layers.Dense(hidden_dim // 2, activation='relu')
        self.head     = layers.Dense(num_classes)  # Logits

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)   # training flag is critical
        x = self.dense2(x)
        return self.head(x)

model = Classifier(num_classes=10)
model.build(input_shape=(None, 784))
model.summary()

# Works with compile/fit just like Sequential
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# ─── Multi-output subclassed model ───
class MultiHead(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.backbone   = layers.Dense(128, activation='relu')
        self.cls_head   = layers.Dense(10, name='classification')
        self.reg_head   = layers.Dense(1, name='regression')

    def call(self, x, training=False):
        shared = self.backbone(x)
        return {
            'classification': self.cls_head(shared),
            'regression':     self.reg_head(shared)
        }
```

## Training

### Built-in Training — model.fit()

```python
import tensorflow as tf

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,          # 20% held out for val
    shuffle=True,
    verbose=1
)

# ─── Loss selection guide ───
# Labels are integers [0, 1, 2, ...]     → SparseCategoricalCrossentropy
# Labels are one-hot [[1,0,0], ...]      → CategoricalCrossentropy
# Binary classification {0, 1}           → BinaryCrossentropy
# Regression                             → MeanSquaredError / MeanAbsoluteError

# ─── Plot training curves ───
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric in zip(axes, ['loss', 'accuracy']):
    ax.plot(history.history[metric], label='Train')
    ax.plot(history.history[f'val_{metric}'], label='Val')
    ax.set_ylabel(metric.capitalize())
    ax.set_xlabel('Epoch')
    ax.legend()
plt.tight_layout()
plt.show()
```

### Custom Training Loop — GradientTape

```python
import tensorflow as tf

optimizer  = tf.keras.optimizers.Adam(1e-3)
loss_fn    = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc  = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
val_acc    = tf.keras.metrics.SparseCategoricalAccuracy(name='val_acc')

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)      # training=True → Dropout active
        loss   = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss.update_state(loss)
    train_acc.update_state(y, logits)

@tf.function
def val_step(x, y):
    logits = model(x, training=False)         # training=False → Dropout off
    val_acc.update_state(y, logits)

# ─── Epoch loop ───
for epoch in range(num_epochs):
    train_loss.reset_state()
    train_acc.reset_state()
    val_acc.reset_state()

    for x_batch, y_batch in train_dataset:
        train_step(x_batch, y_batch)

    for x_batch, y_batch in val_dataset:
        val_step(x_batch, y_batch)

    tf.print(f"Epoch {epoch+1}: loss={train_loss.result():.4f}, "
             f"train_acc={train_acc.result():.4f}, val_acc={val_acc.result():.4f}")
```

### Callbacks — Training Control

```python
import tensorflow as tf

callbacks = [
    # Stop when val_loss plateaus; restore best weights
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1
    ),
    # Save best model to disk
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_accuracy', save_best_only=True,
        mode='max', verbose=1
    ),
    # Reduce LR on plateau
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-6, verbose=1
    ),
    # TensorBoard: tensorboard --logdir=./logs
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
]

model.fit(X_train, y_train, epochs=100,
          validation_split=0.2, callbacks=callbacks)
```

## Common Architectures

### CNN — Image Classification

```python
import tensorflow as tf
from tensorflow.keras import layers

def build_cnn(input_shape, num_classes):
    """Standard CNN with BatchNorm and GlobalAvgPool."""
    model = tf.keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),   # Replaces Flatten+Dense — far fewer params
        layers.Dropout(0.5),

        # Head
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes)          # Logits
    ])
    return model

model = build_cnn((224, 224, 3), num_classes=10)
model.summary()
```

### RNN / LSTM — Sequence Modeling

```python
import tensorflow as tf
from tensorflow.keras import layers

# ─── Text classification ───
def build_lstm_text(vocab_size, embed_dim, max_length, num_classes):
    return tf.keras.Sequential([
        layers.Embedding(vocab_size, embed_dim, input_length=max_length),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(64)),      # return_sequences=False
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes)
    ])

# ─── Time series regression ───
def build_lstm_timeseries(timesteps, num_features, forecast_horizon):
    inputs = tf.keras.Input(shape=(timesteps, num_features))
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(forecast_horizon)(x)
    return tf.keras.Model(inputs, outputs)

model = build_lstm_timeseries(timesteps=30, num_features=5, forecast_horizon=7)
```

### Transformer Block

```python
import tensorflow as tf
from tensorflow.keras import layers

class PositionalEncoding(tf.keras.layers.Layer):
    """Learnable positional embeddings."""
    def __init__(self, max_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = layers.Embedding(max_length, embed_dim)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(seq_len)
        return x + self.pos_emb(positions)

class TransformerBlock(tf.keras.layers.Layer):
    """Single transformer encoder block: self-attention + FFN with residuals."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn    = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn     = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.norm1   = layers.LayerNormalization(epsilon=1e-6)
        self.norm2   = layers.LayerNormalization(epsilon=1e-6)
        self.drop1   = layers.Dropout(dropout)
        self.drop2   = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Self-attention + residual
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        # FFN + residual
        ffn_out  = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x

def build_transformer_classifier(vocab_size, max_length, embed_dim, num_heads, ff_dim, num_classes, num_blocks=2):
    """Text classifier using transformer encoder."""
    inputs = tf.keras.Input(shape=(max_length,))
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    x = PositionalEncoding(max_length, embed_dim)(x)

    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(embed_dim, activation='relu')(x)
    outputs = layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs, outputs)

# Example: 2-block transformer for sentiment analysis
model = build_transformer_classifier(
    vocab_size=30000, max_length=256,
    embed_dim=128, num_heads=8, ff_dim=512,
    num_classes=3, num_blocks=2
)
model.summary()
```

## Transfer Learning

```python
import tensorflow as tf
from tensorflow.keras import layers

# ─── Phase 1: Feature Extraction (fast) ───
base = tf.keras.applications.MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)
base.trainable = False              # Freeze all pretrained weights

inputs  = base.input
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=5)

# ─── Phase 2: Fine-Tuning (unfreeze top layers) ───
for layer in base.layers[-20:]:     # Unfreeze last 20 layers
    layer.trainable = True

# CRITICAL: recompile with much lower LR after unfreezing
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 100x lower than default
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
])

# ─── Available pretrained models (ImageNet) ───
# tf.keras.applications.MobileNetV2      — lightweight, mobile-optimized
# tf.keras.applications.EfficientNetB0   — best accuracy/speed tradeoff
# tf.keras.applications.ResNet50         — classic, strong baseline
# tf.keras.applications.VGG16            — simple, large
# tf.keras.applications.InceptionV3      — multi-scale features
# tf.keras.applications.DenseNet121      — dense connections
```

## Saving and Loading Models

```python
import tensorflow as tf

# ─── Full model save (recommended formats) ───
model.save('my_model')              # SavedModel format (directory) — default
model.save('my_model.keras')        # Keras native (.keras) — TF 2.12+
model.save('my_model.h5')           # HDF5 — legacy, still works

# ─── Full model load ───
loaded = tf.keras.models.load_model('my_model')
loaded = tf.keras.models.load_model('my_model.keras')

# ─── Weights only ───
model.save_weights('weights.tf')          # Save
model.load_weights('weights.tf')          # Load — architecture must match

# ─── Checkpoint (custom training loops) ───
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager   = tf.train.CheckpointManager(checkpoint, './ckpts', max_to_keep=3)

manager.save()                            # Save current state
checkpoint.restore(manager.latest_checkpoint)  # Restore latest

# ─── Export for deployment ───
# TF Serving
tf.saved_model.save(model, 'serving_model')

# TF Lite (mobile/edge)
converter = tf.lite.TFLiteConverter.from_saved_model('my_model')
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Device Management

```python
import tensorflow as tf

# ─── Inspect devices ───
print("GPUs:", tf.config.list_physical_devices('GPU'))
print("CPUs:", tf.config.list_physical_devices('CPU'))

# ─── Memory growth (run BEFORE any TF ops) ───
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.set_memory_growth(gpu, True)

# ─── Limit GPU memory to N GB ───
if gpus:
    tf.config.set_logical_device_configuration(gpus[0], [
        tf.config.LogicalDeviceConfiguration(memory_limit=4096)  # 4 GB
    ])

# ─── Place ops on specific device ───
with tf.device('/GPU:0'):
    result = tf.matmul(a, b)

# ─── Multi-GPU: MirroredStrategy ───
strategy = tf.distribute.MirroredStrategy()  # Auto-detects all GPUs

with strategy.scope():
    model = build_model()                    # Must create model inside scope
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(train_dataset, epochs=10)          # Automatically distributed
print(f"Training on {strategy.num_replicas_in_sync} replicas")
```

## Practical Workflows

### 1. Image Classification Pipeline (End-to-End)

```python
import tensorflow as tf
from tensorflow.keras import layers

def build_image_pipeline(train_dir, val_dir, img_size=(224, 224), batch_size=32, num_classes=10):
    """Full pipeline: load → augment → pretrained model → train → checkpoint."""

    # ─── Load from directory ───
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size, seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size
    )

    # ─── Augmentation (train only) ───
    augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ])

    def preprocess_train(x, y):
        x = x / 255.0
        x = augmentation(x, training=True)
        return x, y

    def preprocess_val(x, y):
        return x / 255.0, y

    train_ds = train_ds.map(preprocess_train).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.map(preprocess_val).prefetch(tf.data.AUTOTUNE)

    # ─── Model: EfficientNet transfer learning ───
    base = tf.keras.applications.EfficientNetB0(
        weights='imagenet', include_top=False, input_shape=(*img_size, 3)
    )
    base.trainable = False

    model = tf.keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # ─── Train with callbacks ───
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=30,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best.keras', save_best_only=True,
                                               monitor='val_accuracy', mode='max')
        ]
    )
    return model, history

# model, history = build_image_pipeline('data/train', 'data/val', num_classes=5)
```

### 2. Text Classification with Embeddings

```python
import tensorflow as tf
import numpy as np

def build_text_classifier(texts, labels, vocab_size=20000, max_length=200, num_classes=2):
    """Tokenize → Embed → BiLSTM → Classify."""

    # ─── Tokenize ───
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding='post', truncating='post'
    )

    # ─── Split ───
    split = int(0.8 * len(padded))
    X_train, X_val = padded[:split], padded[split:]
    y_train, y_val = np.array(labels[:split]), np.array(labels[split:])

    # ─── Model ───
    activation = 'softmax' if num_classes > 2 else 'sigmoid'
    loss       = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes if num_classes > 2 else 1, activation=activation)
    ])

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=20, batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )
    return model, tokenizer, history

# model, tokenizer, history = build_text_classifier(texts, labels, num_classes=3)
```

### 3. Time Series Forecasting (CNN-LSTM Hybrid)

```python
import tensorflow as tf
import numpy as np

def create_sequences(data, window_size, forecast_horizon):
    """Sliding window: input=[t-window..t], target=[t+1..t+horizon]."""
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + forecast_horizon])
    return np.array(X), np.array(y)

def build_timeseries_model(window_size, num_features, forecast_horizon):
    """1D-CNN extracts local patterns; LSTM captures temporal order."""
    inputs = tf.keras.Input(shape=(window_size, num_features))

    # Local pattern extraction
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)

    # Temporal dependencies
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(32)(x)

    # Forecast head
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(forecast_horizon)(x)

    return tf.keras.Model(inputs, outputs)

# ─── Usage ───
# data = load_and_normalize_timeseries()     # shape: (n_steps, n_features)
# X, y = create_sequences(data, window_size=30, forecast_horizon=7)
# # Train/val split
# split = int(0.8 * len(X))
# model = build_timeseries_model(30, data.shape[1], 7)
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]),
#           epochs=50, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
```

### 4. Variational Autoencoder (VAE)

```python
import tensorflow as tf
from tensorflow.keras import layers

class VAE(tf.keras.Model):
    """VAE with reparameterization trick."""

    def __init__(self, latent_dim, input_dim, hidden_dim=256, **kwargs):
        super().__init__(**kwargs)
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            layers.Dense(hidden_dim // 2, activation='relu'),
        ])
        self.mean_net   = layers.Dense(latent_dim)
        self.logvar_net = layers.Dense(latent_dim)
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(hidden_dim // 2, activation='relu', input_shape=(latent_dim,)),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        return self.mean_net(h), self.logvar_net(h)

    def reparameterize(self, mean, log_var):
        """z = mean + std * eps, where eps ~ N(0,1). Gradients flow through mean/std."""
        eps = tf.random.normal(tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps

    def decode(self, z):
        return self.decoder(z)

    def call(self, x, training=False):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon = self.decode(z)
        # Store for loss
        self._mean, self._log_var = mean, log_var
        return recon

    def compute_loss(self, x, x_recon):
        recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_recon))
        kl_loss    = -0.5 * tf.reduce_mean(1 + self._log_var - tf.square(self._mean) - tf.exp(self._log_var))
        return recon_loss + kl_loss

# ─── Custom training ───
vae      = VAE(latent_dim=32, input_dim=784)
optimizer = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        x_recon = vae(x, training=True)
        loss = vae.compute_loss(x, x_recon)
    grads = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(grads, vae.trainable_variables))
    return loss

# ─── Generate samples from latent space ───
z_samples  = tf.random.normal((16, 32))
generated  = vae.decode(z_samples)           # 16 generated images
```

### 5. Multi-Task Learning

```python
import tensorflow as tf
from tensorflow.keras import layers

def build_multitask(input_shape, tasks):
    """
    Shared backbone + task-specific heads.

    tasks: dict — {'name': (num_outputs, activation, loss, weight)}
    Example:
        tasks = {
            'category':  (10, 'softmax', 'sparse_categorical_crossentropy', 1.0),
            'sentiment': (3,  'softmax', 'sparse_categorical_crossentropy', 0.7),
            'toxicity':  (1,  'sigmoid', 'binary_crossentropy',             0.5),
        }
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Shared trunk
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    shared = layers.Dense(128, activation='relu')(x)

    # Task heads
    outputs, losses, weights = {}, {}, {}
    for name, (n_out, act, loss, w) in tasks.items():
        head = layers.Dense(64, activation='relu')(shared)
        outputs[name] = layers.Dense(n_out, activation=act, name=name)(head)
        losses[name]  = loss
        weights[name] = w

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=losses, loss_weights=weights,
                  metrics={name: 'accuracy' for name in tasks})
    return model

# ─── Usage ───
tasks = {
    'category':  (10, 'softmax', 'sparse_categorical_crossentropy', 1.0),
    'sentiment': (3,  'softmax', 'sparse_categorical_crossentropy', 0.7),
    'toxicity':  (1,  'sigmoid', 'binary_crossentropy',             0.5),
}
model = build_multitask(input_shape=(128,), tasks=tasks)

# Train with dict of targets:
# model.fit(X_train, {
#     'category':  y_cat,
#     'sentiment': y_sent,
#     'toxicity':  y_tox
# }, validation_split=0.2, epochs=30)
```

## Performance Tips

### tf.function and XLA

```python
import tensorflow as tf

# Graph compilation — always use on training step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        loss = compute_loss(model(x, training=True), y)
    optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                                  model.trainable_variables))
    return loss

# Prevent retracing by fixing input signatures
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32),
    tf.TensorSpec(shape=[None], dtype=tf.int32)
])
def typed_train_step(x, y):
    pass  # One trace for all batch sizes

# XLA compilation — additional speedup (especially TPU)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', jit_compile=True)
```

### Mixed Precision

```python
import tensorflow as tf

# float16 compute + float32 weights → ~2x throughput on Tensor Core GPUs
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    # Output layer MUST cast back to float32 for loss stability
    tf.keras.layers.Dense(10, activation='softmax', dtype='float32')
])
```

### Dataset Pipeline Order

```python
import tensorflow as tf

# ✅ Optimal order: load → cache → shuffle → batch → augment → prefetch
dataset = (
    tf.data.TFRecordDataset(files)
    .map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)   # Parse raw records
    .cache()                                                # Cache parsed tensors in memory
    .shuffle(buffer_size=10000)                             # Shuffle after cache
    .batch(64)                                              # Batch after shuffle
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)     # Augment per batch (optional)
    .prefetch(tf.data.AUTOTUNE)                             # Overlap with GPU
)

# Key insight: cache() placement determines what gets cached
# Before parse: caches raw bytes (less memory, re-parses each epoch)
# After parse:  caches tensors (more memory, faster epochs) ← usually preferred
```

## Common Pitfalls and Solutions

### Shape Mismatches

```python
# ❌ PROBLEM: Missing batch dimension on single sample
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(784,))])
pred = model(np.zeros(784))          # Error: expected rank 2, got rank 1

# ✅ SOLUTION: Always include batch dimension
pred = model(np.zeros((1, 784)))     # Shape (1, 784) — batch of 1
pred = model.predict(np.zeros((1, 784)))  # predict handles this too
```

### training Flag Forgotten

```python
# ❌ PROBLEM: Dropout and BatchNorm behave identically in train and eval
@tf.function
def bad_eval(x):
    return model(x)                  # Dropout still active!

# ✅ SOLUTION: Explicit training flag
@tf.function
def good_train(x):  return model(x, training=True)   # Dropout ON
@tf.function
def good_eval(x):   return model(x, training=False)  # Dropout OFF
```

### Memory Leaks in Loops

```python
# ❌ PROBLEM: Appending tensors creates growing computation graph
losses = []
for batch in dataset:
    loss = train_step(batch)
    losses.append(loss)             # Keeps all tensors in memory!

# ✅ SOLUTION: Use tf.keras.metrics or extract with .numpy()
loss_metric = tf.keras.metrics.Mean()
for batch in dataset:
    loss = train_step(batch)
    loss_metric.update_state(loss)  # Accumulates efficiently
print(f"Average loss: {loss_metric.result().numpy()}")
```

### Loss / Label Format Mismatch

```python
# Quick reference — pick ONE consistent pair:

# Integer labels [0, 2, 1, 3, ...]
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
# Last layer: layers.Dense(num_classes)  ← no activation

# One-hot labels [[1,0,0], [0,0,1], ...]
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
# Last layer: layers.Dense(num_classes)  ← no activation

# Binary labels {0, 1}
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
# Last layer: layers.Dense(1)            ← no activation

# ⚡ Rule: from_logits=True + no output activation = most stable
```

### Retracing Performance

```python
# ❌ PROBLEM: Different input shapes cause retracing (slow)
@tf.function
def predict(x):
    return model(x)

predict(np.zeros((1, 10)))     # Traces for shape (1, 10)
predict(np.zeros((32, 10)))    # Retraces for shape (32, 10)!
predict(np.zeros((64, 10)))    # Retraces again!

# ✅ SOLUTION: Fix signature with None for batch dim
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 10], dtype=tf.float32)])
def predict(x):
    return model(x)             # Single trace, any batch size
```

---

TensorFlow's power comes from layering: `tf.data` for efficient pipelines, `tf.keras` for clean model definition across three paradigms, `@tf.function` for graph-mode speed, and `tf.GradientTape` for full training control. Master these four building blocks and you can tackle any deep learning task — from quick prototypes with `model.fit()` to production systems with custom loops and deployment.