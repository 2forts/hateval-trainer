from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dropout, Dense, Lambda
from tensorflow.keras.models import Model

def build_gru_model(
    vectorizer: tf.keras.layers.Layer,
    vocab_size: int,
    embedding_dim: int,
    num_classes: int,
) -> tf.keras.Model:
    """
    Bidirectional stacked GRU baseline model.

    Pipeline:
      string -> TextVectorization -> Embedding -> BiGRU(128, seq) -> Dropout(0.3)
              -> BiGRU(64) -> Dropout(0.3) -> Dense(64, ReLU) -> Dense(logits)

    Notes
    -----
    - The input layer expects a scalar string per sample; the provided
      `vectorizer` maps it to an integer sequence.
    - Final Dense layer outputs raw logits (no activation); we compile
      with SparseCategoricalCrossentropy(from_logits=True).
    """
    # Text input as raw string (one string per example)
    inp = Input(shape=(1,), dtype="string", name="text")

    # Tokenize to integer ids with the provided TextVectorization
    x = vectorizer(inp)  # shape: (B, T)

    # Learnable word embeddings (mask_zero=True to respect padding)
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="embedding")(x)

    # Two stacked BiGRU blocks with dropout in between
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(64))(x)
    x = Dropout(0.3)(x)

    # Small dense head before logits
    x = Dense(64, activation="relu")(x)
    out = Dense(num_classes, name="logits")(x)  # raw logits

    model = Model(inp, out, name="gru_stacked")
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def build_hybrid_model(base_model: tf.keras.Model, num_classes: int) -> tf.keras.Model:
    """
    Build the hybrid quantum-classical head on top of the trained backbone.

    Protocol
    --------
    - Freeze the classical extractor up to the penultimate feature layer
      (i.e., everything except the final logits).
    - Project features to `n_qubits` via Dense, squash into [-pi, pi],
      and feed them to a PennyLane Variational Quantum Circuit (VQC):
        AngleEmbedding -> StronglyEntanglingLayers (L layers)
    - Continue with a small dense head and output logits.

    Requirements
    ------------
    - PennyLane and its Keras bridge (tested around PL 0.35).
    - A CPU backend device is used here (`default.qubit`).

    Notes
    -----
    - The base model is assumed to be the GRU backbone defined above.
    - We compile with a smaller learning rate for the hybrid head,
      which is typical for VQC training stability.
    """
    import pennylane as qml
    from pennylane.qnn import KerasLayer  # Keras-compatible wrapper around a QNode

    # Build a feature extractor that outputs the penultimate dense layer (before logits)
    extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    # Freeze all extractor layers so only the hybrid head trains
    for l in extractor.layers:
        l.trainable = False

    # Quantum circuit configuration
    n_qubits = 5
    n_layers = 3
    dev = qml.device("default.qubit", wires=n_qubits)

    # Define the variational circuit: AngleEmbedding + StronglyEntanglingLayers
    def circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        # Return expectation values (one per wire) → R^n_qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Weight shapes for the variational template
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    qnode = qml.QNode(circuit, dev, interface="tf")
    vqc = KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

    # Wire the hybrid head on top of the frozen extractor
    inp = extractor.input
    x = extractor.output                       # penultimate features from the classical model
    x = Dense(n_qubits)(x)                     # project to match qubit count
    # map to [-pi, pi] to serve as angles for AngleEmbedding (avoid float64 constants)
    x = Lambda(lambda t: tf.math.tanh(t) * tf.constant(np.pi, dtype=tf.float32))(x)
    x = vqc(x)                                 # quantum layer outputs n_qubits expectations
    x = Dense(32, activation="relu")(x)
    out = Dense(num_classes, name="logits")(x) # final logits

    model_h = tf.keras.Model(inputs=inp, outputs=out, name="hybrid_vqc")
    model_h.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # conservative LR for the hybrid head
        metrics=["accuracy"],
    )
    return model_h

def build_dense_ablation_model(base_model: tf.keras.Model, num_classes: int) -> tf.keras.Model:
    """
    Classical ablation head that mirrors the hybrid architecture but
    replaces the VQC with a small dense layer of comparable parameter
    budget (~75 trainable parameters in the classifier head).

    Protocol
    --------
    - Freeze the classical extractor up to the penultimate feature layer
      (same as in build_hybrid_model).
    - Project the 64-dimensional feature vector to 5 dimensions (to match
      the qubit count used in the hybrid model).
    - Apply a small dense layer (10 units) on top of this 5D vector,
      with tanh activation, so that the number of parameters in this
      classical block is of the same order as the VQC.
    - Continue with the same 32-unit dense layer and final logits layer
      as in the hybrid model.
    """
    # Reuse the same feature extractor used in the hybrid model
    extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    for l in extractor.layers:
        l.trainable = False

    n_qubits = 5  # keep the same dimensionality as the quantum head

    inp = extractor.input
    x = extractor.output                      # shape: (batch, 64)

    # Projection to 5 dimensions (same as in the hybrid model)
    x = Dense(n_qubits, name="proj_to_5")(x)
    x = Lambda(lambda t: tf.math.tanh(t), name="proj_tanh")(x)

    # Small dense block with similar parameter budget to the VQC
    # 5 -> 10 units: 5*10 + 10 = 60 params (same order of magnitude as the VQC weights)
    x = Dense(10, activation="tanh", name="dense_ablation")(x)

    # Same 32-unit dense layer and logits as in the hybrid model
    x = Dense(32, activation="relu", name="dense_head_32")(x)
    out = Dense(num_classes, name="logits")(x)

    model_dense = tf.keras.Model(inputs=inp, outputs=out, name="dense_ablation_head")
    model_dense.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )
    return model_dense