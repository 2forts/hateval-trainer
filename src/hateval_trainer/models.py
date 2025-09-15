from __future__ import annotations
import numpy as np

# Keep TF imports local to speed up CLI startup
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dropout, Dense, Lambda
from tensorflow.keras.models import Model

def build_gru_model(vectorizer: tf.keras.layers.Layer, vocab_size: int, embedding_dim: int, num_classes: int) -> tf.keras.Model:
  """Bidirectional stacked GRU classifier.
  
  Input: raw strings (the vectorizer is embedded in the graph), outputs logits.
  """
  inp = Input(shape=(1,), dtype="string", name="text")
  x = vectorizer(inp)
  x = Embedding(vocab_size, embedding_dim, mask_zero=True, name="embedding")(x)
  x = Bidirectional(GRU(128, return_sequences=True))(x)
  x = Dropout(0.3)(x)
  x = Bidirectional(GRU(64))(x)
  x = Dropout(0.3)(x)
  x = Dense(64, activation="relu")(x)
  out = Dense(num_classes, name="logits")(x)
  
  model = Model(inp, out, name="gru_stacked")
  model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"],
  )
  return model

def build_hybrid_model(base_model: tf.keras.Model, num_classes: int) -> tf.keras.Model:
  """Hybrid quantum-classical head using PennyLane's KerasLayer.
  
  The base_model's penultimate dense layer is used as a frozen feature extractor.
  """
  import pennylane as qml
   
  # Freeze extractor (all but final logits layer)
  extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
  for l in extractor.layers:
    l.trainable = False
  
  n_qubits = 5
  n_layers = 3
  dev = qml.device("default.qubit", wires=n_qubits)
  
  def circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
  
  weight_shapes = {"weights": (n_layers, n_qubits, 3)}
  qnode = qml.QNode(circuit, dev, interface="tf")
  vqc = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
  
  
  inp = extractor.input
  x = extractor.output
  x = Dense(n_qubits)(x)
  x = Lambda(lambda t: tf.math.tanh(t) * np.pi)(x) # scale to [-pi, pi]
  x = vqc(x)
  x = Dense(32, activation="relu")(x)
  out = Dense(num_classes, name="logits")(x) 
  
  model_h = tf.keras.Model(inputs=inp, outputs=out, name="hybrid_vqc")
  model_h.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"],
  )
  return model_h
