import onnxruntime as ort
import numpy as np
import time

# === CONFIG ===
MODEL_PATH = "ONNX_models/convnextv2.onnx"
IMG_SIZE = 384         # operchè il modello è stato addestrato su immagini di questa dimensione
NUM_RUNS = 100         # numero di inferenze da fare per il benchmark
BATCH_SIZE = 1

# === Prepara input dummy ===
input_shape = (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
dummy_input = np.random.rand(*input_shape).astype(np.float32)

# === Carica modello ONNX ===
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Trova nome input
input_name = session.get_inputs()[0].name

# === Benchmark ===
times = []

# Warm-up, 10 inferenze non misurate per riscaldare il modello
for _ in range(10):
    session.run(None, {input_name: dummy_input})

# Misurazione reale
for _ in range(NUM_RUNS):
    start = time.perf_counter()
    session.run(None, {input_name: dummy_input})
    end = time.perf_counter()
    times.append((end - start) * 1000)  # in millisecondi

# === Risultati ===
avg_latency = np.mean(times)
std_latency = np.std(times)
fps = 1000 / avg_latency

print("\n===== ONNX Benchmark =====")
print(f"Model: {MODEL_PATH}")
print(f"Input shape: {input_shape}")
print(f"Average Latency: {avg_latency:.2f} ms")
print(f"Std Dev Latency: {std_latency:.2f} ms")
print(f"FPS: {fps:.2f}")
