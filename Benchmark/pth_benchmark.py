import torch
import numpy as np
import time
from timm.models.convnext import ConvNeXt
import torch.serialization as serialization

# === CONFIG ===
MODEL_PATH = "./convnextv2_atto.fcmae_ft_in1k999.pth"  # percorso al modello PyTorch
IMG_SIZE = 384
NUM_RUNS = 100
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dummy input ===
input_shape = (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
dummy_input = torch.rand(input_shape).to(DEVICE)

# === Abilita caricamento sicuro ===
serialization.add_safe_globals([ConvNeXt])

# === Caricamento modello completo salvato con torch.save(model) ===
# === Lo mette in modalità inferenza con eval() e lo sposta sul dispositivo ===
model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.eval().to(DEVICE)

# === Benchmark ===
times = []

# Warm-up, no_grad per 10 inferenze non misurate per riscaldare il modello
for _ in range(10):
    with torch.no_grad():
        _ = model(dummy_input)

# Misurazione reale
for _ in range(NUM_RUNS):
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_input)
    end = time.perf_counter()
    times.append((end - start) * 1000)  # in ms

# === Risultati ===
avg_latency = np.mean(times)
std_latency = np.std(times)
fps = 1000 / avg_latency

print("\n===== PyTorch .pth Benchmark =====")
print(f"Model: {MODEL_PATH}")
print(f"Device: {DEVICE}")
print(f"Input shape: {input_shape}")
print(f"Average Latency: {avg_latency:.2f} ms")
print(f"Std Dev Latency: {std_latency:.2f} ms")
print(f"FPS: {fps:.2f}")
