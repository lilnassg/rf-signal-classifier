import pickle
import numpy as np
import torch
import torch.nn as nn

# ── model definition (must match the trained architecture) ──────────────────
class SignalCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(SignalCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x

# ── modulation labels ────────────────────────────────────────────────────────
MODULATIONS = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 
               'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

# ── load trained model ───────────────────────────────────────────────────────
model = SignalCNN(num_classes=11)
model.load_state_dict(torch.load('models/signal_cnn.pth'))
model.eval()  # disable dropout for inference

# ── load one real signal from the dataset ───────────────────────────────────
with open('data/RML2016.10a_dict.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# grab a single signal — shape (2, 128)
true_mod = 'QPSK'
snr = 18
signal = data[(true_mod, snr)][0]

# ── preprocess: add batch dimension → shape (1, 2, 128) ─────────────────────
x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

# ── run inference ────────────────────────────────────────────────────────────
with torch.no_grad():
    output = model(x)                          # raw scores for each class
    probabilities = torch.softmax(output, dim=1)  # convert to probabilities
    predicted_idx = torch.argmax(output, dim=1).item()  # pick highest

predicted_mod = MODULATIONS[predicted_idx]
confidence = probabilities[0][predicted_idx].item() * 100

print(f"True modulation    : {true_mod}")
print(f"Predicted          : {predicted_mod}")
print(f"Confidence         : {confidence:.1f}%")
print(f"\nAll class probabilities:")
for mod, prob in zip(MODULATIONS, probabilities[0]):
    print(f"  {mod:<10} {prob.item()*100:.1f}%")