
from features import extract_features

sample = {
  "keystrokes": [
    {"t": 0, "type": "down", "key": "a"},
    {"t": 80, "type": "up", "key": "a"},
    {"t": 140, "type": "down", "key": "b"},
    {"t": 230, "type": "up", "key": "b"},
    {"t": 260, "type": "down", "key": "Backspace"},
    {"t": 320, "type": "up", "key": "Backspace"}
  ]
}

print(extract_features(sample))
