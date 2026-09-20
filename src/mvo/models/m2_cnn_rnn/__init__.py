"""M2 — CNN (ResNet18) + GRU/LSTM

OWNER: Abhishek Laxman Jadhav
BRANCH: m2/backbone → m2/lstm-head → m2/training
"""

from .model import M2CnnRnn  # noqa: F401 — triggers @register("m2")
