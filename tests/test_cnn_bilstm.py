import numpy as np
import torch
from pathlib import Path

from Models.model_cnn_bilstm import CNNBiLSTM


def test_cnn_bilstm_forward():
    model = CNNBiLSTM(in_channels=123, n_classes=5)
    model.eval()
    x = torch.randn(2, 123, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)


if __name__ == '__main__':
    test_cnn_bilstm_forward()
    print('test_cnn_bilstm: OK')
