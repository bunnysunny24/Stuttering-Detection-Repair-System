from Models.train_90plus_final import AudioAugmentation
import numpy as np
aug = AudioAugmentation(augment_prob=1.0)
spec = np.zeros((123,128), dtype=np.float32)
out = aug(spec)
print('OK, output shape:', out.shape)
