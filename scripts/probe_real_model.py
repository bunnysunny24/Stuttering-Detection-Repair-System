import argparse
import importlib.util, sys, os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', choices=['cpu','cuda'])
parser.add_argument('--min', type=int, default=1)
parser.add_argument('--max', type=int, default=256)
parser.add_argument('--channels', type=int, default=123)
parser.add_argument('--time', type=int, default=273)
parser.add_argument('--dtype', default='float32', choices=['float32','float16'])
args = parser.parse_args()

def load_model():
    # import Models.model_cnn_bilstm
    sys.path.insert(0, os.path.abspath('Models'))
    try:
        mod = importlib.import_module('model_cnn_bilstm')
        model = mod.CNNBiLSTM(in_channels=args.channels)
        return model
    except Exception as e:
        print('Failed to import model_cnn_bilstm:', e)
        raise


def fits(batch, device):
    try:
        torch.cuda.empty_cache() if device.type=='cuda' else None
        model = load_model().to(device)
        model.train()
        dtype = torch.float16 if args.dtype=='float16' and device.type=='cuda' else torch.float32
        x = torch.randn(batch, args.channels, args.time, device=device, dtype=dtype, requires_grad=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        out = model(x)
        loss = out.mean()
        loss.backward()
        opt.step()
        # cleanup
        del x, out, loss, opt, model
        torch.cuda.empty_cache() if device.type=='cuda' else None
        return True
    except RuntimeError as e:
        msg = str(e).lower()
        print('RuntimeError:', e)
        if 'out of memory' in msg or 'memory' in msg:
            return False
        raise


def probe(device):
    lo = args.min
    hi = args.min
    while hi <= args.max and fits(hi, device):
        lo = hi
        hi *= 2
        print('tested ok up to', lo, ' -> trying', hi)
    if hi > args.max:
        print('Reached probe max', hi, ', try increasing --max')
    left = lo
    right = min(hi, args.max)
    while left + 1 < right:
        mid = (left + right)//2
        print('trying', mid)
        if fits(mid, device):
            left = mid
        else:
            right = mid
    print('Estimated max safe batch size (model probe):', left)

if __name__=='__main__':
    dev = torch.device('cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu')
    print('Probe device:', dev)
    probe(dev)
