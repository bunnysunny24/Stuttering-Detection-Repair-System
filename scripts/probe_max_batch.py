# scripts/probe_max_batch.py
import argparse,math,torch,sys
parser=argparse.ArgumentParser()
parser.add_argument('--device',default='cuda',choices=['cuda','cpu'])
parser.add_argument('--min',type=int,default=1)
parser.add_argument('--max',type=int,default=1024)
parser.add_argument('--channels',type=int,default=123)
parser.add_argument('--time',type=int,default=273)
parser.add_argument('--dtype',default='float32',choices=['float32','float16'])
args=parser.parse_args()

dev = torch.device('cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu')
print('Probe device:', dev)

def fits(batch):
    try:
        torch.cuda.empty_cache() if dev.type=='cuda' else None
        x = torch.randn(batch, args.channels, args.time, dtype=getattr(torch,args.dtype), device=dev, requires_grad=True)
        # instantiate a tiny model similar to your net shape to approximate activation cost:
        # A simple conv -> linear to create activations
        m = torch.nn.Sequential(
            torch.nn.Conv1d(args.channels, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 8)
        ).to(dev)
        out = m(x)
        loss = out.mean()
        loss.backward()
        # free
        del x, out, loss, m
        torch.cuda.empty_cache() if dev.type=='cuda' else None
        return True
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() or 'memory' in str(e).lower():
            return False
        raise

# exponential search
lo=args.min
hi=args.min
while hi<=args.max and fits(hi):
    lo=hi
    hi*=2
    print('tested ok up to', lo, ' -> trying', hi)
if hi>args.max:
    print('Reached probe max', hi, ', try increasing --max')
# binary search between lo..hi
left=lo
right=min(hi, args.max)
while left+1<right:
    mid=(left+right)//2
    print('trying', mid)
    if fits(mid):
        left=mid
    else:
        right=mid
print('Estimated max safe batch size:', left)