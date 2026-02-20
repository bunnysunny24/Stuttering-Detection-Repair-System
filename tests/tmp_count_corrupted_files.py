import os
p = 'datasets/corrupted_audio'
files = sorted(os.listdir(p))
print(len(files))
for f in files[:20]:
    print(f)
