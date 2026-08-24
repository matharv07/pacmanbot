import time
import random
import numpy as np

# Simulate safe_area
safe_area = [(random.random() * 40, random.random() * 30) for _ in range(20000)]

start = time.time()
random.shuffle(safe_area)
grid_size = 1.1
spatial_hash = {}
pellets = []
for p in safe_area:
    gx, gy = int(p[0]/grid_size), int(p[1]/grid_size)
    conflict = False
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if (gx+dx, gy+dy) in spatial_hash:
                for pp in spatial_hash[(gx+dx, gy+dy)]:
                    if (p[0]-pp[0])**2 + (p[1]-pp[1])**2 < 1.21:
                        conflict = True
                        break
            if conflict: break
        if conflict: break
    if not conflict:
        pellets.append(p)
        if (gx, gy) not in spatial_hash:
            spatial_hash[(gx, gy)] = []
        spatial_hash[(gx, gy)].append(p)
end = time.time()
print(f"Time: {end - start:.4f}s")
