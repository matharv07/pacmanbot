import time
import random

class Capsule:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

capsules = [Capsule(random.random(), random.random(), random.random(), random.random()) for _ in range(500)]

start = time.time()
for _ in range(200):
    bx, by = random.random(), random.random()
    best_dist = float('inf')
    for c in capsules:
        dx, dy = c.x2 - c.x1, c.y2 - c.y1
        l2 = dx*dx + dy*dy
        if l2 == 0:
            d = (bx - c.x1)**2 + (by - c.y1)**2
        else:
            t = max(0, min(1, ((bx - c.x1)*dx + (by - c.y1)*dy) / l2))
            d = (bx - (c.x1 + t * dx))**2 + (by - (c.y1 + t * dy))**2
        if d < best_dist:
            best_dist = d
end = time.time()
print(f"Time: {end - start:.4f}s")
