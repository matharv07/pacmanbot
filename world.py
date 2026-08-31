import pygame
import math
import random
import numpy as np
import scipy.ndimage
import argparse
from collections import deque

class Obstacle:
    def contains(self, x, y):
        raise NotImplementedError
    def draw(self, surface, scale, offset_x=0, offset_y=0):
        raise NotImplementedError

class Capsule(Obstacle):
    def __init__(self, x1, y1, x2, y2, r):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.r = r
        self.r2 = r * r
        
    def contains(self, x, y):
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        l2 = dx*dx + dy*dy
        if l2 == 0:
            return (x - self.x1)**2 + (y - self.y1)**2 <= self.r2
        t = max(0, min(1, ((x - self.x1)*dx + (y - self.y1)*dy) / l2))
        px = self.x1 + t * dx
        py = self.y1 + t * dy
        return (x - px)**2 + (y - py)**2 <= self.r2

    def draw(self, surface, scale, offset_x=0, offset_y=0):
        pygame.draw.circle(surface, (40, 40, 200), (int(self.x1*scale + offset_x), int(self.y1*scale + offset_y)), int(self.r*scale))
        pygame.draw.circle(surface, (40, 40, 200), (int(self.x2*scale + offset_x), int(self.y2*scale + offset_y)), int(self.r*scale))
        if self.x1 != self.x2 or self.y1 != self.y2:
            angle = math.atan2(self.y2 - self.y1, self.x2 - self.x1)
            dx = math.sin(angle) * self.r * scale
            dy = -math.cos(angle) * self.r * scale
            p1 = (self.x1*scale + offset_x + dx, self.y1*scale + offset_y + dy)
            p2 = (self.x1*scale + offset_x - dx, self.y1*scale + offset_y - dy)
            p3 = (self.x2*scale + offset_x - dx, self.y2*scale + offset_y - dy)
            p4 = (self.x2*scale + offset_x + dx, self.y2*scale + offset_y + dy)
            pygame.draw.polygon(surface, (40, 40, 200), [p1, p2, p3, p4])

class CurvedWall(Obstacle):
    def __init__(self, points, r):
        self.capsules = []
        for i in range(len(points)-1):
            if (i + 1) % 3 != 0:
                self.capsules.append(Capsule(points[i][0], points[i][1], points[i+1][0], points[i+1][1], r))
        
    def contains(self, x, y):
        return any(c.contains(x, y) for c in self.capsules)
        
    def draw(self, surface, scale, offset_x=0, offset_y=0):
        for c in self.capsules:
            c.draw(surface, scale, offset_x, offset_y)

class World:
    def __init__(self, width, height, resolution=0.5):
        self.width = width
        self.height = height
        self.obstacles = []
        self.compiled_segments = np.empty((0, 5), dtype=np.float32) #N*[x1, y1, x2, y2, r]
        self.pellets = []
        self.power_pellets = []
        self.resolution = resolution
        self.safe_area = []
        self.prm_nodes_arr = np.empty((0, 2), dtype=np.float32)
        self.pellets_arr = np.empty((0, 2), dtype=np.float32)
        self.power_pellets_arr = np.empty((0, 2), dtype=np.float32)

    def _update_pellet_arrays(self):
        self.pellets_arr = np.array(self.pellets, dtype=np.float32) if self.pellets else np.empty((0, 2), dtype=np.float32)
        self.power_pellets_arr = np.array(self.power_pellets, dtype=np.float32) if self.power_pellets else np.empty((0, 2), dtype=np.float32)
        self.pellet_set = set(self.pellets)
        self.power_pellet_set = set(self.power_pellets)

    def _compile_obstacles(self):
        segs = []
        for obs in self.obstacles:
            if isinstance(obs, Capsule):
                segs.append([obs.x1, obs.y1, obs.x2, obs.y2, obs.r])
            elif isinstance(obs, CurvedWall):
                for c in obs.capsules:
                    segs.append([c.x1, c.y1, c.x2, c.y2, c.r])
        self.compiled_segments = np.array(segs, dtype=np.float32)
        if len(self.compiled_segments) > 0:
            self._x1 = self.compiled_segments[:, 0][np.newaxis, :]
            self._y1 = self.compiled_segments[:, 1][np.newaxis, :]
            self._x2 = self.compiled_segments[:, 2][np.newaxis, :]
            self._y2 = self.compiled_segments[:, 3][np.newaxis, :]
            self._r  = self.compiled_segments[:, 4][np.newaxis, :]
            self._dx = self._x2 - self._x1
            self._dy = self._y2 - self._y1
            l2 = self._dx**2 + self._dy**2
            self._mask = (l2 == 0)
            self._l2_safe = np.where(self._mask, 1.0, l2)

    def _points_to_segments_dist_sq(self, px, py):
        if len(self.compiled_segments) == 0:
            return np.full((len(px), 0), np.inf), np.empty((1, 0)) 
        px = px[:, np.newaxis] #(M, 1)
        py = py[:, np.newaxis]
        t = ((px - self._x1)*self._dx + (py - self._y1)*self._dy) / self._l2_safe
        t = np.clip(t, 0.0, 1.0)
        t = np.where(self._mask, 0.0, t)        
        proj_x = self._x1 + t * self._dx
        proj_y = self._y1 + t * self._dy
        dist_sq = (px - proj_x)**2 + (py - proj_y)**2 #(M, N_filtered)
        return dist_sq, self._r

    def is_passable(self, x, y, radius=0):
        if x - radius < 0 or x + radius > self.width or y - radius < 0 or y + radius > self.height:
            return False
        if len(self.compiled_segments) == 0:
            return True
        px = np.array([x], dtype=np.float32)
        py = np.array([y], dtype=np.float32)
        dist_sq, r_seg = self._points_to_segments_dist_sq(px, py)
        dist_sq = dist_sq[0]
        r_seg = r_seg[0]
        if np.any(dist_sq <= (r_seg + radius)**2):
            return False
        return True

    def _compile_passable_grids(self):
        import cv2
        res = 0.1
        cols = int(math.ceil(self.width / res))
        rows = int(math.ceil(self.height / res))
        
        def rasterize_collided(radius):
            grid = np.zeros((rows, cols), dtype=np.uint8)
            for obs in self.obstacles:
                capsules = obs.capsules if isinstance(obs, CurvedWall) else [obs]
                for c in capsules:
                    t_res = (c.r + radius) / res
                    thick = max(1, int(round(2 * t_res)))
                    pt1 = (int(c.x1 / res), int(c.y1 / res))
                    pt2 = (int(c.x2 / res), int(c.y2 / res))
                    cv2.line(grid, pt1, pt2, 1, thick)
                    r_px = int(round(t_res))
                    if r_px > 0:
                        cv2.circle(grid, pt1, r_px, 1, -1)
                        cv2.circle(grid, pt2, r_px, 1, -1)
            return grid.astype(bool)
            
        self._grid_0_0 = ~rasterize_collided(0.0)
        self._grid_0_3 = ~rasterize_collided(0.3)
        self._grid_0_4 = ~rasterize_collided(0.4)

    def batch_is_passable(self, px, py, radius=0):
        if hasattr(self, '_grid_0_0'):
            res = 0.1
            cols = self._grid_0_0.shape[1]
            rows = self._grid_0_0.shape[0]
            cx = (px / res).astype(np.int32)
            cy = (py / res).astype(np.int32)
            valid = (cx >= 0) & (cx < cols) & (cy >= 0) & (cy < rows)
            if abs(radius - 0.4) < 1e-4:
                grid = self._grid_0_4
            elif abs(radius - 0.3) < 1e-4:
                grid = self._grid_0_3
            elif radius < 1e-4:
                grid = self._grid_0_0
            else:
                grid = None                
            if grid is not None:
                ans = np.zeros_like(px, dtype=bool)
                ans[valid] = grid[cy[valid], cx[valid]]
                return ans
        out_of_bounds = (px - radius < 0) | (px + radius > self.width) | (py - radius < 0) | (py + radius > self.height)
        if len(self.compiled_segments) == 0:
            return ~out_of_bounds
        px_flat = px.ravel()
        py_flat = py.ravel()
        collided = np.zeros(len(px_flat), dtype=bool)
        chunk_size = 4096
        for i in range(0, len(px_flat), chunk_size):
            px_c = px_flat[i:i+chunk_size]
            py_c = py_flat[i:i+chunk_size]
            dist_sq, r_seg = self._points_to_segments_dist_sq(px_c, py_c)
            collided[i:i+chunk_size] = np.any(dist_sq <= (r_seg + radius)**2, axis=1)
        ans_flat = ~(out_of_bounds.ravel() | collided)
        return ans_flat.reshape(px.shape)

    def line_of_sight(self, p1, p2, radius=0, step_size=0.2):         #checks for obstacles along a line segment
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            return self.is_passable(p1[0], p1[1], radius)
        n_steps = max(2, int(math.ceil(dist / step_size)))
        fracs = np.linspace(0, 1.0, n_steps)
        px = p1[0] + dx * fracs
        py = p1[1] + dy * fracs
        return np.all(self.batch_is_passable(px, py, radius))

    def batch_line_of_sight(self, p1, p2s, radius=0, step_size=0.2):
        if len(p2s) == 0:
            return np.array([], dtype=bool)
        p2s = np.array(p2s)
        dx = p2s[:, 0] - p1[0]
        dy = p2s[:, 1] - p1[1]
        dist = np.hypot(dx, dy)
        n_steps_arr = np.maximum(2, np.ceil(dist / step_size).astype(int))
        total_steps = np.sum(n_steps_arr)
        indices = np.repeat(np.arange(len(p2s)), n_steps_arr)
        ones = np.ones(total_steps, dtype=int)
        start_idx = np.cumsum(n_steps_arr) - n_steps_arr
        ones[start_idx] = 1 - np.roll(n_steps_arr, 1)
        ones[0] = 0
        local_idx = np.cumsum(ones)
        fracs = local_idx / (n_steps_arr[indices] - 1)
        px_flat = p1[0] + fracs * dx[indices]
        py_flat = p1[1] + fracs * dy[indices]
        passable_flat = self.batch_is_passable(px_flat, py_flat, radius)
        fails = ~passable_flat
        fail_counts = np.bincount(indices, weights=fails, minlength=len(p2s))
        return fail_counts == 0

    def batch_line_of_sight_pairs(self, p1s, p2s, radius=0, step_size=0.2):
        if len(p1s) == 0:
            return np.array([], dtype=bool)
        p1s = np.array(p1s)
        p2s = np.array(p2s)
        dx = p2s[:, 0] - p1s[:, 0]
        dy = p2s[:, 1] - p1s[:, 1]
        dist = np.hypot(dx, dy)
        n_steps_arr = np.maximum(2, np.ceil(dist / step_size).astype(int))
        total_steps = np.sum(n_steps_arr)
        indices = np.repeat(np.arange(len(p1s)), n_steps_arr)
        ones = np.ones(total_steps, dtype=int)
        start_idx = np.cumsum(n_steps_arr) - n_steps_arr
        ones[start_idx] = 1 - np.roll(n_steps_arr, 1)
        ones[0] = 0
        local_idx = np.cumsum(ones)
        fracs = local_idx / (n_steps_arr[indices] - 1)
        px_flat = p1s[indices, 0] + fracs * dx[indices]
        py_flat = p1s[indices, 1] + fracs * dy[indices]
        passable_flat = self.batch_is_passable(px_flat, py_flat, radius)
        fails = ~passable_flat
        fail_counts = np.bincount(indices, weights=fails, minlength=len(p1s))
        return fail_counts == 0

    def batch_raycast(self, origin, directions, max_dist=10.0):
        if len(self.compiled_segments) == 0:
            return np.array([]), np.array([])
        Ox, Oy = origin[0], origin[1]
        Dx = directions[:, 0][:, np.newaxis]
        Dy = directions[:, 1][:, np.newaxis]
        Ax, Ay = self._x1, self._y1
        Bx, By = self._x2, self._y2
        R = self._r
        Sx = Bx - Ax
        Sy = By - Ay
        denom = Dx * Sy - Dy * Sx
        denom = np.where(denom == 0, 1e-8, denom)
        t = ((Ax - Ox)*Sy - (Ay - Oy)*Sx) / denom
        u = ((Ax - Ox)*Dy - (Ay - Oy)*Dx) / denom
        valid = (t >= 0) & (u >= -0.1) & (u <= 1.1)
        t_hit = t - R
        t_hit[~valid] = np.inf
        t_hit[t_hit > max_dist] = np.inf
        t_hit[t_hit < 0] = 0
        min_t = np.min(t_hit, axis=1)
        hit_mask = min_t < np.inf
        hit_x = Ox + min_t[hit_mask] * directions[hit_mask, 0]
        hit_y = Oy + min_t[hit_mask] * directions[hit_mask, 1]
        return hit_x, hit_y

    def rasterize(self, cell_size, buffer=0.0):
        cols = int(self.width / cell_size)
        rows = int(self.height / cell_size)
        grid_y, grid_x = np.mgrid[0:rows, 0:cols]
        px = (grid_x.ravel() * cell_size) + (cell_size / 2)
        py = (grid_y.ravel() * cell_size) + (cell_size / 2)
        dist_sq, r = self._points_to_segments_dist_sq(px, py)
        if len(self.compiled_segments) > 0:
            blocked = np.any(dist_sq <= (r + buffer)**2, axis=1)
        else:
            blocked = np.zeros_like(px, dtype=bool)
        return blocked.reshape((rows, cols))

    def resolve_collision(self, px, py, radius, max_iters=10):
        # reuse class-level scratch to avoid per-call numpy allocation
        if not hasattr(self, '_rc_px'):
            self._rc_px = np.empty((1,), dtype=np.float32)
            self._rc_py = np.empty((1,), dtype=np.float32)
        for _ in range(max_iters):
            if len(self.compiled_segments) == 0:
                break
            self._rc_px[0] = px
            self._rc_py[0] = py
            dist_sq, r_seg = self._points_to_segments_dist_sq(self._rc_px, self._rc_py)
            dist_sq = dist_sq[0]
            r_seg = r_seg[0]
            min_dist_idx = np.argmin(dist_sq)
            min_dist = math.sqrt(dist_sq[min_dist_idx])
            buffer = r_seg[min_dist_idx] + radius
            if min_dist < buffer:
                x1 = self.compiled_segments[min_dist_idx, 0]
                y1 = self.compiled_segments[min_dist_idx, 1]
                x2 = self.compiled_segments[min_dist_idx, 2]
                y2 = self.compiled_segments[min_dist_idx, 3]
                dx, dy = x2 - x1, y2 - y1
                l2 = dx*dx + dy*dy
                if l2 == 0:
                    proj_x, proj_y = x1, y1
                else:
                    t = ((px - x1)*dx + (py - y1)*dy) / l2
                    t = max(0.0, min(1.0, t))
                    proj_x = x1 + t * dx
                    proj_y = y1 + t * dy
                nx, ny = px - proj_x, py - proj_y
                n_len = math.hypot(nx, ny)
                if n_len > 0:
                    nx /= n_len
                    ny /= n_len
                else:
                    nx, ny = 0.0, 1.0
                overlap = buffer - min_dist
                px += nx * overlap
                py += ny * overlap
            else:
                break
        px = max(radius, min(self.width - radius, px))
        py = max(radius, min(self.height - radius, py))
        return px, py

    def generate(self, n_obstacles=25, complexity=2):
        cols = int(self.width / self.resolution)
        rows = int(self.height / self.resolution)
        target_area = cols * rows * 0.35
        attempt = 0
        while True:
            attempt += 1
            print(f"Generating map... (Attempt {attempt})")
            self.obstacles = []
            self.pellets = []
            self.power_pellets = []
            t = 0.5
            self.obstacles.append(Capsule(0, 0, self.width, 0, t))
            self.obstacles.append(Capsule(0, self.height, self.width, self.height, t))
            self.obstacles.append(Capsule(0, 0, 0, self.height, t))
            self.obstacles.append(Capsule(self.width, 0, self.width, self.height, t))
            for _ in range(n_obstacles):
                length = random.randint(3, 8)
                thickness = random.uniform(0.4, 0.8)
                points = []
                x, y = random.uniform(2, self.width-2), random.uniform(2, self.height-2)
                angle = random.uniform(0, 2*math.pi)
                points.append((x, y))
                for _ in range(length):
                    angle += random.uniform(-0.8, 0.8)
                    step = random.uniform(2.5, 4.0)
                    x += math.cos(angle) * step
                    y += math.sin(angle) * step
                    points.append((x, y))
                self.obstacles.append(CurvedWall(points, thickness))
            self._compile_obstacles()
            #rasterize and flood fill to find connected safe space
            eval_res = 0.25
            eval_cols = int(self.width / eval_res)
            eval_rows = int(self.height / eval_res)
            target_area_eval = (eval_cols * eval_rows) * 0.35
            grid_y, grid_x = np.mgrid[0:eval_rows, 0:eval_cols]
            px = (grid_x.ravel() * eval_res) + (eval_res / 2)
            py = (grid_y.ravel() * eval_res) + (eval_res / 2)

            import cv2
            
            def get_conn_grid(rad_add):
                grid_cv = np.zeros((eval_rows, eval_cols), dtype=np.uint8)
                for obs in self.obstacles:
                    capsules = obs.capsules if isinstance(obs, CurvedWall) else [obs]
                    for c in capsules:
                        t_res = (c.r + rad_add) / eval_res
                        thick = max(1, int(round(2 * t_res)))
                        pt1 = (int(c.x1 / eval_res), int(c.y1 / eval_res))
                        pt2 = (int(c.x2 / eval_res), int(c.y2 / eval_res))
                        cv2.line(grid_cv, pt1, pt2, 1, thick)
                        r_px = int(round(t_res))
                        if r_px > 0:
                            cv2.circle(grid_cv, pt1, r_px, 1, -1)
                            cv2.circle(grid_cv, pt2, r_px, 1, -1)
                return ~grid_cv.astype(bool)

            # Carve openings to connect all regions
            k_size = min(31, max(3, int(min(eval_cols, eval_rows) // 4) * 2 + 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            for _ in range(200):
                grid = get_conn_grid(0.35)
                num_labels, labeled_array = cv2.connectedComponents(grid.astype(np.uint8), connectivity=8)
                num_features = num_labels - 1   
                if num_features <= 1:
                    break
                component_sizes = np.bincount(labeled_array.ravel())
                component_sizes[0] = 0
                largest_component = component_sizes.argmax()
                valid_components = np.where(component_sizes >= 15)[0]
                if len(valid_components) <= 1:
                    break
                mask_largest = (labeled_array == largest_component).astype(np.uint8)
                mask_other = (np.isin(labeled_array, valid_components) & (labeled_array != largest_component)).astype(np.uint8)
                dilated_largest = cv2.dilate(mask_largest, kernel)
                dilated_other = cv2.dilate(mask_other, kernel)
                boundary = (dilated_largest > 0) & (dilated_other > 0) & (labeled_array == 0)
                boundary_pts = np.argwhere(boundary)
                if len(boundary_pts) == 0:
                    break
                pt = boundary_pts[random.randint(0, len(boundary_pts)-1)]
                bx = pt[1] * eval_res + eval_res / 2
                by = pt[0] * eval_res + eval_res / 2
                best_capsule, best_obs, best_dist = None, None, float('inf')
                for obs in self.obstacles[4:]:
                    capsules_to_check = obs.capsules if isinstance(obs, CurvedWall) else [obs]
                    for c in capsules_to_check:
                        dx, dy = c.x2 - c.x1, c.y2 - c.y1
                        l2 = dx*dx + dy*dy
                        if l2 == 0:
                            d = (bx - c.x1)**2 + (by - c.y1)**2
                        else:
                            t = max(0, min(1, ((bx - c.x1)*dx + (by - c.y1)*dy) / l2))
                            d = (bx - (c.x1 + t * dx))**2 + (by - (c.y1 + t * dy))**2
                        if d < best_dist:
                            best_dist = d
                            best_capsule = c
                            best_obs = obs
                if best_capsule and best_obs:
                    if isinstance(best_obs, CurvedWall) and best_capsule in best_obs.capsules:
                        best_obs.capsules.remove(best_capsule)
                    elif isinstance(best_obs, Capsule) and best_obs in self.obstacles:
                        self.obstacles.remove(best_obs)
                    self._compile_obstacles()
            grid = get_conn_grid(0.35)
            num_labels, labeled_array = cv2.connectedComponents(grid.astype(np.uint8), connectivity=8)
            num_features = num_labels - 1
            if num_features > 0:
                component_sizes = np.bincount(labeled_array.ravel())
                component_sizes[0] = 0
                largest_component = component_sizes.argmax()
                if component_sizes[largest_component] >= target_area_eval:
                    in_largest = (labeled_array == largest_component).ravel()
                    pellet_safe_mask = get_conn_grid(0.40).ravel()
                    valid_mask = in_largest & pellet_safe_mask                    
                    px_valid = px[valid_mask]
                    py_valid = py[valid_mask]
                    self.safe_area = list(zip(px_valid, py_valid))
                    break
                else:
                    print(f"Failed area check: {component_sizes[largest_component]} < {target_area_eval}")
            else:
                print(f"Failed num_features: {num_features}")
            if attempt >= 50:
                print("Max attempts reached! Returning fallback safe area.")
                self.safe_area = [(self.width/2, self.height/2)]
                break
        #filter safe area for pellets (poisson diskish approximation)
        random.shuffle(self.safe_area)
        grid_size = 1.1
        spatial_hash = {}
        for p in self.safe_area:
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
                self.pellets.append(p)
                if (gx, gy) not in spatial_hash:
                    spatial_hash[(gx, gy)] = []
                spatial_hash[(gx, gy)].append(p)
        n_power = min(28, len(self.pellets) // 4)
        if n_power > 0:
            pellets_arr = np.array(self.pellets)
            #Farthest Point Sampling (FPS) to maximize distance between power pellets
            power_indices = [random.randint(0, len(self.pellets)-1)]
            distances = np.sum((pellets_arr - pellets_arr[power_indices[0]])**2, axis=1)
            for _ in range(1, n_power):
                farthest = int(np.argmax(distances))
                power_indices.append(farthest)
                new_dists = np.sum((pellets_arr - pellets_arr[farthest])**2, axis=1)
                distances = np.minimum(distances, new_dists)
            power_indices.sort(reverse=True)
            for idx in power_indices:
                self.power_pellets.append(self.pellets.pop(idx))
        self.generate_roadmap()
    
    def generate_roadmap(self, n_samples=600):
        print("Building continuous PRM (Probabilistic Roadmap)...")
        x_cands = np.random.uniform(0.5, self.width - 0.5, size=n_samples * 5)
        y_cands = np.random.uniform(0.5, self.height - 0.5, size=n_samples * 5)
        passable = self.batch_is_passable(x_cands, y_cands, radius=0.4)
        self.prm_nodes = list(zip(y_cands[passable][:n_samples], x_cands[passable][:n_samples]))        
        for p in self.pellets + self.power_pellets:
            yx_p = (float(p[1]), float(p[0]))
            if yx_p not in self.prm_nodes:
                self.prm_nodes.append(yx_p)
        nodes_arr = np.array(self.prm_nodes, dtype=np.float32)
        self.prm_nodes = [tuple(row) for row in nodes_arr]
        self.prm_graph = {n: [] for n in self.prm_nodes}
        if not self.prm_nodes: return
        import scipy.spatial
        tree = scipy.spatial.cKDTree(nodes_arr)
        pairs = list(tree.query_pairs(r=7.0))
        if pairs:
            p1_idx, p2_idx = zip(*pairs)
            p1_idx, p2_idx = np.array(p1_idx), np.array(p2_idx)
            p1, p2 = nodes_arr[p1_idx], nodes_arr[p2_idx]
            #nodes are (y, x), so index 1 is x, index 0 is y
            dx, dy = p2[:, 1] - p1[:, 1], p2[:, 0] - p1[:, 0]
            dist = np.hypot(dx, dy)
            valid = dist > 0.01
            p1_idx, p2_idx = p1_idx[valid], p2_idx[valid]
            p1, dx, dy, dist = p1[valid], dx[valid], dy[valid], dist[valid]
            if len(dist) > 0:
                step_size = 0.4
                max_steps = max(2, int(np.ceil(dist.max() / step_size)))
                fracs = np.linspace(0, 1.0, max_steps)[:, None]
                px = p1[:, 1] + dx * fracs
                py = p1[:, 0] + dy * fracs
                pass_mat = self.batch_is_passable(px.flatten(), py.flatten(), radius=0.4)
                pass_mat = pass_mat.reshape((max_steps, len(p1)))
                los_valid = np.all(pass_mat, axis=0)
                for k, is_valid in enumerate(los_valid):
                    if is_valid:
                        n1, n2 = self.prm_nodes[p1_idx[k]], self.prm_nodes[p2_idx[k]]
                        dist_val = float(dist[k])
                        self.prm_graph[n1].append((dist_val, n2))
                        self.prm_graph[n2].append((dist_val, n1))
        self.prm_nodes_arr = np.array(self.prm_nodes, dtype=np.float32) if self.prm_nodes else np.empty((0, 2), dtype=np.float32)
        self._update_pellet_arrays()
        self._compile_passable_grids()
        self._compute_apsp()
        
    def _compute_apsp(self):
        import scipy.sparse as sp
        from scipy.sparse import csgraph
        n = len(self.prm_nodes)
        if n == 0:
            self.apsp = np.zeros((0,0))
            self.prm_node_idx = {}
            return
        self.prm_node_idx = {n: i for i, n in enumerate(self.prm_nodes)}
        row, col, data = [], [], []
        for i, n1 in enumerate(self.prm_nodes):
            for cost, n2 in self.prm_graph.get(n1, []):
                j = self.prm_node_idx.get(n2)
                if j is not None:
                    row.append(i)
                    col.append(j)
                    data.append(cost)
        matrix = sp.csr_matrix((data, (row, col)), shape=(n, n))
        print("Computing APSP for PRM graph...")
        self.apsp, self.apsp_pred = csgraph.shortest_path(matrix, directed=False, return_predecessors=True)

    def random_open_point(self):
        if hasattr(self, 'safe_area') and self.safe_area:
            return random.choice(self.safe_area)
        return (self.width / 2.0, self.height / 2.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="World Generation Debug Visualizer")
    parser.add_argument("--obstacles", type=int, default=25, help="Number of curved wall obstacles")
    parser.add_argument("--scale", type=int, default=20, help="Render scale multiplier")
    parser.add_argument("--width", type=int, default=41, help="World width")
    parser.add_argument("--height", type=int, default=33, help="World height")
    args = parser.parse_args()
    pygame.init()
    scale = args.scale
    w, h = args.width, args.height
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("Generated Maze")
    world = World(w, h)
    world.generate(n_obstacles=args.obstacles)
    #spawn agents
    pacman_pos = world.random_open_point()
    ghosts_pos = [world.random_open_point() for _ in range(7)]
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False            
        screen.fill((0, 0, 0))
        for obs in world.obstacles:
            obs.draw(screen, scale)
        for p in world.pellets:
            pygame.draw.circle(screen, (255, 184, 174), (int(p[0]*scale), int(p[1]*scale)), int(0.15*scale))
        for p in world.power_pellets:
            pygame.draw.circle(screen, (255, 255, 255), (int(p[0]*scale), int(p[1]*scale)), int(0.4*scale))
        pygame.draw.circle(screen, (255, 255, 0), (int(pacman_pos[0]*scale), int(pacman_pos[1]*scale)), int(0.5*scale))
        colors = [(255,0,0), (255,184,255), (0,255,255), (255,184,81), (0,255,0), (255,0,255), (255,255,255)]
        for i, gp in enumerate(ghosts_pos):
            pygame.draw.circle(screen, colors[i%len(colors)], (int(gp[0]*scale), int(gp[1]*scale)), int(0.5*scale))
        pygame.display.flip()
        pygame.time.delay(50)
    pygame.quit()