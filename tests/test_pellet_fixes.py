import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import numpy as np
import cv2
from pacman import generate_map, Player
from worker import Env

def test_map_reachability():
    print('--- TEST 1: Map Reachability across 20 seeds ---')
    for seed in range(20):
        np.random.seed(seed)
        random.seed(seed)
        grid, player_start, world = generate_map(33, 41)
        passable = (grid != 1).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(passable, connectivity=4)
        start_label = labels[player_start[0], player_start[1]]
        pellet_cells = np.isin(grid, (2, 3))
        total_pellets = int(np.sum(pellet_cells))
        reachable = int(np.sum(pellet_cells & (labels == start_label)))
        assert total_pellets == reachable, f'Seed {seed}: {reachable}/{total_pellets} reachable!'
    print('✓ Test 1 Passed: 100% of pellets reachable in all 20 seeds!')

def test_ghost_conversion_and_pickup():
    print('--- TEST 2: Ghost Power Pellet Conversion & Pickup ---')
    env = Env(0, num_ghosts=3, world_height=13, world_width=17)
    env.reset()
    pt = env.world.power_pellets[0]
    r = int(pt[1])
    c = int(pt[0])
    g = env.ghosts[0]
    g.x, g.y = pt[0], pt[1]
    g.update((env.player.y, env.player.x), False, env.ghosts)
    assert env.grid[r][c] == 2, f'Expected grid[{r}][{c}] == 2, got {env.grid[r][c]}'
    assert pt in env.world.pellets, f'{pt} not in world.pellets!'
    assert pt not in env.world.power_pellets, f'{pt} still in world.power_pellets!'
    env.player.x, env.player.y = pt[0], pt[1]
    init_score = env.player.score
    env.player.update(env.ghosts)
    assert env.player.score == init_score + 10, f'Expected score +10, got {env.player.score - init_score}'
    assert env.player.powered == False, f'Pacman should NOT be powered from converted pellet!'
    assert env.grid[r][c] == 0, f'Expected grid[{r}][{c}] == 0 (EMPTY), got {env.grid[r][c]}'
    assert pt not in env.world.pellets, f'{pt} was not removed from world.pellets!'
    assert pt not in env.world.power_pellets, f'{pt} in world.power_pellets!'
    print('✓ Test 2 Passed: Ghost conversion updates grid, and Pacman eats it cleanly with no phantom pellet!')

def test_swept_volume_pickup():
    print('--- TEST 3: Swept Volume Pickup (No Tunneling) ---')
    grid, player_start, world = generate_map(13, 17)
    player = Player(grid, player_start, world)
    pellet_pt = world.pellets[0]
    r, c = int(pellet_pt[1]), int(pellet_pt[0])
    player.x = pellet_pt[0] - 0.4
    player.y = pellet_pt[1]
    player.vx = 0.9  
    player.vy = 0.0
    player.update({})
    assert grid[r][c] == 0, f'Expected grid[{r}][{c}] == 0, got {grid[r][c]}'
    assert pellet_pt not in world.pellets, f'{pellet_pt} should have been eaten!'
    print('✓ Test 3 Passed: Swept-volume pickup successfully prevents tunneling!')

if __name__ == '__main__':
    test_map_reachability()
    test_ghost_conversion_and_pickup()
    test_swept_volume_pickup()
    print('\nALL PELLET BUG TESTS PASSED!')