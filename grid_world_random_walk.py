import random
import numpy as np
import matplotlib.pyplot as plt

# Grid size
grid_size = 4

# Starting position
start_x = random.randint(0, grid_size - 1)
start_y = random.randint(0, grid_size - 1)

# Goal position
goal_x = random.randint(0, grid_size - 1)
goal_y = random.randint(0, grid_size - 1)

# Ensure start and goal positions are different
while start_x == goal_x and start_y == goal_y:
    goal_x = random.randint(0, grid_size - 1)
    goal_y = random.randint(0, grid_size - 1)

# Starting position
x = start_x
y = start_y

start_pos = [x, y]


goal_pos = [goal_x, goal_y]


# Track positions
positions = [(x, y)]

# Perform random walk until goal is found
while (x, y) != (goal_x, goal_y):
    # Randomly select a direction
    direction = random.choice(["up", "down", "left", "right"])
    
    # Update position based on the selected direction
    if direction == "up" and y > 0:
        y -= 1
    elif direction == "down" and y < grid_size - 1:
        y += 1
    elif direction == "left" and x > 0:
        x -= 1
    elif direction == "right" and x < grid_size - 1:
        x += 1
    
    # Track the new position
    positions.append((x, y))

path = np.array(positions)
print(path)
np.save("grid_world_random_walk_path.npy",path)



# Plotting the grid and random walk
fig, ax = plt.subplots()
ax.plot(start_x, start_y, marker='o', color='green', label='Start', markersize =15)
ax.plot(goal_x, goal_y, marker='X', color='red', label='Goal',markersize=15)
for i, pos in enumerate(positions):
    ax.annotate(i, pos, fontsize='small')
x_vals, y_vals = zip(*positions)
ax.plot(x_vals, y_vals, marker='o', linestyle='-', color='blue', label='Random Walk')
ax.grid(True)
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.set_xlim([-0.5, grid_size - 0.5])
ax.set_ylim([-0.5, grid_size - 0.5])
ax.set_title("Random Walk")
ax.set_xlabel("X-coordinate")
ax.set_ylabel("Y-coordinate")
ax.legend()
plt.show()
