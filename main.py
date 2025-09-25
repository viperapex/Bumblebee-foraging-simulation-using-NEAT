import os
import pygame
import random
import sys
import math
import time
import threading
import tkinter as tk
from tkinter import messagebox
import neat
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Bumblebee Foraging Simulation")

WHITE = (255, 255, 255)
PINK = (255, 105, 180)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
SPECIAL_FLOWER_COLOR = (0, 0, 255)
BEE_COLOR = (255, 165, 0)
OBSTACLE_COLOR = (128, 128, 128)


flowers = []
special_flowers = []

target_full_hive_bouts = 10
foraging_efficiency = []
search_efficiency = []
weather_changes_enabled = False
rain_enabled = False
dying_flowers_enabled = False
random_spawn_flowers_enabled = False
random_spawn_despawn_enabled = False
random_obstacles_enabled = False
obstacles_enabled = False
visualize_fov = False
visualize_pheromones = False
evaporation_rate = 0.010


class Obstacle:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size

    def draw(self):
        pygame.draw.rect(screen, OBSTACLE_COLOR,
                         pygame.Rect(self.x - self.size // 2, self.y - self.size // 2, self.size, self.size))


class Hive:
    def __init__(self):
        self.x = width // 2
        self.y = height - 50
        self.size = 30
        self.polygon = self.create_hexagon()
        self.total_foraging_bouts = 0
        self.full_hive_bouts = 0

    def create_hexagon(self):
        angle = math.pi * 2 / 6
        return [(self.x + self.size * math.cos(i * angle),
                 self.y + self.size * math.sin(i * angle)) for i in range(6)]

    def draw(self):
        pygame.draw.polygon(screen, YELLOW, self.polygon)


# Environment class for managing pheromones, obstacles, and weather
class Environment:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = np.zeros((width // cell_size, height // cell_size))
        self.obstacles = []
        self.weather = "clear"

    def deposit_pheromone(self, x, y, amount):
        i, j = int(x / self.cell_size), int(y / self.cell_size)
        self.grid[i, j] += amount

    def evaporate_pheromones(self, evaporation_rate):
        self.grid *= (1 - evaporation_rate)

    def reset_pheromones(self):
        self.grid = np.zeros(
            (self.width // self.cell_size, self.height // self.cell_size))
        print("Pheromone trails have been reset.")

    def get_pheromone_level(self, x, y):
        i, j = int(x / self.cell_size), int(y / self.cell_size)
        return self.grid[i, j]

    def add_obstacle(self, x, y, size):
        self.obstacles.append(Obstacle(x, y, size))

    def clear_obstacles(self):
        self.obstacles.clear()

    def is_obstacle(self, x, y):
        for obstacle in self.obstacles:
            if abs(x - obstacle.x) < obstacle.size // 2 and abs(y - obstacle.y) < obstacle.size // 2:
                return True
        return False

    def update_weather(self):
        global weather_changes_enabled, rain_enabled
        if rain_enabled:
            self.weather = "rainy"
        elif weather_changes_enabled:
            if int(time.time()) % 20 < 10:
                self.weather = "clear"
            else:
                self.weather = "rainy"
        else:
            self.weather = "clear"

    def get_weather(self):
        return self.weather

    def draw_obstacles(self):
        for obstacle in self.obstacles:
            obstacle.draw()

    def draw_pheromones(self):
        if visualize_pheromones:
            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    pheromone_level = self.grid[i, j]
                    if pheromone_level > 0.001:
                        intensity = min(int(pheromone_level * 255), 255)
                        color = (0, intensity, 0)
                        pygame.draw.rect(screen, color,
                                         pygame.Rect(i * self.cell_size, j * self.cell_size, self.cell_size,
                                                     self.cell_size), 0)


environment = Environment(width, height, 20)


class Bumblebee:
    def __init__(self, flowers, special_flowers, hive, net):
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
        self.energy = 100
        self.speed = 2
        self.direction = random.uniform(0, 2 * math.pi)
        self.flowers = flowers
        self.special_flowers = special_flowers
        self.visited_flowers = set()
        self.route_length = 0
        self.best_route_length = float('inf')
        self.hive = hive
        self.at_hive = False
        self.hive_arrival_time = 0
        self.foraging_bouts = 0
        self.net = net
        self.total_nectar_collected = 0
        self.flowers_visited = 0
        self.total_distance_traveled = 0
        self.fov_radius = 100

    def update(self):
        if self.at_hive:
            if time.time() - self.hive_arrival_time >= 5:
                self.at_hive = False
                self.energy = 100
            return

        environment.deposit_pheromone(self.x, self.y, 1.0)

        if self.energy <= 0 or len(self.visited_flowers) == len(self.flowers) + len(self.special_flowers):
            self.return_to_hive()
            return

        if environment.get_weather() == "rainy":
            self.speed = max(1, self.speed * 0.5)
            self.energy -= 0.15
            self.fov_radius = 60
        else:
            self.fov_radius = 100

        inputs = self.get_inputs()
        output = self.net.activate(inputs)
        self.process_output(output)

        nearest_flower = self.find_nearest_flower()
        if nearest_flower:
            self.move_towards(nearest_flower)
            if self.distance_to(nearest_flower) < 5:
                self.visit_flower(nearest_flower)

    def get_inputs(self):
        nearest_flower = self.find_nearest_flower()
        distance = self.distance_to(nearest_flower) if nearest_flower else 1.0

        pheromone_level = environment.get_pheromone_level(self.x, self.y)

        weather = environment.get_weather()
        weather_input = 1 if weather is None or weather == "clear" else 0

        nearest_landmark = self.find_nearest_flower_to_hive()
        landmark_distance = self.distance_to(
            nearest_landmark) if nearest_landmark else 1.0

        inputs = [
            distance,
            self.energy,
            pheromone_level,
            weather_input,
            landmark_distance
        ]

        return inputs

    def find_nearest_flower_to_hive(self):
        min_distance = float('inf')
        nearest_flower = None
        for flower in self.flowers + self.special_flowers:
            distance = math.sqrt((self.hive.x - flower.x)
                                 ** 2 + (self.hive.y - flower.y) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_flower = flower
        return nearest_flower

    def process_output(self, output):
        self.direction += output[0] * 2 * math.pi - math.pi
        self.speed = max(1, min(5, self.speed + output[1] * 2 - 1))

    def find_nearest_flower(self):
        min_distance = float('inf')
        nearest_flower = None

        for flower in self.special_flowers:
            if flower not in self.visited_flowers:
                distance = self.distance_to(flower)
                if distance < min_distance:
                    min_distance = distance
                    nearest_flower = flower

        if nearest_flower is None:
            for flower in self.flowers:
                if flower not in self.visited_flowers:
                    distance = self.distance_to(flower)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_flower = flower

        return nearest_flower

    def move_towards(self, target):
        direction = math.atan2(target.y - self.y, target.x - self.x)
        if environment.is_obstacle(self.x + self.speed * math.cos(direction),
                                   self.y + self.speed * math.sin(direction)):
            direction += random.uniform(-math.pi / 2,
                                        math.pi / 2)

        self.x += self.speed * math.cos(direction)
        self.y += self.speed * math.sin(direction)
        self.route_length += self.speed
        self.energy -= 0.1
        self.total_distance_traveled += self.speed

    def visit_flower(self, flower):
        self.visited_flowers.add(flower)
        self.energy += flower.nectar
        self.total_nectar_collected += flower.nectar
        self.flowers_visited += 1
        if self.route_length < self.best_route_length:
            self.best_route_length = self.route_length
            self.speed = min(self.speed * 1.1, 5)

    def return_to_hive(self):
        if self.distance_to(self.hive) > 5:
            self.move_towards(self.hive)
            environment.deposit_pheromone(self.x, self.y, 1.0)
        else:
            self.visited_flowers.clear()
            self.route_length = 0
            self.x, self.y = self.hive.x, self.hive.y
            self.at_hive = True
            self.hive_arrival_time = time.time()
            self.foraging_bouts += 1
            self.hive.total_foraging_bouts += 1

    def distance_to(self, obj):
        return math.sqrt((self.x - obj.x) ** 2 + (self.y - obj.y) ** 2)

    def draw(self):
        color = RED if self.at_hive else BEE_COLOR
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 5)

        if visualize_fov:
            end_angle = self.direction + math.pi / 3
            start_angle = self.direction - math.pi / 3
            points = [(self.x, self.y)]
            for angle in np.linspace(start_angle, end_angle, 30):
                points.append((self.x + self.fov_radius * math.cos(angle),
                              self.y + self.fov_radius * math.sin(angle)))
            pygame.draw.polygon(screen, (0, 0, 255, 50), points, 1)


# Define Flower class
class Flower:
    def __init__(self, x=None, y=None, special=False):
        self.x = x if x is not None else random.randint(0, width)
        self.y = y if y is not None else random.randint(0, height)
        self.nectar = random.randint(10, 30)
        self.special = special

    def reposition(self):
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)

    def draw(self):
        color = SPECIAL_FLOWER_COLOR if self.special else PINK
        pygame.draw.circle(screen, color, (self.x, self.y), 5)


def create_random_array(num_flowers):
    global flowers
    flowers.clear()
    for _ in range(num_flowers):
        flowers.append(Flower())
    print("Created random array of flowers.")


def create_independent_array(num_flowers):
    global flowers
    flowers.clear()
    rows = int(math.sqrt(num_flowers))
    cols = num_flowers // rows
    x_spacing = width // (cols + 1)
    y_spacing = height // (rows + 1)
    for i in range(rows):
        for j in range(cols):
            if len(flowers) < num_flowers:
                flowers.append(
                    Flower(x=(j + 1) * x_spacing, y=(i + 1) * y_spacing))
    print("Created independent array of flowers.")


def create_positive_array(num_flowers):
    global flowers
    flowers.clear()
    radius = min(width, height) // 4
    angle_increment = 2 * math.pi / num_flowers
    for i in range(num_flowers):
        angle = i * angle_increment
        x = width // 2 + int(radius * math.cos(angle))
        y = height // 2 + int(radius * math.sin(angle))
        flowers.append(Flower(x=x, y=y))
    print("Created positive array of flowers.")


def create_negative_array(num_flowers):
    global flowers
    flowers.clear()
    if num_flowers < 10:
        num_flowers = 10

    positions = [
        (150, 100), (450, 100), (300, 150), (100, 300), (500, 300),
        (150, 500), (450, 500), (300, 450), (200, 300), (400, 300)
    ]

    for i in range(num_flowers):
        pos = positions[i % 10]
        flowers.append(Flower(x=pos[0], y=pos[1]))

    print("Created negative array of flowers.")


# New Arrays

def create_negative_array_v2(num_flowers):
    global flowers
    flowers.clear()
    positions = [
        (100, 100), (300, 150), (200, 200),
        (400, 250), (300, 300), (500, 350),
        (400, 400), (600, 450), (500, 500), (700, 550)
    ]
    for i in range(num_flowers):
        flowers.append(Flower(x=positions[i][0], y=positions[i][1]))
    print("Created negative array v2 of flowers.")


def create_independent_array_v2(num_flowers):
    global flowers
    flowers.clear()

    base_length = 3

    spacing_x = 150
    spacing_y = 100

    start_x = width // 2
    start_y = height - 300

    positions = []

    # Row 1 (top row with 4 flowers)
    for i in range(base_length + 1):
        x = start_x - (base_length * spacing_x // 2) + (i * spacing_x)
        y = start_y - (2 * spacing_y)
        positions.append((x, y))

    # Row 2 (middle row with 3 flowers)
    for i in range(base_length):
        x = start_x - ((base_length - 1) * spacing_x // 2) + (i * spacing_x)
        y = start_y - spacing_y
        positions.append((x, y))

    # Row 3 (bottom row with 2 flowers)
    for i in range(base_length - 1):
        x = start_x - ((base_length - 2) * spacing_x // 2) + (i * spacing_x)
        y = start_y
        positions.append((x, y))

    additional_flower_x = start_x
    additional_flower_y = start_y + spacing_y
    positions.append((additional_flower_x, additional_flower_y))

    for i in range(num_flowers):
        flowers.append(Flower(x=positions[i][0], y=positions[i][1]))

    print("Created equidistant inverted triangle array of flowers with an additional flower above the hive.")


def create_positive_array_v2(num_flowers):
    global flowers
    flowers.clear()
    positions = [
        (250, 100), (300, 150), (350, 200),
        (400, 250), (450, 300), (500, 350),
        (550, 400), (600, 450), (650, 500), (700, 550)
    ]
    for i in range(num_flowers):
        flowers.append(Flower(x=positions[i][0], y=positions[i][1]))
    print("Created positive array v2 of flowers.")


def initialize_simulation(num_flowers, num_special_flowers, num_bees, array_type='random', net=None):
    global flowers, special_flowers, bees, hive, foraging_efficiency, search_efficiency, obstacles_enabled

    # initializing hive here
    hive = Hive()

    if 'v2' in array_type:
        hive.y = height - 50  # Place hive at the bottom for v2 arrays

    foraging_efficiency.clear()
    search_efficiency.clear()

    environment.reset_pheromones()

    if array_type == 'positive':
        create_positive_array(num_flowers)
    elif array_type == 'independent':
        create_independent_array(num_flowers)
    elif array_type == 'negative':
        create_negative_array(num_flowers)
    elif array_type == 'positive_v2':
        create_positive_array_v2(num_flowers)
    elif array_type == 'independent_v2':
        create_independent_array_v2(num_flowers)
    elif array_type == 'negative_v2':
        create_negative_array_v2(num_flowers)
    else:
        create_random_array(num_flowers)

    special_flowers = [Flower(special=True)
                       for _ in range(num_special_flowers)]

    if obstacles_enabled:
        create_obstacles()

    if net is None:
        if bees:
            net = bees[0].net
        else:
            genome = neat.DefaultGenome(0)
            net = neat.nn.FeedForwardNetwork.create(genome, config)

    bees = [Bumblebee(flowers, special_flowers, hive, net)
            for _ in range(num_bees)]
    print(
        f"Initialized simulation with {num_flowers} flowers, {num_special_flowers} special flowers, and {num_bees} bees.")


def create_obstacles():
    global environment
    num_obstacles = random.randint(5, 10)
    for _ in range(num_obstacles):
        x = random.randint(50, width - 50)
        y = random.randint(50, height - 50)
        size = random.randint(20, 50)
        environment.add_obstacle(x, y, size)
    print(f"Created {num_obstacles} obstacles.")


def create_random_obstacles():
    global environment
    environment.clear_obstacles()
    num_obstacles = random.randint(5, 10)
    for _ in range(num_obstacles):
        x = random.randint(50, width - 50)
        y = random.randint(50, height - 50)
        size = random.randint(20, 50)
        environment.add_obstacle(x, y, size)
    print(f"Created {num_obstacles} random obstacles after full hive bout.")


def add_special_flower():
    global special_flowers
    special_flowers.append(Flower(special=True))
    print(
        f"Added a special flower. Total special flowers: {len(special_flowers)}")


def add_normal_flower():
    global flowers
    flowers.append(Flower())
    print(f"Added a normal flower. Total flowers: {len(flowers)}")


def randomly_spawn_flower():
    global flowers
    flowers.append(Flower())
    print(f"Randomly spawned a flower. Total flowers: {len(flowers)}")


def reposition_flowers():
    global flowers, special_flowers
    for flower in flowers + special_flowers:
        flower.reposition()
    print("Repositioned all flowers.")


def delete_flower_at_position(pos):
    global flowers, special_flowers
    x, y = pos
    for flower_list in (flowers, special_flowers):
        for flower in flower_list[:]:
            if math.sqrt((flower.x - x) ** 2 + (flower.y - y) ** 2) < 5:
                flower_list.remove(flower)
    print("Deleted flower at position:", pos)


def add_bee():
    global hive
    if bees:
        net = bees[0].net
    else:
        genome = neat.DefaultGenome(0)
        net = neat.nn.FeedForwardNetwork.create(genome, config)

    new_bee = Bumblebee(flowers, special_flowers, hive, net)
    new_bee.x, new_bee.y = hive.x, hive.y
    bees.append(new_bee)

    print(f"Bee added. Total bees: {len(bees)}")


def increase_speed():
    global bees
    for bee in bees:
        bee.speed = min(bee.speed + 1.0, 10)
    print("Increased speed of all bees.")


def decrease_speed():
    global bees
    for bee in bees:
        bee.speed = max(bee.speed - 1.0, 0.5)
    print("Decreased speed of all bees.")


def randomly_kill_flower():
    global flowers
    if flowers:
        flower_to_kill = random.choice(flowers)
        flowers.remove(flower_to_kill)
        print("A flower has died.")


def start_dying_flowers():
    global dying_flowers_enabled
    while dying_flowers_enabled:
        randomly_kill_flower()
        time.sleep(random.uniform(1, 3))


def start_random_spawning_flowers():
    global random_spawn_flowers_enabled
    while random_spawn_flowers_enabled:
        randomly_spawn_flower()
        time.sleep(random.uniform(2, 5))


def random_spawn_despawn_flowers():
    global flowers, random_spawn_despawn_enabled
    if random_spawn_despawn_enabled:
        num_to_change = max(1, int(len(flowers) * 0.1))

        for _ in range(num_to_change):
            if flowers:
                flower_to_remove = random.choice(flowers)
                flowers.remove(flower_to_remove)
                print("A flower has been removed due to random spawn/despawn.")

        for _ in range(num_to_change):
            flowers.append(Flower())
            print("A new flower has been spawned due to random spawn/despawn.")


def show_help():
    help_window = tk.Toplevel()
    help_window.title("Help")
    help_text = tk.Text(help_window, wrap='word', width=60, height=20)
    help_text.pack(expand=True, fill='both')

    instructions = """
    1.FHB stands for full hive bouts and the associated changes appear after a full hive bout is completed
    2.Array types requires the user to start the simulation again with the array type selected
    3.Rest of the options can be enabled mid simulation
    4.The Simulation may crash sometime as it still has some bugs. Try to restart in your IDE (source code)
    5.Sometime bees conserve energy initially by moving slower which leads to them learning to move slow. If their behavior doesnt change restart the simulation, by executing the source again.
    """
    help_text.insert(tk.END, instructions)
    help_text.config(state='disabled')

    tk.Button(help_window, text="Close", command=help_window.destroy).pack()


def configure_simulation():
    root = tk.Tk()
    root.title("Bumblebee Foraging Simulation Configuration")

    tk.Label(root, text="Number of Flowers:").pack()
    entry_flowers = tk.Entry(root)
    entry_flowers.insert(0, '15')
    entry_flowers.pack()

    tk.Label(root, text="Number of Special Flowers:").pack()
    entry_special_flowers = tk.Entry(root)
    entry_special_flowers.insert(0, '0')
    entry_special_flowers.pack()

    tk.Label(root, text="Number of Bumblebees:").pack()
    entry_bees = tk.Entry(root)
    entry_bees.insert(0, '10')
    entry_bees.pack()

    tk.Label(root, text="Full Hive Bouts (to end simulation):").pack()
    entry_full_hive_bouts = tk.Entry(root)
    entry_full_hive_bouts.insert(0, '10')
    entry_full_hive_bouts.pack()

    weather_toggle = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Enable Weather Changes",
                   variable=weather_toggle, command=update_weather_changes).pack()

    rain_toggle = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Enable Rain",
                   variable=rain_toggle, command=update_rain).pack()

    dying_flowers_toggle = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Dying Flowers",
                   variable=dying_flowers_toggle, command=update_dying_flowers).pack()

    random_spawn_flowers_toggle = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Flower Spawning", variable=random_spawn_flowers_toggle,
                   command=update_random_spawn).pack()

    random_spawn_despawn_toggle = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Spawn/Despawn Flowers FHB",
                   variable=random_spawn_despawn_toggle, command=update_random_spawn_despawn).pack()

    obstacles_toggle = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Obstacles",
                   variable=obstacles_toggle, command=update_obstacles).pack()

    random_obstacles_toggle = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Random Obstacles FHB", variable=random_obstacles_toggle,
                   command=update_random_obstacles).pack()

    fov_toggle = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Visualize Bees' Fields of View", variable=fov_toggle,
                   command=update_visualize_fov).pack()

    pheromones_toggle = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Visualize Pheromones", variable=pheromones_toggle,
                   command=update_visualize_pheromones).pack()

    tk.Label(root, text="Array Type:").pack()
    array_type = tk.StringVar(value='random')
    tk.Radiobutton(root, text="Positive", variable=array_type,
                   value='positive').pack()
    tk.Radiobutton(root, text="Independent",
                   variable=array_type, value='independent').pack()
    tk.Radiobutton(root, text="Negative", variable=array_type,
                   value='negative').pack()
    tk.Radiobutton(root, text="Positive v2",
                   variable=array_type, value='positive_v2').pack()
    tk.Radiobutton(root, text="Independent v2",
                   variable=array_type, value='independent_v2').pack()
    tk.Radiobutton(root, text="Negative v2",
                   variable=array_type, value='negative_v2').pack()
    tk.Radiobutton(root, text="Random", variable=array_type,
                   value='random').pack()
    tk.Button(root, text="Help", command=show_help).pack()

    def start_simulation():
        try:
            num_flowers = int(entry_flowers.get())
            num_special_flowers = int(entry_special_flowers.get())
            num_bees = int(entry_bees.get())
            global target_full_hive_bouts, weather_changes_enabled, rain_enabled, dying_flowers_enabled, random_spawn_flowers_enabled, random_spawn_despawn_enabled, obstacles_enabled, random_obstacles_enabled, visualize_fov, visualize_pheromones
            target_full_hive_bouts = int(entry_full_hive_bouts.get())
            initialize_simulation(
                num_flowers, num_special_flowers, num_bees, array_type.get())

            if dying_flowers_enabled:
                dying_flowers_thread = threading.Thread(
                    target=start_dying_flowers)
                dying_flowers_thread.daemon = True
                dying_flowers_thread.start()

            if random_spawn_flowers_enabled:
                random_spawn_flowers_thread = threading.Thread(
                    target=start_random_spawning_flowers)
                random_spawn_flowers_thread.daemon = True
                random_spawn_flowers_thread.start()

        except ValueError:
            messagebox.showerror(
                "Error", "Please enter valid integer values for all fields.")

    tk.Button(root, text="Start Simulation", command=start_simulation).pack()
    tk.Button(root, text="Add Special Flower",
              command=add_special_flower).pack()
    tk.Button(root, text="Add Normal Flower", command=add_normal_flower).pack()
    tk.Button(root, text="Add Bee", command=add_bee).pack()
    tk.Button(root, text="Increase Speed", command=increase_speed).pack()
    tk.Button(root, text="Decrease Speed", command=decrease_speed).pack()
    tk.Button(root, text="Reposition Flowers",
              command=reposition_flowers).pack()

    root.mainloop()


# Event handlers for Tkinter toggles
def update_weather_changes():
    global weather_changes_enabled
    weather_changes_enabled = not weather_changes_enabled


def update_rain():
    global rain_enabled
    rain_enabled = not rain_enabled


def update_dying_flowers():
    global dying_flowers_enabled
    dying_flowers_enabled = not dying_flowers_enabled
    if dying_flowers_enabled:
        dying_flowers_thread = threading.Thread(target=start_dying_flowers)
        dying_flowers_thread.daemon = True
        dying_flowers_thread.start()


def update_random_spawn():
    global random_spawn_flowers_enabled
    random_spawn_flowers_enabled = not random_spawn_flowers_enabled
    if random_spawn_flowers_enabled:
        random_spawn_flowers_thread = threading.Thread(
            target=start_random_spawning_flowers)
        random_spawn_flowers_thread.daemon = True
        random_spawn_flowers_thread.start()


def update_random_spawn_despawn():
    global random_spawn_despawn_enabled
    random_spawn_despawn_enabled = not random_spawn_despawn_enabled


def update_obstacles():
    global obstacles_enabled
    obstacles_enabled = not obstacles_enabled


def update_random_obstacles():
    global random_obstacles_enabled
    random_obstacles_enabled = not random_obstacles_enabled


def update_visualize_fov():
    global visualize_fov
    visualize_fov = not visualize_fov


def update_visualize_pheromones():
    global visualize_pheromones
    visualize_pheromones = not visualize_pheromones


tk_thread = threading.Thread(target=configure_simulation)
tk_thread.daemon = True
tk_thread.start()

# NEAT evaluation function
generation = 0
fitness_history = []


def eval_genomes(genomes, config):
    global generation
    generation += 1
    print(f"Generation: {generation}")

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        initialize_simulation(15, 0, 10, net=net)
        fitness = run_simulation()
        genome.fitness = fitness
        print(f"Genome {genome_id} fitness: {genome.fitness}")

    fitness_history.append(
        max([genome.fitness for genome_id, genome in genomes]))


def run_simulation():
    global running, evaporation_rate
    running = True
    clock = pygame.time.Clock()

    def calculate_full_hive_bouts(bees):
        min_bouts = min(bee.foraging_bouts for bee in bees)
        return min_bouts

    print("Simulation started.")
    start_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                delete_flower_at_position(pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    evaporation_rate = min(evaporation_rate + 0.01, 1.0)
                    print(
                        f"Evaporation rate increased to {evaporation_rate:.2f}")
                elif event.key == pygame.K_DOWN:
                    evaporation_rate = max(evaporation_rate - 0.01, 0.0)
                    print(
                        f"Evaporation rate decreased to {evaporation_rate:.2f}")

        screen.fill(WHITE)

        environment.update_weather()

        if environment.get_weather() == "rainy":
            adjusted_evaporation_rate = min(evaporation_rate * 2, 0.99)
        else:
            adjusted_evaporation_rate = evaporation_rate

        hive.draw()

        environment.draw_obstacles()
        environment.draw_pheromones()

        for flower in flowers + special_flowers:
            flower.draw()

        for bee in bees:
            bee.update()
            bee.draw()

        environment.evaporate_pheromones(adjusted_evaporation_rate)

        font = pygame.font.Font(None, 26)
        text_y_position = 10
        y_offset = 20

        total_foraging_bouts = hive.total_foraging_bouts
        text = font.render(
            f"Individual Foraging Bouts: {total_foraging_bouts}", True, BLACK)
        screen.blit(text, (10, text_y_position))

        text_y_position += y_offset
        full_hive_bouts = calculate_full_hive_bouts(bees)
        text = font.render(f"Full Hive Bouts: {full_hive_bouts}", True, BLACK)
        screen.blit(text, (10, text_y_position))

        text_y_position += y_offset
        text = font.render(f"Bee Count: {len(bees)}", True, BLACK)
        screen.blit(text, (10, text_y_position))

        text_y_position += y_offset
        flower_count = len(flowers) + len(special_flowers)
        text = font.render(f"Flower Count: {flower_count}", True, BLACK)
        screen.blit(text, (10, text_y_position))

        text_y_position += y_offset
        if bees:
            avg_speed = sum(bee.speed for bee in bees) / len(bees)
            speed_text = font.render(
                f"Average Speed: {avg_speed:.2f}", True, BLACK)
        else:
            speed_text = font.render("Average Speed: N/A", True, BLACK)
        screen.blit(speed_text, (10, text_y_position))

        text_y_position += y_offset
        weather_status = environment.get_weather()
        weather_text = font.render(
            f"Weather: {weather_status.capitalize()}", True, BLACK)
        screen.blit(weather_text, (10, text_y_position))

        pygame.display.flip()
        clock.tick(30)

        if full_hive_bouts > 0:
            nectar_collected = sum(
                bee.total_nectar_collected for bee in bees) / full_hive_bouts
            foraging_efficiency.append(nectar_collected)
            avg_search_efficiency = sum(
                bee.flowers_visited / bee.total_distance_traveled if bee.total_distance_traveled > 0 else 0 for bee in
                bees) / len(bees)
            search_efficiency.append(avg_search_efficiency)

            if random_obstacles_enabled and full_hive_bouts > hive.full_hive_bouts:
                create_random_obstacles()
                hive.full_hive_bouts = full_hive_bouts

        if full_hive_bouts >= target_full_hive_bouts:
            print(
                f"Simulation ended after reaching {full_hive_bouts} full hive bouts.")
            break

    print("Simulation finished. Duration:", time.time() - start_time)
    pygame.quit()

    plot_efficiencies()

    sys.exit()

    return random.random()


def plot_efficiencies():
    plt.figure(figsize=(10, 6))
    plt.plot(foraging_efficiency,
             label='Foraging Efficiency (Nectar Collected per Bout)', color='blue')
    plt.xlabel('Data Points - Full Hive Bouts')
    plt.ylabel('Foraging Efficiency')
    plt.title('Foraging Efficiency Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(search_efficiency,
             label='Search Efficiency (Flowers Visited per Distance)', color='green')
    plt.xlabel('Data Points - Full Hive Bouts')
    plt.ylabel('Search Efficiency')
    plt.title('Search Efficiency Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


# Load NEAT configuration
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "config-feedforward")
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)

population = neat.Population(config)

population.run(eval_genomes, 50)
