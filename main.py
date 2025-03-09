import pygame
import random
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from collections import deque

# Fix ALSA issues
import os
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Game settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
CELL_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = SCREEN_WIDTH // CELL_SIZE, SCREEN_HEIGHT // CELL_SIZE

# Kleuren
BLACK = (0, 0, 0)
WHITE = (255,255,255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Richtingen
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Actions
ACTIONS = [UP, DOWN, LEFT, RIGHT] 

pygame.init()

# Images
snake_head_img = pygame.image.load("images/snake_head.png")
snake_head_img = pygame.transform.scale(snake_head_img, (CELL_SIZE, CELL_SIZE))
food_img = pygame.image.load("images/food.png")
food_img = pygame.transform.scale(food_img, (CELL_SIZE, CELL_SIZE))


# Font
font_counter = pygame.font.SysFont("monospace", 30)
font_gameover = pygame.font.SysFont("Rubberstamplet", 100)

# Screen and clock
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# Neuraal netwerk
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
   
    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        return self.fc3(x)

# Snake Agent
class SnakeAgent:
    def __init__(self, state_size, action_size, reward_history_length=5):
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9
        self.epsilon = 1.0 
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_target_freq = 10 
        self.train_step = 0

        # Initialize reward history
        self.reward_history_length = reward_history_length
        self.reward_history = deque(maxlen=reward_history_length)

        self.fitness = 0
        self.score = 0
   
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return torch.argmax(action_values).item()
   
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        # Update the recent reward history
        self.reward_history.append(reward)

    def get_average_reward(self):
        # Calculate the average of recent rewards
        if len(self.reward_history) > 0:
            return sum(self.reward_history) / len(self.reward_history)
        else:
            return 0.0
   
    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
       
        states = torch.tensor(numpy.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(numpy.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(numpy.array(next_states), dtype=torch.float32)
        dones = torch.tensor(numpy.array(dones), dtype=torch.float32)
       
        q_values = self.model(states).gather(1, actions).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
       
        loss = func.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
       
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

# SnakeGame Class
class SnakeGame:
    def __init__(self, agent):
        self.agent = agent
        self.reward_history = deque(maxlen=5)
        # Initialize the game state
        self.reset()

    def reset(self):
        # Reset game state (snake pos, score, food)
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]  # Snake starts in the middle
        self.direction = RIGHT
        self.food = self.generate_food()  # Random food pos
        self.done = False
        self.score = 0  # Number food eaten
        self.steps = 0  # Number of steps taken

        # Return the initial state of the game
        return self.get_state()

    def generate_food(self):
        # Randomly generate a new food pos
        while True:
            food_pos = (random.randint(1, GRID_WIDTH - 2), random.randint(1, GRID_HEIGHT - 2))
            if food_pos not in self.snake:  # No food spawn in snake
                return food_pos
            
            
    def get_food_direction_vector(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        food_direction = [0, 0, 0, 0]
        if food_y < head_y:
            food_direction[0] = 1
        elif food_y > head_y:
            food_direction[1] = 1
        if food_x < head_x:
            food_direction[2] = 1
        elif food_x > head_x:
            food_direction[3] = 1
        return food_direction

    def distance_to_obstacle(self, direction):
        head_x, head_y = self.snake[0]
        distance = 0
        current = (head_x, head_y)
        while True:
            current = (current[0] + direction[0], current[1] + direction[1])
            distance += 1
            # Controleer of snake buiten grenzen gaat of met het lichaam botst
            if current[0] <= 0 or current[0] >= GRID_WIDTH - 1 or current[1] <= 0 or current[1] >= GRID_HEIGHT - 1 or current in self.snake:
                break
        return distance

# Geeft 1.0 terug als de volgende cel in de gegeven richting gevaarlijk is (botsing), anders 0.0.
    def check_danger(self, direction):
        head_x, head_y = self.snake[0]
        next_cell = (head_x + direction[0], head_y + direction[1])
        if next_cell[0] <= 0 or next_cell[0] >= GRID_WIDTH - 1 or next_cell[1] <= 0 or next_cell[1] >= GRID_HEIGHT - 1 or next_cell in self.snake:
            return 1.0
        else:
            return 0.0

    def get_state(self):
        # Current state of the game: snake head pos, food pos, direction
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        # Afstanden tot de schermranden
        dist_left = head_x
        dist_right = (GRID_WIDTH - 1) - head_x
        dist_top = head_y
        dist_bottom = (GRID_HEIGHT - 1) - head_y

        # Afstanden tot obstakels in elke richting
        dist_obstacle_up = self.distance_to_obstacle(UP)
        dist_obstacle_down = self.distance_to_obstacle(DOWN)
        dist_obstacle_left = self.distance_to_obstacle(LEFT)
        dist_obstacle_right = self.distance_to_obstacle(RIGHT)

        # Danger indicators
        danger_ahead = self.check_danger(self.direction)
        left_direction = (-self.direction[1], self.direction[0])
        right_direction = (self.direction[1], -self.direction[0])
        danger_left = self.check_danger(left_direction)
        danger_right = self.check_danger(right_direction)

        # Food direction
        food_direction = self.get_food_direction_vector()

        # Snake length
        snake_length = len(self.snake)

        # Get the average recent reward from the agent
        average_reward = self.agent.get_average_reward()

        return numpy.array([
            head_x, head_y,
            food_x, food_y,
            self.direction[0], self.direction[1],
            dist_left, dist_right, dist_top, dist_bottom,
            dist_obstacle_up, dist_obstacle_down, dist_obstacle_left, dist_obstacle_right,
            danger_ahead, danger_left, danger_right, *food_direction, snake_length, average_reward
        ], dtype=numpy.float32)

    def step(self, action):
        # Update direction <- action
        new_direction = ACTIONS[action]

        # Prevent the snake from reversing direction
        opposite_directions = {
        UP: DOWN,
        DOWN: UP,
        LEFT: RIGHT,
        RIGHT: LEFT
    }
        if new_direction != opposite_directions[self.direction]:
            self.direction = new_direction

        # Move snake
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        old_distance = abs(head_x - food_x) + abs(head_y - food_y)
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check for collisions (self or walls)
        if (
        new_head in self.snake  # Collision with itself
        or new_head[0] <= 0 or new_head[0] >= GRID_WIDTH - 1  # Wall collision (left/right)
        or new_head[1] <= 0 or new_head[1] >= GRID_HEIGHT - 1  # Wall collision (top/bottom)
        ):
            self.done = True  # Set done to True when game over
            return self.get_state(), -4, self.done  # -1 reward for losing

        # Add new head
        self.snake.insert(0, new_head)

        # If snake eats food
        if new_head == self.food:
            self.food = self.generate_food()  # Generate new food
            self.score += 1  # Increase score
            reward = 10  # Reward for eating food
        else:
            self.snake.pop()  # Remove the tail of the snake if no food eaten       
            new_distance = abs(new_head[0] - food_x) + abs(new_head[1] - food_y)

            if new_distance < old_distance:
                reward = 0.5
            elif new_distance > old_distance:
                reward = -0.5
            else:
                reward = -0.02 + 0.01* len(self.snake)

        # Store the reward in the reward history
        self.reward_history.append(reward)
        
        return self.get_state(), reward, self.done

    def render(self):
        # Render the game
        screen.fill(BLACK)  # Ensure screen cleared -> anders glitch

        # Draw border
        pygame.draw.rect(screen, WHITE, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), CELL_SIZE)

        # Draw food
        food_x, food_y = self.food
        screen.blit(food_img, (food_x * CELL_SIZE, food_y * CELL_SIZE))

        # Draw snake
        for index, segment in enumerate(self.snake):
            if index == 0:  # Snake head
                rotated_head_img = self.get_rotated_head(snake_head_img, self.direction)
                screen.blit(rotated_head_img, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE))
            else:  # Snake body
                pygame.draw.rect(screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Draw the score
        score_text = font_counter.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (SCREEN_WIDTH - 1250, 20))
        pygame.display.flip()  # Update nodig zodat er minder glitch is!!!

    def get_rotated_head(self, image, direction):
        # Rotate the snake head image based on its direction
        if direction == UP:
            return pygame.transform.rotate(image, -90)
        elif direction == DOWN:
            return pygame.transform.rotate(image, 90)
        elif direction == LEFT:
            return pygame.transform.rotate(image, 180)
        else:
            return image  # Default = Right

    def show_game_over(self):
        # Display the "Game Over" message
        screen = pygame.display.get_surface()
        print("Game Over! Score: " + str(self.score))
        screen.fill(BLACK)
        game_over_text = font_gameover.render("GAME OVER", True, RED)
        text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(game_over_text, text_rect)
        pygame.display.flip()
        pygame.time.delay(1000)  # 1sec before quitting

#--------------------------------Genetic Algoritme--------------------------------
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, agent_class, state_size, action_size, reward_history_length=5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.agent_class = agent_class
        self.state_size = state_size
        self.action_size = action_size
        self.population = [self.agent_class(state_size, action_size, reward_history_length)
                           for _ in range(population_size)]
        self.best_agents = []

    def evaluate_fitness(self):
        # Evaluate each agent by having it play a full game (without rendering)
        for agent in self.population:
            game = SnakeGame(agent)
            state = game.reset()
            done = False
            total_reward = 0
            total_score = 0
            while not done:
                action = agent.get_action(state)
                next_state, reward, done = game.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if reward == 10:
                    total_score += 1
            agent.fitness = total_reward
            agent.score = total_score

    def select_parents(self):
        # Select the top 20% of agents as parents
        self.population.sort(key=lambda agent: agent.fitness, reverse=True)
        self.best_agents = self.population[:max(1, len(self.population) // 5)]

    def crossover(self, parent1, parent2):
        child = self.agent_class(self.state_size, self.action_size)
        parent1_weights = parent1.model.state_dict()
        parent2_weights = parent2.model.state_dict()
        child_weights = {}
        for key in parent1_weights:
            if random.random() < 0.5:
                child_weights[key] = parent1_weights[key].clone()
            else:
                child_weights[key] = parent2_weights[key].clone()
        child.model.load_state_dict(child_weights)
        return child

    def mutate(self, agent):
        for param in agent.model.parameters():
            if random.random() < self.mutation_rate:
                param.data += torch.randn(param.size()) * 0.01

    def create_new_generation(self):
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(self.best_agents, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        self.population = new_population

#---------------------------Loop---------------------------------
if __name__ == "__main__":
    population_size = 125
    mutation_rate = 0.05
    generations = 100
    state_size = 23 # Number of inputs
    action_size = 4

    ga = GeneticAlgorithm(population_size, mutation_rate, SnakeAgent, state_size, action_size)
    
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        ga.evaluate_fitness()
        ga.select_parents()
        best_agent = ga.best_agents[0]
        print(f"Best Fitness: {best_agent.fitness}, Best Score: {best_agent.score}")

        game = SnakeGame(best_agent)
        state = game.reset()
        done = False
        while not done:
            # Process events so the window remains responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            action = best_agent.get_action(state)
            next_state, reward, done = game.step(action)
            state = next_state
            game.render()
            clock.tick(20)
        
        # Show game over message (this method now uses self.score)
        game.show_game_over()

        # Wait for a key press before proceeding to the next generation
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    waiting = False
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        ga.create_new_generation()

    pygame.quit()