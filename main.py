import pygame
import random
import numpy

# Fix ALSA issues
import os
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Initialize Pygame
pygame.init()

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

# Images
snake_head_img = pygame.image.load("images/snake_head.png")
snake_head_img = pygame.transform.scale(snake_head_img, (CELL_SIZE, CELL_SIZE))

# Font
font_counter = pygame.font.SysFont("monospace", 30)
font_gameover = pygame.font.SysFont("Rubberstamplet", 100)

# Screen and clock
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# SnakeGame Class
class SnakeGame:
    def __init__(self):
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

    def get_state(self):
        # Current state of the game: snake head pos, food pos, direction
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return numpy.array([head_x, head_y, food_x, food_y, self.direction[0], self.direction[1]], dtype=numpy.float32)

    def step(self, action):
        # Update direction <- action
        self.direction = ACTIONS[action]

        # Prevent the snake from reversing direction
        if self.direction == LEFT and action == RIGHT :
            action = random.choice([0, 1, 3])
        elif self.direction == RIGHT and action == LEFT :
            action = random.choice([0, 1, 2])
        elif self.direction == DOWN and action == UP :
            action = random.choice([1, 2, 3])
        elif self.direction == UP and action == DOWN :
            action = random.choice([0, 2, 3])

        # Move snake
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check for collisions (self or walls)
        if (
        new_head in self.snake  # Collision with itself
        or new_head[0] <= 0 or new_head[0] >= GRID_WIDTH - 1  # Wall collision (left/right)
        or new_head[1] <= 0 or new_head[1] >= GRID_HEIGHT - 1  # Wall collision (top/bottom)
        ):
            self.done = True  # Set done to True when game over
            return self.get_state(), -1, self.done  # -1 reward for losing

        # Add new head
        self.snake.insert(0, new_head)

        # If snake eats food
        if new_head == self.food:
            self.food = self.generate_food()  # Generate new food
            self.score += 1  # Increase score
            reward = 1  # Reward for eating food
        else:
            self.snake.pop()  # Remove the tail of the snake if no food eaten
            reward = 0  # No reward for just moving

        return self.get_state(), reward, self.done

    def render(self):
        # Render the game
        screen.fill(BLACK)  # Ensure screen cleared -> anders glitch

        # Draw border
        pygame.draw.rect(screen, WHITE, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), CELL_SIZE)

        # Draw food
        food_x, food_y = self.food
        pygame.draw.rect(screen, RED, (food_x * CELL_SIZE, food_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Draw snake
        for index, segment in enumerate(self.snake):
            if index == 0:  # Snake head
                rotated_head_img = self.get_rotated_head(snake_head_img, self.direction)
                screen.blit(rotated_head_img, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE))
            else:  # Snake body
                pygame.draw.rect(screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Draw the score
        score_text = font_counter.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (SCREEN_WIDTH - 150, 20))
        screen.blit(score_text, (SCREEN_WIDTH - 250, 20))

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
        print("Game Over! Score: " + str(game.score))
        screen.fill(BLACK)
        game_over_text = font_gameover.render("GAME OVER", True, RED)
        text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(game_over_text, text_rect)
        pygame.display.flip()
        pygame.time.delay(1000)  # 1sec before quitting


# Game loop TESTING
if __name__ == "__main__":
    game = SnakeGame()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # randomly choose actions
        action = random.randint(0, 3)
        state, reward, done = game.step(action)

        # Display the game window
        game.render()

        if done:
            game.show_game_over()  # Game Over message
            game.reset()  # Reset
            pygame.time.delay(1000)  # 1 sec before resetting