import pygame
import random

# Initialize Pygame
pygame.init()

# Game settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
CELL_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = SCREEN_WIDTH // CELL_SIZE, SCREEN_HEIGHT // CELL_SIZE

# Kleuren
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Richtingen
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Images
snake_head_img = pygame.image.load("images/snake_head.png")
snake_head_img = pygame.transform.scale(snake_head_img, (CELL_SIZE, CELL_SIZE))

# Function head rotation based on richting
def get_rotated_head(image, direction):
    if direction == UP:
        return pygame.transform.rotate(image, -90)
    elif direction == DOWN:
        return pygame.transform.rotate(image, 90)
    elif direction == LEFT:
        return pygame.transform.rotate(image, 180)
    else:
        return image  # Default = Right

# Create game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game Head image")

# Function to generate random food position
def generate_food():
    while True:
        food_pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if food_pos not in snake:  # No food pawn in snake
            return food_pos

# Initialize snake + game variables
snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]  # Start in the center
direction = RIGHT
food = generate_food()
running = True
clock = pygame.time.Clock()

# Game loop
while running:
    screen.fill(BLACK)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and direction != DOWN:
                direction = UP
            elif event.key == pygame.K_DOWN and direction != UP:
                direction = DOWN
            elif event.key == pygame.K_LEFT and direction != RIGHT:
                direction = LEFT
            elif event.key == pygame.K_RIGHT and direction != LEFT:
                direction = RIGHT

    # Move snake
    head_x, head_y = snake[0]
    new_head = (head_x + direction[0], head_y + direction[1])

    # Add new head
    snake.insert(0, new_head)

    # If snake eats food
    if new_head == food:
        food = generate_food()  # Generate new food
    else:
        snake.pop()  # Remove last segment

    # Draw food
    pygame.draw.rect(screen, RED, (food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw snake
    for index, segment in enumerate(snake):
        if index == 0:
            rotated_head_img = get_rotated_head(snake_head_img, direction)
            screen.blit(rotated_head_img, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE))
        else:
            pygame.draw.rect(screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()