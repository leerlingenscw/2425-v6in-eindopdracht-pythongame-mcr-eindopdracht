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

#Images
snake_head_img = pygame.image.load("snake_head.png")
snake_head_img = pygame.transform.scale(snake_head_img, (CELL_SIZE, CELL_SIZE))  # Resize om grid te fitten

# Function head rotation based on richting
def get_rotated_head(image, direction):
    if direction == UP:
        return pygame.transform.rotate(image, 90)
    elif direction == DOWN:
        return pygame.transform.rotate(image, -90)
    elif direction == LEFT:
        return pygame.transform.rotate(image, 180)
    else:
        return image  # Default = Right
    
# Create game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game Head image")

# Functie food spawn random position
def generate_food():
    return (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

# Generate the first food position
food = generate_food()

# Initialize snake
snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]  # Snake starts in the center
direction = RIGHT  # Initial direction

# Game loop
running = True
clock = pygame.time.Clock()  # Control game speed

while running:
    screen.fill(BLACK)  # Clear screen

    # Draw food
    pygame.draw.rect(screen, RED, (food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw snake
    for index, segment in enumerate(snake):
        if index == 0:  
            # Draw head image
            rotated_head = get_rotated_head(snake_head_img, direction)
            screen.blit(rotated_head, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE))
        else:
            # Draw body as groene blokjes
            pygame.draw.rect(screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
 
    # Move snake
    head_x, head_y = snake[0]
    new_head = (head_x + direction[0], head_y + direction[1])
    snake.insert(0, new_head)  # Add new head
    snake.pop()  # Remove tail

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

    pygame.display.flip()  # Update display
    clock.tick(10)  # Control speed (10 FPS)

# Quit Pygame
pygame.quit()