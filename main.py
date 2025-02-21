import pygame, time
import random

# Game settings
FPS = 30 # Frames Per Second
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
CELL_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = SCREEN_WIDTH // CELL_SIZE, SCREEN_HEIGHT // CELL_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Environment (food spawn;)
def generate_food(snake_body):
    while True:
        food_pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)) # random X-coordinaat & Y-coordinaat in the grid
        if food_pos not in snake_body:
            return food_pos

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.running = True
        self.reset_game()

    def reset_game(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.food = generate_food(self.snake)
        self.game_over = False
        self.won = False

    def move_snake(self):
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

# Check for collisions
        if (
            new_head in self.snake or # Snake collides with itself
            new_head[0] < 0 or new_head[0] >= GRID_WIDTH or # Wall collision (left/right)
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT # Wall collision (top/bottom)
        ):
            self.game_over = True
            return
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check if snake eats food
        if new_head == self.food:
            self.food = generate_food(self.snake)
        else:
            self.snake.pop()
        
        # Check win condition
        if len(self.snake) == GRID_WIDTH * GRID_HEIGHT:
            self.won = True
            self.game_over = True

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != DOWN:
                    self.direction = UP
                elif event.key == pygame.K_DOWN and self.direction != UP:
                    self.direction = DOWN
                elif event.key == pygame.K_LEFT and self.direction != RIGHT:
                    self.direction = LEFT
                elif event.key == pygame.K_RIGHT and self.direction != LEFT:
                    self.direction = RIGHT

    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Draw food
        pygame.draw.rect(self.screen, RED, (self.food[0] * CELL_SIZE, self.food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Display game over message
        if self.game_over:
            font = pygame.font.Font(None, 36)
            message = "You Win!" if self.won else "Game Over"
            text = font.render(message, True, WHITE)
            self.screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
        
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_input()
            if not self.game_over:
                self.move_snake()
            self.draw()
            self.clock.tick(10)  # Constant speed (10 FPS)
        
        pygame.quit()

# Run the game
if __name__ == "__main__":
    game = SnakeGame()
    game.run()        