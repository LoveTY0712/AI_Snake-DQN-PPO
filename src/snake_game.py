import pygame
import numpy as np
import random
import sys


class SnakeGame:
    def __init__(self):
        pygame.init()
        self.X_BLOCK_NUM =21
        self.Y_BLOCK_NUM =21
        self.BLOCK_SIZE = 20
        self.WINDOW_WIDTH = self.X_BLOCK_NUM * self.BLOCK_SIZE
        self.WINDOW_HEIGHT = self.Y_BLOCK_NUM * self.BLOCK_SIZE
        self.HEADER_HEIGHT = 40

        self.BG_COLOR = (0, 0, 0)
        self.SNAKE_COLOR = (255, 0, 0)
        self.SNAKE_HEAD_COLOR = (255,255,255)
        self.FOOD_COLOR = (0, 0, 255)
        self.BORDER_COLOR = (125, 125, 125)
        self.OBSTACLE_COLOR = (102,204,255)
        self.TEXT_COLOR = (117, 162, 89)

        self.SPEED = 10

        self.game_display = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT + self.HEADER_HEIGHT))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.clock_tick = 10

        self.snake_pos = [(5, 5), (5, 4), (5, 3)]
        self.obstacle_pos= [(i,8) for i in range(7,14)]+[(i,12) for i in range(7,14)]
        self.snake_direction = (1, 0)  # 初始方向 (向右)
        self.food_pos = self.generate_food()

        self.score = 0
        self.game_over = False

    def generate_food(self):
        while True:
            food_x = random.randint(0, (self.WINDOW_WIDTH // self.BLOCK_SIZE) - 1)
            food_y = random.randint(0, (self.WINDOW_HEIGHT // self.BLOCK_SIZE) - 1)
            if (food_x, food_y) not in self.snake_pos and (food_x, food_y) not in self.obstacle_pos:
                return (food_x, food_y)
    def is_collision(self,pt=None):
        if pt==None:
            pt=self.snake_pos[0]
        if (pt in self.snake_pos[:-1]) or pt in self.obstacle_pos or (pt[0] < 0) or (pt[0] >= self.WINDOW_WIDTH // self.BLOCK_SIZE) or (pt[1] < 0) or (pt[1] >= (self.WINDOW_HEIGHT // self.BLOCK_SIZE)):
            return True
        return False


    def move_snake(self):
        if not self.game_over:
            head_x, head_y = self.snake_pos[0]
            new_head = (head_x + self.snake_direction[0], head_y + self.snake_direction[1])

            if self.is_collision(new_head):
                self.game_over = True
            else:
                self.snake_pos.insert(0, new_head)  # 在蛇头插入新位置

                if new_head == self.food_pos:
                    self.score += 1
                    self.food_pos = self.generate_food()  # 生成新的食物
                else:
                    self.snake_pos.pop()  # 移除蛇尾

    def draw_snake(self):
        colors = np.linspace(self.SNAKE_COLOR, (150,0,0), len(self.snake_pos)).astype(int)
        #渐变色蛇身
        pygame.draw.rect(self.game_display, self.SNAKE_COLOR,
                        (self.snake_pos[0][0] * self.BLOCK_SIZE,
                        self.snake_pos[0][1] * self.BLOCK_SIZE + self.HEADER_HEIGHT,
                        self.BLOCK_SIZE, self.BLOCK_SIZE))

        for i, pos in enumerate(self.snake_pos[1:]):
            pygame.draw.rect(self.game_display, colors[i + 1],
                            (pos[0] * self.BLOCK_SIZE,
                            pos[1] * self.BLOCK_SIZE + self.HEADER_HEIGHT,
                            self.BLOCK_SIZE, self.BLOCK_SIZE))
    def draw_food(self):
        pygame.draw.rect(self.game_display, self.FOOD_COLOR,
                         (self.food_pos[0] * self.BLOCK_SIZE, self.food_pos[1] * self.BLOCK_SIZE + self.HEADER_HEIGHT,
                          self.BLOCK_SIZE, self.BLOCK_SIZE))
    def draw_obstacle(self):
        for pos in self.obstacle_pos:
            pygame.draw.rect(self.game_display, self.OBSTACLE_COLOR,
                             (pos[0] * self.BLOCK_SIZE, pos[1] * self.BLOCK_SIZE + self.HEADER_HEIGHT,
                              self.BLOCK_SIZE, self.BLOCK_SIZE))

    def display_score(self):
        font = pygame.font.SysFont('Arial', 18)
        score_surface = font.render(f'Score: {self.score}', True, self.TEXT_COLOR)
        self.game_display.blit(score_surface, (10, 10))

    def draw_grid(self):
        for x in range(0, self.WINDOW_WIDTH, self.BLOCK_SIZE):
            pygame.draw.line(self.game_display, self.BORDER_COLOR, (x, self.HEADER_HEIGHT), (x, self.WINDOW_HEIGHT + self.HEADER_HEIGHT))
        for y in range(0, self.WINDOW_HEIGHT, self.BLOCK_SIZE):
            pygame.draw.line(self.game_display, self.BORDER_COLOR, (0, y + self.HEADER_HEIGHT), (self.WINDOW_WIDTH, y + self.HEADER_HEIGHT))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.snake_direction != (0, 1):
                    self.snake_direction = (0, -1)  # 向上
                elif event.key == pygame.K_DOWN and self.snake_direction != (0, -1):
                    self.snake_direction = (0, 1)  # 向下
                elif event.key == pygame.K_LEFT and self.snake_direction != (1, 0):
                    self.snake_direction = (-1, 0)  # 向左
                elif event.key == pygame.K_RIGHT and self.snake_direction != (-1, 0):
                    self.snake_direction = (1, 0)  # 向右
                # 使用 W, A, S, D 控制方向
                elif event.key == pygame.K_w and self.snake_direction != (0, 1):
                    self.snake_direction = (0, -1)  # 向上
                elif event.key == pygame.K_s and self.snake_direction != (0, -1):
                    self.snake_direction = (0, 1)  # 向下
                elif event.key == pygame.K_a and self.snake_direction != (1, 0):
                    self.snake_direction = (-1, 0)  # 向左
                elif event.key == pygame.K_d and self.snake_direction != (-1, 0):
                    self.snake_direction = (1, 0)  # 向右

    def reset_game(self):
        self.snake_pos = [(5, 5), (5, 4), (5, 3)]
        self.snake_direction = (1, 0)
        self.food_pos = self.generate_food()
        self.score = 0
        self.game_over = False

    def game_loop(self):
        while True:
            self.handle_events()
            self.move_snake()

            self.game_display.fill(self.BG_COLOR)
            self.draw_grid()
            self.draw_food()
            self.draw_obstacle()
            self.draw_snake()
            self.display_score()

            if self.game_over:
                font = pygame.font.SysFont('Arial', 36)
                game_over_surface = font.render('Game Over!', True, self.TEXT_COLOR)
                self.game_display.blit(game_over_surface, (self.WINDOW_WIDTH // 4, self.WINDOW_HEIGHT // 2))
                pygame.display.update()
                pygame.time.wait(500) 
                self.reset_game()

            pygame.display.update()
            self.clock.tick(self.SPEED)

if __name__ == "__main__":
    game = SnakeGame()
    game.game_loop()