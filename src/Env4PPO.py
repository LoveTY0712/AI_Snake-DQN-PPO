import gymnasium as gym
from gymnasium import spaces
from snake_game import SnakeGame
from config import CONFIG
from stable_baselines3.common.env_checker import check_env
import numpy as np
import wandb
import math
import torch
import pygame



class SnakeGymEnv(gym.Env):
    def __init__(self):
        super(SnakeGymEnv, self).__init__()
        self.game = SnakeGame()  # 初始化自定义的贪吃蛇游戏
        self.action_space = spaces.Discrete(4)  # 动作空间：上下左右
        self.SPEED = 5000
        self.steps = 0
        self.total_steps = 0
        self.total_reward = 0
        self.previous_score_step=0
        self.max_tolerance_step=30

        if CONFIG.log_info:
            wandb.init(
                project='Snake_PPO_Agent_final', 
                config={
                    'model': "PPO"
                }
            )


        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if CONFIG.log_info:
            wandb.log({
                'Total Steps': self.total_steps,
                'Score': self.game.score,
                'Steps by Game': self.steps,
                "Reward":self.total_reward
            })
        super().reset(seed=seed)
        self.game.reset_game()
        self.reward = 0
        self.steps = 0
        self.total_reward = 0
        self.previous_score_step = 0
        
        state = self.get_state()
        return state, {}  # 返回状态和附加信息（新接口要求）
    def move_snake(self):
        if not self.game.game_over:
            head_x, head_y = self.game.snake_pos[0]
            food_distance = math.dist(self.game.snake_pos[0],self.game.food_pos)
            new_head = (head_x + self.game.snake_direction[0], head_y + self.game.snake_direction[1])
            new_food_distance = math.dist(new_head,self.game.food_pos)

            if self.game.is_collision(new_head):
                self.reward = -1
                self.game.game_over = True
            else:
                self.game.snake_pos.insert(0, new_head)  # 在蛇头插入新位置

                if new_head == self.game.food_pos:
                    self.game.score += 1
                    self.reward=1
                    self.previous_score_step=self.steps
                    self.game.food_pos = self.game.generate_food()  # 生成新的食物
                else:
                    self.game.snake_pos.pop()  # 移除蛇尾
                    if self.steps - self.previous_score_step > len(self.game.snake_pos)*2*self.max_tolerance_step:
                        self.reward = -0.05
                        self.game.game_over = True
                        print("for oversteps")
                    elif self.steps- self.previous_score_step > len(self.game.snake_pos)*self.max_tolerance_step:
                        self.reward = -0.01
                    else :
                        self.reward = 0

    def step(self, action):
        if action == 0 and self.game.snake_direction != (0,1):  # 向上
            self.game.snake_direction = (0, -1)
        elif action == 1 and self.game.snake_direction != (0,-1):  # 向下
            self.game.snake_direction = (0, 1)
        elif action == 2 and self.game.snake_direction != (1,0):  # 向左
            self.game.snake_direction = (-1, 0)
        elif action == 3 and self.game.snake_direction != (-1,0):  # 向右
            self.game.snake_direction = (1, 0)

        food_distance = math.dist(self.game.snake_pos[0],self.game.food_pos)

        self.move_snake()
        self.steps += 1
        self.total_steps += 1
        self.total_reward += self.reward
        self.render()
        
        state = self.get_state()

        done = self.game.game_over

        info = {"score": self.game.score}

        return state, self.reward, done, False, info  # 新接口需要返回 `done` 和 `truncated`

    # def get_state(self):
    #     game=self.game
    #     head = game.snake_pos[0]
    #     point_l = (head[0] - 1, head[1])
    #     point_r = (head[0] + 1, head[1])
    #     point_u = (head[0], head[1] - 1)
    #     point_d = (head[0], head[1] + 1)
        
    #     dir_u = game.snake_direction == (0, -1)
    #     dir_d = game.snake_direction == (0, 1)
    #     dir_l = game.snake_direction == (-1, 0)
    #     dir_r = game.snake_direction == (1, 0)

    #     state = [
    #         # Danger straight
    #         (dir_r and game.is_collision(point_r)) or 
    #         (dir_l and game.is_collision(point_l)) or 
    #         (dir_u and game.is_collision(point_u)) or 
    #         (dir_d and game.is_collision(point_d)),

    #         # Danger right
    #         (dir_u and game.is_collision(point_r)) or 
    #         (dir_d and game.is_collision(point_l)) or 
    #         (dir_l and game.is_collision(point_u)) or 
    #         (dir_r and game.is_collision(point_d)),

    #         # Danger left
    #         (dir_d and game.is_collision(point_r)) or 
    #         (dir_u and game.is_collision(point_l)) or 
    #         (dir_r and game.is_collision(point_u)) or 
    #         (dir_l and game.is_collision(point_d)),
            
    #         # Move direction
    #         dir_l,
    #         dir_r,
    #         dir_u,
    #         dir_d,
            
    #         # Food location 
    #         game.food_pos[0] < head[0],  # food left
    #         game.food_pos[0] > head[0],  # food right
    #         game.food_pos[1] < head[1],  # food up
    #         game.food_pos[1] > head[1],  # food down
    #         (game.food_pos[0] - head[0])/22,
    #         (game.food_pos[1] - head[1])/22
    #         ]

    #     return torch.tensor(state,dtype=torch.float32)
    def get_state(self):
        head=self.game.snake_pos[0]
        [xhead, yhead] = [head[0], head[1]]
        [xfood, yfood] = [self.game.food_pos[0], self.game.food_pos[1]]
        deltax = (xfood - xhead) / self.game.X_BLOCK_NUM
        deltay = (yfood - yhead) / self.game.Y_BLOCK_NUM
        checkPoint = [[xhead,yhead-1],[xhead-1,yhead],[xhead,yhead+1],[xhead+1,yhead]]
        tem = [0,0,0,0]
        for coord in self.game.snake_pos[1:-1]+self.game.obstacle_pos:
            if [coord[0],coord[1]] in checkPoint:
                index = checkPoint.index([coord[0],coord[1]])
                tem[index] = 1
        for i,point in enumerate(checkPoint):
            if point[0]>=self.game.X_BLOCK_NUM or point[0]<0 or point[1]>=self.game.Y_BLOCK_NUM or point[1]<0:
                tem[i] = 1
        state = [deltax,deltay]
        state.extend(tem)
        return state
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.show_snake= not self.show_snake
                if event.key == pygame.K_DOWN:
                    if self.clock_tick == 5000:
                        self.clock_tick = 10
                    elif self.clock_tick == 10:
                        self.clock_tick = 5000
    def render(self, mode="human"):
        # self.game.handle_events()
        self.game.handle_events()
        
        self.game.game_display.fill(self.game.BG_COLOR)
        
        self.game.draw_grid()
        self.game.draw_food()
        self.game.draw_obstacle()
        self.game.draw_snake()
        self.game.display_score()
        pygame.display.update()
        self.game.clock.tick(self.SPEED)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    check_env(SnakeGymEnv)