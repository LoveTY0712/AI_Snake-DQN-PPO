from snake_game import SnakeGame
from my_dqn import DQN_AGENT,CONFIG
import numpy as np
import torch
import wandb
import pygame
import math
import copy
import os

os.environ["WANDB_MODE"] = "offline"

device=torch.device("cuda" if CONFIG.use_gpu and torch.cuda.is_available() else "cpu")
print(f"using device:{device}")

class GmaeAgent(SnakeGame):
    def __init__(self,agent):
        super().__init__()
        self.dqn_agent=agent
        self.action=3  #1 up 2 down 3 left 4 right
        self.reward=0
        self.total_reward=0
        self.step=0
        self.show_snake=True
        self.clock_tick=5000
        self.previous_score_step=0
        self.max_tolerance_step=40
        # 初始化一个 W&B 实验/项目
        if CONFIG.log_info:
            wandb.init(
                project='Snake_DQN_Agent_final', 
                config={
                    'model': "DQN",
                    'learning_rate': self.dqn_agent.learning_rate,
                    'epsilon': self.dqn_agent.epsilon
                }
            )

    def move_snake_and_get_reward(self,action):
        if action not in [1,2,3,4]:
            return
        if action == 1 and self.snake_direction != (0, 1):
            self.snake_direction = (0, -1)  # 向上
        elif action == 2 and self.snake_direction != (0, -1):
            self.snake_direction = (0, 1)  # 向下
        elif action == 3 and self.snake_direction != (1, 0):
            self.snake_direction = (-1, 0)  # 向左
        elif action == 4 and self.snake_direction != (-1, 0):
            self.snake_direction = (1, 0)  # 向右
        if not self.game_over:
            head_x, head_y = self.snake_pos[0]
            new_head = (head_x + self.snake_direction[0], head_y + self.snake_direction[1])

            if (new_head[0] < 0) or (new_head[0] >= self.WINDOW_WIDTH // self.BLOCK_SIZE) or (new_head[1] < 0) or (new_head[1] >= (self.WINDOW_HEIGHT // self.BLOCK_SIZE))  or new_head in self.obstacle_pos:
                self.game_over = True
                self.reward = -2
            elif new_head in self.snake_pos[:-1]:
                self.game_over = True
                self.reward = -5
            else:
                self.snake_pos.insert(0, new_head)  # 在蛇头插入新位置

                if new_head == self.food_pos:
                    self.reward = 1
                    self.score += 1
                    self.previous_score_step = self.step
                    self.food_pos = self.generate_food()  # 生成新的食物
                else:
                    self.reward = 0
                    self.snake_pos.pop()  # 移除蛇尾

    def display_steps(self):
        font = pygame.font.SysFont('Arial', 18)
        score_surface = font.render(f'Step: {self.dqn_agent.total_steps}', True, self.TEXT_COLOR)
        self.game_display.blit(score_surface, (230, 10))
    def display_circle(self):
        font = pygame.font.SysFont('Arial', 18)
        score_surface = font.render(f'Circle: {self.dqn_agent.total_circles}', True, self.TEXT_COLOR)
        self.game_display.blit(score_surface, (100, 10))

    def draw_game(self):
        self.game_display.fill(self.BG_COLOR)

        self.draw_grid()
        self.draw_food()
        self.draw_obstacle()
        self.draw_snake()
        self.display_score()
        self.display_steps()
        self.display_circle()
        pygame.display.update()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                #切换画面帧数，5000用于训练，10用于观察
                if event.key == pygame.K_DOWN:
                    if self.clock_tick == 5000:
                        self.clock_tick = 10
                    elif self.clock_tick == 10:
                        self.clock_tick = 5000
    def reset_game(self):
        self.dqn_agent.total_circles+=1
        if CONFIG.log_info:
            wandb.log({
                'Total Steps': self.dqn_agent.total_steps,
                'Score': self.score,
                'Steps by Game': self.step,
                'epsilon':self.dqn_agent.epsilon,
                "Reward":self.total_reward
            })
        super().reset_game()
        self.reward=0
        self.action=3
        self.step=0
        self.previous_score_step=0
        self.total_reward=0
    def get_game_state(self):
        game_state=torch.ones((self.WINDOW_HEIGHT // self.BLOCK_SIZE, self.WINDOW_WIDTH // self.BLOCK_SIZE),device=device)
        game_state[self.snake_pos[0][0]][self.snake_pos[0][1]]=9
        for x,y in self.snake_pos[1:]:
            game_state[x][y]=3
        for x,y in self.obstacle_pos:
            game_state[x][y]=-1
        game_state[self.food_pos[0]][self.food_pos[1]]=20
        return game_state
    def agent_play_game(self):
        while True:
            self.handle_events()
            state=self.get_game_state()

            food_distance = math.dist(self.snake_pos[0],self.food_pos)

            if self.step>0:
                self.action=self.dqn_agent.select_next_action(state)
            previous_state=copy.deepcopy(self.dqn_agent.saved_state)

            self.move_snake_and_get_reward(self.action)
            new_food_distance = math.dist(self.snake_pos[0],self.food_pos)

            if not self.game_over:
                if self.step - self.previous_score_step > self.max_tolerance_step*len(self.snake_pos) :
                    self.game_over = True
                elif new_food_distance < food_distance:
                    self.reward = 0.05
                        
            self.step+=1
            self.total_reward += self.reward

            self.draw_game()

            next_state=copy.deepcopy(self.dqn_agent.saved_state)
            next_state.pop(0)
            next_state.append(self.get_game_state())
            self.dqn_agent.add_experience(previous_state,self.action,self.reward,next_state,self.game_over)

            if self.dqn_agent.total_steps % 100 == 0 and len(self.dqn_agent.saved_experience) > CONFIG.experience_requirement:
                self.dqn_agent.update_q_network()
            if self.dqn_agent.total_steps % CONFIG.replace_target_iter == 0 and self.dqn_agent.total_steps != 0:
                self.dqn_agent.update_target_network()

            if CONFIG.save_model and self.dqn_agent.total_steps %CONFIG.save_model_frequency == 0 and self.dqn_agent.total_steps != 0:
                torch.save(self.dqn_agent.model.state_dict(),"~/temp/saved_model.pth")
                torch.save(self.dqn_agent.target_model.state_dict(),"~/temp/saved_target_model.pth")

            if self.game_over:
                self.reset_game()
            self.clock.tick(self.clock_tick)

if __name__ == '__main__':
    dqn_agent = DQN_AGENT()

    # dqn_agent.model.load_model_weights("~/temp/saved_model.pth")
    # dqn_agent.target_model.load_model_weights("~/temp/saved_target_model.pth")

    game = GmaeAgent(dqn_agent)
    game.agent_play_game()
            

            

            
