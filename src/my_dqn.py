import torch
import torch.nn as nn
from collections import deque
import random
from math import exp
from config import CONFIG

device=torch.device("cuda" if CONFIG.use_gpu and torch.cuda.is_available() else "cpu")
print(f"using device:{device}")

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=4, kernel_size=1, stride=1,padding=0)
        self.fc1 = nn.Linear(4*17*17,128)
        self.fc2=nn.Linear(128,4)
    def load_model_weights(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print('Weights loaded')
    def forward(self, x ,batch_size=1):
        x = torch.relu(self.conv1(x))
        #print(x.shape)
        x = torch.relu(self.conv2(x))
        #print(x.shape)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = torch.relu(self.fc1(x))
        #print(x.shape)
        x=self.fc2(x)
        return x
class DQN_AGENT():
    def __init__(self,epsilon=CONFIG.initial_epsilon):
        self.model=DQN().to(device)
        self.target_model=DQN().to(device)
        self.total_steps=0
        self.total_circles=0
        self.learning_rate=CONFIG.learning_rate
        self.saved_state=[torch.zeros((21,21),device=device) for _ in range(4)]
        self.saved_experience=deque(maxlen=CONFIG.buffer_size)
        self.epsilon=epsilon
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    def select_next_action(self,game_state,epsilon_greedy=True):
        self.saved_state.pop(0)
        self.saved_state.append(torch.tensor(game_state).to(device))
        state=torch.stack(self.saved_state,dim=0)

        self.total_steps += 1
        self.epsilon=max(CONFIG.final_epsilon,self.epsilon)
        self.epsilon -= (CONFIG.initial_epsilon-CONFIG.final_epsilon)/CONFIG.explore
        if epsilon_greedy and random.random() < self.epsilon:
            action=random.randint(1,4)
        else:
            with torch.no_grad():
                q_values=self.model(state.unsqueeze(0))
                action=torch.argmax(q_values).item()+1
        return action
    def add_experience(self,state,action,reward,next_state,is_over):
        state_tensor=torch.stack(state,dim=0).to(dtype=torch.float32,device=device)
        next_state_tensor=torch.stack(next_state,dim=0).to(dtype=torch.float32,device=device)
        experience=(state_tensor,action,reward,next_state_tensor,is_over)
        self.saved_experience.append(experience)
    def update_q_network(self,batch_size=CONFIG.batch_size,gamma=CONFIG.gamma):
        if len(self.saved_experience) < batch_size:
            return
        batch=random.sample(self.saved_experience,batch_size)

        states=torch.stack([exp[0] for exp in batch],dim=0).to(dtype=torch.float32,device=device)
        actions=torch.tensor([exp[1] for exp in batch],dtype=torch.int,device=device)
        rewards=torch.tensor([exp[2] for exp in batch],dtype=torch.float32,device=device)
        next_states=torch.stack([exp[3] for exp in batch]).to(dtype=torch.float32,device=device)
        dones=torch.tensor([exp[4] for exp in batch],dtype=torch.bool,device=device)

        q_values=self.model(states)
        next_q_values=self.target_model(next_states)
        
        expected_q_values = q_values.clone()
        max_next_q_values = torch.max(next_q_values, dim=1).values
        target = rewards + gamma * max_next_q_values * (~dones)
        
        expected_q_values[torch.arange(len(batch)), actions - 1] = target

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
    def DDQN_update_q_network(self,batch_size=CONFIG.batch_size,gamma=CONFIG.gamma):
        #DDQN算法的尝试，训练过程确实比DQN要平稳
        if len(self.saved_experience) < batch_size:
            return
        batch=random.sample(self.saved_experience,batch_size)

        states=torch.stack([exp[0] for exp in batch],dim=0).to(dtype=torch.float32,device=device)
        actions=torch.tensor([exp[1] for exp in batch],dtype=torch.int,device=device)
        rewards=torch.tensor([exp[2] for exp in batch],dtype=torch.float32,device=device)
        next_states=torch.stack([exp[3] for exp in batch]).to(dtype=torch.float32,device=device)
        dones=torch.tensor([exp[4] for exp in batch],dtype=torch.bool,device=device)

        q_values=self.model(states)
        
        max_q_value_action = torch.argmax(self.model(next_states),dim=1)
        next_q_values = self.target_model(next_states)
        target_next_q_values = next_q_values[torch.arange(len(batch)),max_q_value_action]

        expected_q_values = q_values.clone()
        target = rewards + gamma * target_next_q_values * (~dones)
        expected_q_values[torch.arange(len(batch)), actions - 1] = target

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
            self.target_model.load_state_dict(self.model.state_dict())



