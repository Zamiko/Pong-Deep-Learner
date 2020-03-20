from collections import deque
import collections
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x): #Forward pass to the NN
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon: #YT: To deteriine if the angent will choose exploreation/explotatin at each time step
                                        # we genrate a random number between 0 and 1. If this number is greater than epsilon, 
                                        # the agent will chose its next action via explotation (highest value in the Qtable)
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            # TODO: Given state, you should write code to get the Q value and chosen action
            qvalues = self.forward(state.data)
            np_qvalues = qvalues.detach().cpu().numpy()
            action = np.argmax(np_qvalues)

        else:
            action = random.randrange(self.env.action_space.n)

        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

        
def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    
    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    # implement the loss function here:   Lossi(Θi) = (yi − Q(s, a; Θi))^2
    #target_y = target_model.forward(next_state) #DO I need to get the max q value??
    #predict_y = mode.forward(state)
    #yi_qvalues =  model.forward(state)
    
    np_state = np.squeeze(state.detach().cpu().numpy())
    np_state = np.expand_dims(np_state, axis = 1)
    np_state = Variable(torch.FloatTensor(np_state))
    np_action = action.detach().cpu().numpy()
    np_reward = reward.detach().cpu().numpy()
    np_done = done.detach().cpu().numpy()
    
    q = target_model.forward(next_state).detach().cpu().numpy()
    y = model.forward(np_state).detach().cpu().numpy()

    Yi = np.zeros((batch_size, 1)) 
    Q = np.zeros((batch_size, 1))
    for a in range(0, batch_size):
        Yi[a] = y[a][np_action[a]]
        if np_done[a] == 1:
            Q[a] = np_reward[a]
        
        else:
            Q[a] = gamma * q[a][np.argmax(q[a])] + np_reward[a]
        
    loss = np.power(Yi - Q,2) 
    loss = np.sum(loss)
    loss = Variable(torch.FloatTensor([loss]), requires_grad=True)    
    #for i in range(0, batch_size):
     #   y = model.forward(torch.LongTensor(np_state[i]))
      #  print(y)
    # predicted_Q_vals = model.forward(state)
    # next_Q_vals = target_model.forward(next_state.data)
    # target_Q_vals =(next_Q_vals * gamma) + reward
    # loss = (target_Q_vals - predicted_Q_vals).square()

    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # TODO: Randomly sampling data with specific batch size from the buffer
        sampling = random.sample(self.buffer, batch_size)
        Experience = collections.namedtuple(
            'Experience',
            ('state', 'action', 'reward', 'next_state', 'done')
        )
        batch = Experience(*zip(*sampling))
        #print(batch)
        state = batch.state
        action = batch.action
        reward = batch.reward
        next_state = batch.next_state
        done = batch.done
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
