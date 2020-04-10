import torch
import torch.nn as nn
import numpy as np
import random
from opts import randomseed

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)

class hockey_classifier(nn.Module):
    def __init__(self):
        super(hockey_classifier, self).__init__()

        self.fc_switch = nn.Linear(4096, 10)
        self.fc_advance = nn.Linear(4096, 10)
        self.fc_faceoff = nn.Linear(4096, 10)
        self.fc_play_make = nn.Linear(4096, 10)
        self.fc_play_receive = nn.Linear(4096, 10)
        self.fc_whistle = nn.Linear(4096, 10)
        self.fc_shot = nn.Linear(4096, 10)
        self.fc_hit = nn.Linear(4096, 10)
        self.fc_shot_block = nn.Linear(4096, 10)
        self.fc_penalty = nn.Linear(4096, 10)
        self.fc_ricochet = nn.Linear(4096, 10)


    def forward(self, x):
        switch = self.fc_switch(x)
        advance = self.fc_advance(x)
        faceoff = self.fc_faceoff(x)
        play_make = self.fc_play_make(x)
        play_receive = self.fc_play_receive(x)
        whistle = self.fc_whistle(x)
        shot = self.fc_shot(x)
        hit = self.fc_hit(x)
        shot_block = self.fc_shot_block(x)
        penalty = self.fc_penalty(x)
        ricochet = self.fc_ricochet(x)

        return switch, advance, faceoff, play_make, play_receive, whistle, shot, hit, shot_block, penalty, ricochet