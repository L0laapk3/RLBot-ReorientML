import sys, os
from pathlib import Path
import msvcrt
import math

import torch
import gc
from torch.optim.adadelta import Adadelta
from quicktracer import trace
from device import device

delta_time = 1 / 120
steps = int(round(1.8 / delta_time))
hidden_size = 32
hidden_size_2 = 32
load = True
rotation_eps = 1 / 180 * math.pi
model_name = f'2layer_{hidden_size}_{hidden_size_2}'


class Trainer:
    def __init__(self):
        global load
        from policy import Policy
        from simulation import Simulation
        from optimizer import Yeet, andt

        self.policy = Policy(hidden_size, hidden_size_2).to(device)
        self.simulation = Simulation(self.policy)
        self.optimizer = Yeet(self.policy.parameters())
        # self.optimizer = Adadelta(self.policy.parameters())
        self.andt = andt

        self.max_reward = 0

        if load and not os.path.exists(f"{model_name}.state"):
            print("not loading cuz it doesnt exist")
            load = False

        self.reachesEnd = load

        if load:
            self.policy.load_state_dict(torch.load(model_name + '.mdl'), False)
            self.optimizer.load_state_dict(torch.load(model_name + '.state'))

        for group in self.optimizer.param_groups:
            group['rho'] = 0.5
            group['lr'] = 0.0002

    def train(self):
        while not msvcrt.kbhit():
            self.episode()

        torch.save(self.policy.state_dict(), model_name + '.mdl')
        torch.save(self.optimizer.state_dict(), model_name + '.state')

    def episode(self):
        self.simulation.random_state()

        reward = torch.zeros((self.simulation.o.shape[0],), device=device)
        framesDone = torch.zeros((self.simulation.o.shape[0],), device=device)

        # profile()
        # sys.exit()

        for i in range(steps):
            self.simulation.step(delta_time)
            diff = rotation_eps - self.simulation.error()
            # reward *= 0.8
            # reward += diff.clamp(max=0)

            reward += diff.clamp(max=0, min=-rotation_eps/2 if self.reachesEnd else None)

            finished = (diff > 0).float()
            # reward = diff.clamp(max=0)
            framesDone += 1
            framesDone *= finished
            # reward = finished * (reward + 1)
            # if i == steps-1:
            #     framesDone = reward.clone().detach()
            #     reward += diff.clamp(max=0) / rotation_eps
        # reward = framesDone


        trace(((steps - framesDone) * delta_time * 120).mean(0).item(), reset_on_parent_change=False, key='game frames to destination')
        failed = (framesDone == 0).float().mean(0).item()
        self.reachesEnd = failed < 0.2
        trace(failed, reset_on_parent_change=False, key='amount failed')

        # reward[:, steps - 1] = self.andt(reward[:, steps - 1])
        # for i in reversed(range(steps - 1)):
        #     reward[:, i] = self.andt(reward[:, i], reward[:, i+1])

        loss = reward.mean(0).neg()

        # average_reward = sum(reward[:, steps - 1]) / len(reward[:, steps - 1])
        # if average_reward.item() > self.max_reward:
        #     self.max_reward = average_reward.item()
        #     torch.save(self.policy.state_dict(), f'out/{model_name}_{round(self.max_reward, 1)}.mdl')
        #     torch.save(self.optimizer.state_dict(), f'out/{model_name}_{round(self.max_reward, 1)}.state')


        self.optimizer.zero_grad()
        loss.backward() # spits out error
        self.optimizer.step()
        trace(loss.item(), reset_on_parent_change=False, key='loss')


        # trace((reward < 0).float().mean(0).item(), reset_on_parent_change=False, key='frame weight')





def profile():
    print("----------------------------------------")
    print("PROFILING")
    print("----------------------------------------")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass





if __name__ == '__main__':
    current_path = Path(__file__).absolute().parent
    sys.path.insert(0, str(current_path.parent.parent))  # this is for first process imports

    torch.autograd.set_detect_anomaly(True)

    trainer = Trainer()
    trainer.train()




