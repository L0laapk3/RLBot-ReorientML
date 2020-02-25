from rlbot.utils.game_state_util import GameState, Vector3
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from rlutilities.linear_algebra import euler_to_rotation, dot, transpose, look_at, vec3, norm, normalize, angle_between, orthogonalize

from policy import Policy
from simulation import Simulation
import math

import torch
from device import device
from random import random

hidden_size = 40
model_name = f'full_rotation_{hidden_size}_yeet_0.01'
model = torch.load(model_name + '.mdl')

class TestAgent(BaseAgent):
	def __init__(self, name, team, index):
		self.name = name

		self.index = index

		self.policy = Policy(hidden_size).to(device)

		self.policy.load_state_dict(model)
		self.simulation = Simulation(self.policy)
		self.controls = SimpleControllerState()
		self.finished = False

		self.FPS = 120
		
		self.lastTime = 0
		self.realLastTime = 0
		self.currentTick = 0
		self.skippedTicks = 0
		self.doneTicks = 0
		self.ticksNowPassed = 0

		self.lastDodgeTick = -math.inf
		self.lastDodgePitch = 0
		self.lastDodgeRoll = 0

		self.lastReset = 0
		self.target = vec3(1, 0, 0)
		self.up = vec3(0, 0, 1)
		self.targetOrientation = look_at(self.target, self.up)


	game_state = None

	def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
		self.renderer.begin_rendering()


		if self.lastReset + 300 < self.currentTick:
			self.lastReset = self.currentTick
			self.target = vec3(2*random()-1, 2*random()-1, 2*random()-1)
			self.up = orthogonalize(vec3(2*random()-1, 2*random()-1, 2*random()-1), self.target)
			self.targetOrientation = look_at(self.target, self.up)

		self.packet = packet
		self.handleTime()

		self.game_state = GameState.create_from_gametickpacket(packet)
		self.game_state.cars[self.index].jumped = None
		self.game_state.cars[self.index].double_jumped = None
		self.game_state.cars[self.index].physics.location = Vector3(0, 1000, 1000)
		self.game_state.cars[self.index].physics.rotation = None
		self.game_state.cars[self.index].physics.angular_velocity = None
		self.game_state.cars[self.index].physics.velocity = Vector3(0, 0, 0)
		self.game_state.ball.physics.velocity.z = 10
		self.set_game_state(self.game_state)
			
		car = packet.game_cars[self.index]
		position = vec3(car.physics.location.x, car.physics.location.y, car.physics.location.z)

		self.renderer.draw_line_3d(car.physics.location, position + 300 * normalize(self.target), self.renderer.yellow())
		self.renderer.draw_line_3d(car.physics.location, position + 300 * normalize(self.up), self.renderer.pink())

		carOrientation = rotationToOrientation(car.physics.rotation)
		ang = parseVector(car.physics.angular_velocity)
		

		print(angle_between(carOrientation, self.targetOrientation))

		o_rlu = dot(transpose(self.targetOrientation), carOrientation)
		w_rlu = dot(transpose(self.targetOrientation), ang)
		o = torch.tensor([[o_rlu[i, j] for j in range(3)] for i in range(3)])[None, :].to(device)
		w = torch.tensor([w_rlu[i] for i in range(3)])[None, :].to(device)

		# if self.simulation.o is not None and self.simulation.w is not None:
		#     self.simulation.step(info.time_delta)
		#     print(self.simulation.o - o)
		#     print(self.simulation.w - w)

		noPitchTime = max(0, self.lastDodgeTick + .95 - self.currentTick)
		dodgeTime = max(0, self.lastDodgeTick + .65 - self.currentTick)
		if dodgeTime == 0:
			self.lastDodgePitch = 0
			self.lastDodgeRoll = 0


		self.simulation.o = o
		self.simulation.w = w
		self.simulation.noPitchTime = torch.tensor([noPitchTime]).float().to(device)
		self.simulation.dodgeTime = torch.tensor([dodgeTime]).float().to(device)
		self.simulation.dodgeDirection = torch.tensor([(self.lastDodgeRoll, self.lastDodgePitch)]).float().to(device)

		rpy = self.policy(
			self.simulation.o.permute(0, 2, 1),
			self.simulation.w_local(),
			self.simulation.noPitchTime,
			self.simulation.dodgeTime,
			self.simulation.dodgeDirection
		)[0]

		print(rpy)

		self.controls.roll, self.controls.pitch, self.controls.yaw = rpy

		if self.simulation.error()[0].item() < 0.01:
			self.frames_done += 1
		else:
			self.frames_done = 0

		if self.frames_done >= 10:
			self.finished = True



		self.renderer.end_rendering()
		return self.controls


	def get_mechanic_controls(self):
		return self.mechanic.step(self.info)






	def handleTime(self):
		# this is the most conservative possible approach, but it could lead to having a "backlog" of ticks if seconds_elapsed
		# isnt perfectly accurate.
		if not self.lastTime:
			self.lastTime = self.packet.game_info.seconds_elapsed
		else:
			if self.realLastTime == self.packet.game_info.seconds_elapsed:
				return

			if int(self.lastTime) != int(self.packet.game_info.seconds_elapsed):
				if self.skippedTicks > 0:
					print(f"did {self.doneTicks}, skipped {self.skippedTicks}")
				self.skippedTicks = self.doneTicks = 0

			self.ticksNowPassed = round(max(1, (self.packet.game_info.seconds_elapsed - self.lastTime) * self.FPS))
			self.lastTime = min(self.packet.game_info.seconds_elapsed, self.lastTime + self.ticksNowPassed)
			self.realLastTime = self.packet.game_info.seconds_elapsed
			self.currentTick += self.ticksNowPassed
			if self.ticksNowPassed > 1:
				#print(f"Skipped {ticksPassed - 1} ticks!")
				self.skippedTicks += self.ticksNowPassed - 1
			self.doneTicks += 1





def parseVector(u):
	return vec3(u.x, u.y, u.z)

def rotationToOrientation(rotation):
	return euler_to_rotation(vec3(
		rotation.pitch,
		rotation.yaw,
		rotation.roll 
	))
	