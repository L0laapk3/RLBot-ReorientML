from rlbot.utils.game_state_util import GameState, Vector3, CarState, Physics, Rotator
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from rlutilities.linear_algebra import euler_to_rotation, dot, transpose, look_at, vec2, vec3, norm, normalize, angle_between, orthogonalize
from rlutilities.simulation import Ball, Field, Game, Car
from rlutilities.mechanics import ReorientML

from policy import Policy
from simulation import Simulation
import math

import torch
import gc
from device import device
from random import random, seed

hidden_size = 32
hidden_size_2 = 32
model_name = f'2layer_{hidden_size}_{hidden_size_2}'
model = torch.load(model_name + '.mdl')
seed(0)

class TestAgent(BaseAgent):
	def __init__(self, name, team, index):
		self.name = name

		self.index = index

		self.policy = Policy(hidden_size, hidden_size_2).to(device)

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
		self.lastDoneTick = 0
		self.totalScore = 0
		self.tests = 0

		self.stage = 0


		

	def initialize_agent(self):
		self.game = Game()
		self.game.set_mode("soccar")
		self.car = self.game.cars[self.index]
		self.reorientML = ReorientML(self.car)



	game_state = None

	def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
		self.renderer.begin_rendering()
		
		self.game.read_game_information(packet, self.get_rigid_body_tick(), self.get_field_info())



		if self.lastReset + 300 < self.currentTick:
			if self.tests > 0:
				score = min(300, self.currentTick - self.lastDoneTick)
				self.totalScore += score
				print(self.tests, score, round(self.totalScore / self.tests, 2))
			self.tests += 1
			self.lastReset = self.currentTick
			self.target = vec3(2*random()-1, 2*random()-1, 2*random()-1)
			self.up = orthogonalize(vec3(2*random()-1, 2*random()-1, 2*random()-1), self.target)
			self.targetOrientation = look_at(self.target, self.up)
			car_state = CarState(physics=Physics(location=Vector3(0, 1000, 17), velocity=Vector3(0, 0, 0), rotation=Rotator(0, 0, 0), angular_velocity=Vector3(0, 0, 0)))
			self.set_game_state(GameState(cars={self.index: car_state}))
			self.stage = 0
			self.lastDodgeTick = -math.inf
			# print("TELEPORT TO GROUND")
			return self.controls
		else:
			car_state = CarState(physics=Physics(location=Vector3(0, 0, 400), velocity=Vector3(0, 0, 0)))
			self.set_game_state(GameState(cars={self.index: car_state}))


		if self.stage <= 5:
			self.stage += 1
		if self.stage == 6:
			self.dodgeDirection = normalize(vec2(0, 2*random()-1))
			self.controls.jump = True#random() > 0.5
			if self.controls.jump:
				self.lastDodgeTick = self.currentTick

			self.controls.roll, self.controls.pitch, self.controls.yaw = self.dodgeDirection[0], self.dodgeDirection[1], 0
			self.stage += 1
			return self.controls
		else:
			self.controls.jump = False

		self.packet = packet
		self.handleTime()

			
		car = packet.game_cars[self.index]
		position = vec3(car.physics.location.x, car.physics.location.y, car.physics.location.z)

		self.renderer.draw_line_3d(car.physics.location, position + 300 * normalize(self.target), self.renderer.yellow())
		self.renderer.draw_line_3d(car.physics.location, position + 300 * normalize(self.up), self.renderer.pink())

		carOrientation = rotationToOrientation(car.physics.rotation)
		ang = parseVector(car.physics.angular_velocity)
		

		if angle_between(carOrientation, self.targetOrientation) > 1 / 180 * math.pi:
			self.lastDoneTick = self.currentTick

		o_rlu = dot(transpose(self.targetOrientation), carOrientation)
		w_rlu = dot(transpose(self.targetOrientation), ang)
		o = torch.tensor([[o_rlu[i, j] for j in range(3)] for i in range(3)])[None, :].float().to(device)
		w = torch.tensor([w_rlu[i] for i in range(3)])[None, :].float().to(device)

		noPitchTime = max(0, (self.lastDodgeTick - self.currentTick)/120 + .95)
		dodgeTime = max(0, (self.lastDodgeTick - self.currentTick)/120 + .65)
		if dodgeTime == 0:
			self.dodgeDirection = vec2(0, 0)
		
		noPitchTime = torch.tensor([noPitchTime]).float().to(device)
		dodgeTime = torch.tensor([dodgeTime]).float().to(device)
		dodgeDirection = torch.tensor([self.dodgeDirection[i] for i in range(2)])[None, :].float().to(device)

		
		# if self.simulation.o is not None and self.simulation.w is not None:
		# 	print("=====================================")
		# 	print("-------------------------------------")
		# 	print(self.simulation.o, o)
		# 	print(self.simulation.w, w)
		# 	print(self.simulation.noPitchTime, noPitchTime)
		# 	print(self.simulation.dodgeTime, dodgeTime)
		# 	print(self.simulation.dodgeDirection, dodgeDirection)
		# 	self.simulation.step(self.ticksNowPassed / 120)
		# 	print(self.simulation.o, o)
		# 	print(self.simulation.w, w)
		# 	print(self.simulation.noPitchTime, noPitchTime)
		# 	print(self.simulation.dodgeTime, dodgeTime)
		# 	print(self.simulation.dodgeDirection, dodgeDirection)



		self.simulation.o = o
		self.simulation.w = w
		self.simulation.noPitchTime = noPitchTime
		self.simulation.dodgeTime = dodgeTime
		self.simulation.dodgeDirection = dodgeDirection



		if True:

			rpy = self.policy(
				self.simulation.o.permute(0, 2, 1),
				self.simulation.w_local(),
				self.simulation.noPitchTime,
				self.simulation.dodgeTime,
				self.simulation.dodgeDirection
			)[0]
			self.controls.roll, self.controls.pitch, self.controls.yaw = rpy
		
		else:

			self.reorientML.target_orientation = self.targetOrientation
			self.reorientML.step(1/self.FPS)
			self.controls.roll, self.controls.pitch, self.controls.yaw = self.reorientML.controls.roll, self.reorientML.controls.pitch, self.reorientML.controls.yaw

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
	