class Car:
	speed = 50
	car_em = 139

	@staticmethod
	def	emissions(time):
		return time * (Car.car_em * Car.speed)

	@staticmethod
	def awareness():
		return 0.2

	@staticmethod
	def credits():
		return 0

class Bus:

	speed = 50
	pass_em = 1250/20.0

	@staticmethod
	def emissions(time):
		return time * (Bus.pass_em * Bus.speed)
	
	@staticmethod
	def awareness():
		return 0.9

	@staticmethod
	def credits():
		return 3


class Bike:

	speed = 20
	
	@staticmethod
	def emissions(time):
		return 0

	@staticmethod
	def awareness():
		return 1

	@staticmethod
	def credits():
		return 5


class Walk:

	speed = 5

	@staticmethod
	def emissions(time):
		return 0

	@staticmethod
	def awareness():
		return 1

	@staticmethod
	def credits():
		return 5
