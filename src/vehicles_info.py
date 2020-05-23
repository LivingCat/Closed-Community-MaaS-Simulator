class Car:
	speed = 50

	@staticmethod
	def	emissions(time):
		return time * (139 * Car.speed)

	@staticmethod
	def awareness():
		return 0.2

	@staticmethod
	def credits():
		return 0

class Bus:

	speed = 50

	@staticmethod
	def emissions(time):
		return time * (1250 * Bus.speed)
	
	@staticmethod
	def awareness():
		return 0.9

	@staticmethod
	def credits():
		return 2


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
		return 4


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
		return 4
