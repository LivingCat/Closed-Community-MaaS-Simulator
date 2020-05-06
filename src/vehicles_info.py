class Car:
	@staticmethod
	def	emissions(time):
		return time * 139

	@staticmethod
	def awareness():
		return 0.2

class Bus:
	@staticmethod
	def emissions(time):
		return time * 1250
	
	@staticmethod
	def awareness():
		return 0.9


class Bike:
	@staticmethod
	def emissions(time):
		return 0

	@staticmethod
	def awareness():
		return 1
