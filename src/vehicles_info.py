class Car:
	@staticmethod
	def	emissions(distance):
		return distance * 139

	@staticmethod
	def awareness():
		return 0.2

class Bus:
	@staticmethod
	def emissions(distance):
		return distance * 81.63
	
	@staticmethod
	def awareness():
		return 1
