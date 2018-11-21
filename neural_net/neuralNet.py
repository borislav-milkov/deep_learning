from numpy import exp, array, random, dot

class NeuralNetwork():

	def __init__(self):
		random.seed(1)
		self.synaptic_weights = 2 * random.random((3,1)) - 1

	def __sigmoid(self, x):
		return 1/(1 + exp(-x))

	def train(self, trainX, trainY, epochs):
		for iteration in range(epochs):
			output = self.predict(trainX)
			error = trainY - output
			adjustment = dot(trainX.T, error * self.__sigmoid_derivative(output))

			self.synaptic_weights += adjustment

	def __sigmoid_derivative(self, x):
		return x * ( 1 - x)

	def predict(self, inputs):
		return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == '__main__':
	neural_net = NeuralNetwork()

	print ('Random starting weights: ')
	print (neural_net.synaptic_weights)


	training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
	training_set_outputs = array([[0,1,1,0]]).T

	neural_net.train(training_set_inputs, training_set_outputs, 10000)

	print ('New weights, after training: ')
	print (neural_net.synaptic_weights)

	print('predicting [1,0,0]:')
	print(neural_net.predict(array([1,0,0])))