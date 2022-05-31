import numpy as np

class train():

    def __init__(self, model):
        # Weights are numpy arrays and that list contains that np arrays.
        self.model = model
        # Epsilon value (for preventing devide by zero or log0)
        self.e = 10 ** -8

    # Loading training data into the ai.
    def load_data(self, datas, classes):
        self.X = datas
        self.Y = classes
        self.dataset_len = self.Y.shape[1]

    def forward_propocation(self, weights, biases, activations, input_data, dropout):
        # A: activation function.
        # Z: inside of every activation function.
        # D: matrix.
        A, Z, D = [], [], []
        # Our activaion functions result, Initialy its input data.
        A.append(input_data)
        # Iterating over all layers and calulating each of them according to previus ones.
        for weight_vector, bias_vector, activation, drop_prob in zip(weights, biases, activations, dropout):
            Z.append(weight_vector @ A[-1] + bias_vector)
            A.append(activation(Z[-1]))
            # By probability drop_prob we zeroing nodes outputs (droping them
            # If drop_prob = 0 we are not calculating it will automatickly be no dropout state.
            if drop_prob > 0:
                # If any layer has a dropout we adding dropout matrix for that layer into the 
                # list D. If we don't have any dropout layer list D will be empty.
                D.append(np.random.rand(A[-1].shape[0], A[-1].shape[1]) > drop_prob)
                A[-1] = A[-1] *  D[-1] / (1 - drop_prob)
        return A, Z, D

    def calculate_derivatives(self, weights, biases, activations, input_data, output_data, L2, dropout):
        m = self.dataset_len
        L = len(weights)
        dweights, dbiases = [0] * (self.model.L), [0] * (self.model.L)
        A, Z, D = self.forward_propocation(weights, biases, activations, input_data, dropout)
        dz = 1/m * (A[L] - output_data)
        for i, drop_prob in zip(reversed(range(0, self.model.L)), reversed(dropout)):
            # If we are on last layer dz will be above one so we skipping for L.
            if i != self.model.L - 1: dz = 1/m * (np.transpose(weights[i + 1]) @ dz) * activations[i](Z[i], derivative=1)
            # if we using dropout reguralization, if dropout probability for tahat 
            # layer is non zero we calculating its droped out derivitive.
            if drop_prob > 0: dz *= D.pop() / (1 - drop_prob)
            dweights[i] = dz @ np.transpose(A[i])
            # If we using L2 regularization we adding that to dweights.
            if L2: dweights[i] += (L2/m) * weights[i]
            dbiases[i] = np.sum(dz, axis=1, keepdims=True)
        return dbiases, dweights

    def calculate_cost(self, input_data, output_data, L2):
        hypthoses = self.model.predict(input_data, one_hot=1)
        m = self.dataset_len
        cost = (1/m) * np.sum(-output_data * np.log(hypthoses + self.e) - (1 - output_data) * np.log(1 - hypthoses + self.e))
        # If we using L2 regularization we adding cost of weights.
        if L2:
            for weight in self.model.weights:
                cost +=  L2/(2*m) * np.sum(np.square(weight))
        return cost

    def minimize(self, gweights, gbiases, alpha):
        for i in reversed(range(0, self.model.L)):
            self.model.weights[i] = self.model.weights[i] - alpha * gweights[i]
            self.model.biases[i] = self.model.biases[i] - alpha * gbiases[i]

    def Momentum(self, dweights, dbiases, vdweights, vdbiases, Momentum_B):
        for dweight, dbias, a in zip(dweights, dbiases, range(self.model.L)):
            vdweights[a] = Momentum_B * vdweights[a] + (1 - Momentum_B) * dweight
            vdbiases[a] = Momentum_B * vdbiases[a] + (1 - Momentum_B) * dbias
        return vdweights, vdbiases

    def RMS(self, dweights, dbiases, sdweights, sdbiases, RMS_B, vdweights=0, vdbiases=0, Adam=0):
        # Calclating RMS or Adams output
        for dweight, dbias, a in zip(dweights, dbiases, range(self.model.L)):
            sdweights[a] = RMS_B * sdweights[a] + (1 - RMS_B) * np.square(dweight)
            sdbiases[a] = RMS_B * sdbiases[a] + (1 - RMS_B) * np.square(dbias)
            if Adam:
                # At this point sdweights name is not representing what we doing
                # that new sdweights is actually result of Adam
                sdweights[a] = vdweights[a] / (np.sqrt(np.absolute(sdweights[a]) + self.e))
                sdbiases[a] = vdbiases[a] / (np.sqrt(np.absolute(sdbiases[a]) + self.e))
            else:
                sdweights[a] = dweight / (np.sqrt(np.absolute(sdweights[a]) + self.e))
                sdbiases[a] = dbias / (np.sqrt(np.absolute(sdbiases[a]) + self.e))
        return sdweights, sdbiases

    def train(self, alpha, iteration, L2=0, dropout=0, optimizer=None, Momentum_B=0.9, RMS_B=0.999):
        # If we use dropout,
        # we don't want to drop out last layer so we making it's dropout probability 0.
        if dropout: dropout.append(0)
        # If we don't use any dropout, making all dropout probabilities 0.
        else: dropout = [0] * (self.model.L)

        # If we want to use momentum optimizer initilizing vdb and vdb to zero first.
        if optimizer == "Adam" or "Momentum":
            vdweights, vdbiases = [], []
            for weights, biases in zip(self.model.weights, self.model.biases):
                vdweights.append(np.zeros_like(weights))
                vdbiases.append(np.zeros_like(biases))

        if optimizer == "Adam" or "RMS":
            sdweights, sdbiases = [] , []
            for weights, biases in zip(self.model.weights, self.model.biases):
                sdweights.append(np.zeros_like(weights))
                sdbiases.append(np.zeros_like(biases))

        # Gradient descent.
        for i in range(iteration):
            # Printing the 10 times while iterating.
            if i % (iteration / 10) == 0:
                print(self.calculate_cost(self.X, self.Y, L2))
            dbiases, dweights = self.calculate_derivatives(self.model.weights, self.model.biases, self.model.activations, self.X, self.Y, L2, dropout)
            # If we use momentum optimizer minimizing according to vdw and vdb.
            if optimizer == "Momentum":
                vdweights, vdbiases = self.Momentum(dweights, dbiases, vdweights, vdbiases,  Momentum_B)
                self.minimize(vdweights, vdbiases, alpha)
            if optimizer == "RMS":
                sdweights, sdbiases = self.RMS(dweights, dbiases, sdweights, sdbiases, RMS_B)
                self.minimize(sdweights, sdbiases, alpha)
            if optimizer == "Adam":
                vdweights, vdbiases = self.Momentum(dweights, dbiases, vdweights, vdbiases,  Momentum_B)
                a, b = self.RMS(dweights, dbiases, sdweights, sdbiases, RMS_B, vdweights, vdbiases, Adam=1)
                self.minimize(a, b, alpha)
            # If we not using any optimizer minimazing according to normal derivitives.
            if optimizer == None:
                self.minimize(dweights, dbiases, alpha)