# Neural-Network-Implementation
 This is my implementation of neural network.

Features<br/>
You can easily add layers to network.<br/>
You can save or load your trained network.<br/>


Activation Functions:<br/>
Sigmoid<br/>
ReLu<br/>
Softmax<br/>

Regularization:<br/>
L2<br/>
Dropout (You can individually drop hidden layers)<br/>

Optimizers:<br/>
Momentum<br/>
RMS<br/>
Adam<br/>

Weight initialization: (Kernel Initializer)<br/>
He<br/>
Xavier<br/>
(You can choose kernel for individual layers)

Adding Layer:
node_count: Layers node count.
activation: Activation function for this layer (Sigmoid, Relu, Softmax), you have to pass the function not the name.
input_len: Input lenght for this layer (Only use for first layer because network automatickly fills it for other layers).
kernel_initializer: Weight initialization (He, Xavier).

```python
add_layer(node_count, activation, input_len=0, kernel_initializer=None)
```

Training:
alpha: Learning rate value for gradient descent
iteration: How many times we want to train the network with all the training data
L2: λ value for L2 regularization
dropout: If you want to use dropout you have to pass list of dropout probabilities for all layers(Example: [0, 0.2, 0.2])
Momentum_B: Beta value for momentum
RMS_B: Beta value for RMS

```python
train(alpha, iteration, L2=0, dropout=0, optimizer=None, Momentum_B=0.9, RMS_B=0.999)
```

Saving and loading weights:
file_name: Name for the file that we want to save out values
```python
save_network(file_name)

load_network(file_name)
```

Example Usage:
```python
model = network()
model.add_layer(120, model.ReLu, input_len=81, kernel_initializer="He")
model.add_layer(120, model.ReLu, kernel_initializer="He")
model.add_layer(81, model.ReLu, kernel_initializer="He")
trainer = train(model)
trainer.load_data(x, y)
trainer.train(0.001, 10000, optimizer="Adam")
model.save_network("test")
```
