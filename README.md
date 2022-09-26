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


<h1>Usage:</h1>

<h2>Adding Layer:</h2>

```python
add_layer(node_count, activation, input_len=0, kernel_initializer=None)
```

<h5>node_count:</h5> Layers node count.<br/>
activation: Activation function for this layer (Sigmoid, Relu, Softmax), you have to pass the function not the name.<br/>
input_len: Input lenght for this layer (Only use for first layer because network automatickly fills it for other layers).<br/>
kernel_initializer: Weight initialization (He, Xavier).<br/>



<h2>Training:</h2>

```python
train(alpha, iteration, L2=0, dropout=0, optimizer=None, Momentum_B=0.9, RMS_B=0.999)
```

alpha: Learning rate value for gradient descent<br/>
iteration: How many times we want to train the network with all the training data<br/>
L2: λ value for L2 regularization<br/>
dropout: If you want to use dropout you have to pass list of dropout probabilities for all layers(Example: [0, 0.2, 0.2])<br/>
Momentum_B: Beta value for momentum<br/>
RMS_B: Beta value for RMS<br/>

<h2>Saving and loading weights:</h2>

```python
save_network(file_name)
load_network(file_name)
```

file_name: Name for the file that we want to save out values<br/>


<h2>Example Usage:</h2>

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
