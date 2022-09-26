# Neural-Network-Implementation
 This is my implementation of neural network.

<h2>Features</h2>
You can easily add layers to network.<br/>
You can save or load your trained network.<br/>


<h3>Activation Functions:</h3>
Sigmoid<br/>
ReLu<br/>
Softmax<br/>

<h3>Regularization:</h3>
L2<br/>
Dropout (You can individually drop hidden layers)<br/>

<h3>Optimizers:</h3>
Momentum<br/>
RMS<br/>
Adam<br/>

<h3>Weight initialization: (Kernel Initializer)</h3>
He<br/>
Xavier<br/>
(You can choose kernel for individual layers)


<h1>Usage:</h1>

<h2>Adding Layer:</h2>

```python
add_layer(node_count, activation, input_len=0, kernel_initializer=None)
```

<b>node_count:</b> Layers node count.<br/>
<b>activation:</b> Activation function for this layer (Sigmoid, Relu, Softmax), you have to pass the function not the name.<br/>
<b>input_len:</b> Input lenght for this layer (Only use for first layer because network automatickly fills it for other layers).<br/>
<b>kernel_initializer:</b> Weight initialization (He, Xavier).<br/>



<h2>Training:</h2>

```python
train(alpha, iteration, L2=0, dropout=0, optimizer=None, Momentum_B=0.9, RMS_B=0.999)
```

<b>alpha:</b> Learning rate value for gradient descent<br/>
<b>iteration:</b> How many times we want to train the network with all the training data<br/>
<b>L2:</b> λ value for L2 regularization<br/>
<b>dropout:</b> If you want to use dropout you have to pass list of dropout probabilities for all layers(Example: [0, 0.2, 0.2])<br/>
<b>Momentum_B:</b> Beta value for momentum<br/>
<b>RMS_B:</b> Beta value for RMS<br/>

<h2>Saving and loading weights:</h2>

```python
save_network(file_name)
load_network(file_name)
```

<b>file_name:</b> Name for the file that we want to save out values<br/>


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
