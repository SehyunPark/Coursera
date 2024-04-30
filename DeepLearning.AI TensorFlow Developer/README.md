# DeepLearning.AI TensorFlow Developer
- A summary of what 've learned from this lecture

## C1 - Introduction to TensorFlow for AI, ML & DL

### Week 1 - A New Programming Paradigm

- ML: we can get a buch of examples for what we want to see and then have the computer figure out the rules

- example code

```
import keras
import numpy as np
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
#Sequential: succesive layers are defined in sequence
#Dense: defining a layer of connected neurons
#units=1 : single neuron
#input: one value(input_shape=[1])

model.compile(optimizer='sgd', loss='mean_squared_error') 
#loss function measures this X/Y given data
#optimizer thinks how good/bad this guess was done using the data from the loss function
#as guess get better and better (approahces to 100%) the convergence is used

xs=np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys=np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500) #training
#go through the training loop 500 times: repeating 'measuring loss - optimizer - make better guess' 

print(model.predict([10.0])) #[[18.981611]] not exactly 19
```

- Keyword: ML, Sequential Class, Dense layer, input_shape, loss, optimizer, fit(), epochs
  - loss: measures the guessed answers against the known correct answers and measures how well or how badly it did
  - optimizer: determines how the model's parameters should be adjusted to minimize the loss, guiding the model to make another guess or update its parameters to improve its predictions

----

### Week 2 - Introduction to Computer Vision

- CV: the the field of having a computer understand and label what is present in an image
  - ex) FashionMNIST: the images are in gray scale(to redue the amount of info). With 28 x 28 pixles in an image, only 784 bytes are needed to store the entire image (check this [blog post](https://sh-avid-learner.tistory.com/322) for more info)
    - splitting into traning set / testing set
    - normalization
    - building a classification model: flattening - dense layer(activation: relu) - dense layer(activation: softmax)
    - compiling & fitting
    - tuning(more or less neurons & epochs / removing Flatten layer & differing the number of ouput nodes / adding another layer / without normalization)
    - accuracy analysis
      <br>
      <img src="https://github.com/SehyunPark/Coursera/assets/28240330/c965fb18-d464-4194-bfba-d04cfff92dba" width="600" height="300">
      
----

### Week 3 - Enhancing Vision with CNN

- 


