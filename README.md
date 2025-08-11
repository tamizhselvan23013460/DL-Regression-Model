# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: TAMIZHSELVAN B

### Register Number: 212223230225

```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


torch.manual_seed(71)
X=torch.linspace(1,50,50).reshape(-1,1)
e=torch.randint(-8,9,(50,1),dtype=torch.float)
y=2*X+1+e


plt.scatter(X,y,color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()

class Model(nn.Module):
  def __init__(self,in_features,out_features):
    super().__init__()
    self.linear=nn.Linear(in_features,out_features)

  def forward(self,x):
    return self.linear(x)
torch.manual_seed(59)
model=Model(1,1)
initial_weight=model.linear.weight.item()
initial_bias=model.linear.bias.item()
print("\nName: Tamizhselvan B")
print("Register No: 212223230225")
print(f"Initial Weight: {initial_weight:.8f},Initial Bias: {initial_bias:.8f}\n")


loss_function=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
epochs=100
losses=[]
for epoch in range(1,epochs+1):
  optimizer.zero_grad()
  y_pred=model(X)
  loss=loss_function(y_pred,y)
  losses.append(loss.item())
  loss.backward()
  optimizer.step()
  print(f"epoch: {epoch:2} loss: {loss.item():10.8f}"
         f"weight: {model.linear.weight.item():10.8f}"
         f"bias: {model.linear.bias.item():10.8f}")


plt.plot(range(epochs),losses,color="Blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()


final_weight=model.linear.weight.item()
final_bias=model.linear.bias.item()
print("\nName : Tamizhselvan B")
print("Register No : 212223230225")
print(f"\nFinal Weight : {final_weight:.8f}, Final Bias : {final_bias:.8f}")


x1=torch.tensor([X.min().item(),X.max().item()])
y1=x1*final_weight+final_bias

plt.scatter(X,y,label="Original Data")
plt.plot(x1,y1,'r',label="Best=Fit line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()


x_new=torch.tensor([[120.0]])
y_new_pred=model(x_new).item()
print("\nName : Tamizhselvan B")
print("Register No : 212223230225")
print(f"\nPrediction for x = 120 : {y_new_pred:.8f}")

```

### OUTPUT

## Dataset Information

<img width="717" height="528" alt="image" src="https://github.com/user-attachments/assets/829d6d36-74f7-4a9c-b361-f691a9307921" />

## Initial weight & Bias:

<img width="522" height="120" alt="image" src="https://github.com/user-attachments/assets/6ca2691d-104a-4f31-b6b5-6fe148173add" />

## Training Loss Vs Iteration Plot

<img width="527" height="737" alt="image" src="https://github.com/user-attachments/assets/544c08f7-966c-4e53-9cd1-dbe5b6f98425" />
<img width="501" height="738" alt="image" src="https://github.com/user-attachments/assets/4338950d-be33-4be1-84c8-2bc04b1905d5" />

## Loss Curve:

<img width="572" height="398" alt="image" src="https://github.com/user-attachments/assets/1a46c422-91f7-4db2-ae7a-aaf6a6ad908a" />

## Final weight & Bias:

<img width="407" height="107" alt="image" src="https://github.com/user-attachments/assets/7f528eea-914f-4408-b68c-1401d22b774e" />


## Best Fit line plot

<img width="525" height="397" alt="image" src="https://github.com/user-attachments/assets/856297b9-d60a-4343-ba43-8d95ae281865" />


### New Sample Data Prediction

<img width="322" height="115" alt="image" src="https://github.com/user-attachments/assets/30e05187-8cfb-4095-91ce-ad8e8cf9e50f" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
