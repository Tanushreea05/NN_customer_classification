# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model


![NEURAL MODEL](https://github.com/user-attachments/assets/1e57d1d9-f50a-4e69-9820-7eb3d06a5cbc)

## DESIGN STEPS

### STEP 1:
Data Preprocessing: Clean, normalize, and split data into training, validation, and test sets.

### STEP 2:
Model Design:

Input Layer: Number of neurons = features.
Hidden Layers: 2 layers with ReLU activation.
Output Layer: 4 neurons (segments A, B, C, D) with softmax activation.
### STEP 3:
Model Compilation: Use categorical crossentropy loss, Adam optimizer, and track accuracy.

### STEP 4:
Training: Train with early stopping, batch size (e.g., 32), and suitable epochs.

### STEP 5:
Model Compilation: Use categorical crossentropy loss, Adam optimizer, and track accuracy.

### STEP 6:
Training: Train with early stopping, batch size (e.g., 32), and suitable epochs.





## PROGRAM

### Name: Tanushree
### Register Number:212223100057

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)

    def forward(self,x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x
        

```
```python
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

def train_model(model, train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      output = model(X_batch)
      loss = criterion(output,y_batch)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      output = model(X_batch)
      loss = criterion(output,y_batch)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
```



## Dataset Information

![Screenshot 2025-03-17 051507](https://github.com/user-attachments/assets/c0566034-a1f5-43e7-995a-de4459d88e81)


## OUTPUT



### Confusion Matrix
![Screenshot 2025-03-17 051610](https://github.com/user-attachments/assets/d703b720-a3ab-4118-b3cc-41579a9954bd)



### Classification Report

![Screenshot 2025-03-17 051642](https://github.com/user-attachments/assets/19c60ceb-9991-4391-8eb2-6df3c6a92da7)



### New Sample Data Prediction

![Screenshot 2025-03-17 051714](https://github.com/user-attachments/assets/958a1de6-26a5-4e2b-9e21-76411923d255)


## RESULT
So, To develop a neural network classification model for the given dataset is executed successfully.
