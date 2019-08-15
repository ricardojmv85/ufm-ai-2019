#!/usr/bin/env python
# coding: utf-8

# In[365]:


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ## Incisco 1

# In[387]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] # Temperatura en grados celsios
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] # Unidades desconocidas
t_c = torch.tensor(t_c).unsqueeze(1) # Agregamos una dimension para tener B x N_inputs
t_u = torch.tensor(t_u).unsqueeze(1) # Agregamos una dimension para tener B x N_inputs

n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u


# In[389]:


class SubclassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 13)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(13, 1)

        
    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t
    

subclass_model = SubclassModel()

optimizer = optim.SGD(subclass_model.parameters(), lr=1e-3)

def training_loop(model, n_epochs, optimizer, loss_fn, train_x, val_x, train_y, val_y):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_x)
        train_loss = loss_fn(train_t_p, train_y)
        
        with torch.no_grad(): 
            val_t_p = model(val_x)
            val_loss = loss_fn(val_t_p, val_y)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss}")
    
class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 100)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(100, 1)

        
    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t
    

subclass_model = MYNN()

optimizer = optim.SGD(subclass_model.parameters(), lr=1e-9)    


# In[390]:


get_ipython().run_cell_magic('time', '', 'training_loop(\n    n_epochs=5000,\n    optimizer=optimizer,\n    model=subclass_model,\n    loss_fn=nn.MSELoss(), # Ya no estamos usando nuestra loss function hecha a mano\n    train_x = train_t_un,\n    val_x = val_t_un,\n    train_y = train_t_c,\n    val_y = val_t_c)')


# * Que cambios resultan en un output mas lineal del modelo?
#     * Probe poner 100 neuronas en lugar de 13 y el learning rate lo reduje a 1e-9, por lo que el loss disminuye una fraccion nada mas cada 1000 epochs.
#     * AL poner un learning rate pequeno y aumentando el numero de neuronas, pero el numero de epochs necesarios si aumentaria.

# ## Inciso 2

# In[308]:


from numpy import genfromtxt
import numpy as np
data = genfromtxt('winequality-white.csv', delimiter=';')
data = np.array(data[1:])
data = torch.from_numpy(data).float()
features = data[:,:-1]
target = data[:,-1].unsqueeze(1)


# In[309]:


print(features.shape, target.shape)


# In[342]:


class WineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(11, 10)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(10, 1)

        
    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t


# In[394]:


validation_losses = []
epochs_losses = []
training_losses = []

def training_loop(model, n_epochs, optimizer, loss_fn, train_x, val_x, train_y, val_y):
    print(train_x.shape)
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_x) # ya no tenemos que pasar los params
        train_loss = loss_fn(train_t_p, train_y)
        with torch.no_grad(): # todos los args requires_grad=False
            val_t_p = model(val_x)
            val_loss = loss_fn(val_t_p, val_y)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss}")
        
        validation_losses.append(val_loss)
        epochs_losses.append(epoch)
        training_losses.append(train_loss)


# In[395]:


wine_model = WineModel()

t_c = target
t_u = features

n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = train_t_u
val_t_un = val_t_u

optimizer = optim.SGD(wine_model.parameters(), lr=1e-3)


# In[368]:


print(train_t_un.shape,val_t_un.shape,train_t_c.shape,val_t_c.shape)


# In[396]:


get_ipython().run_cell_magic('time', '', 'training_loop(\n    n_epochs=5000,\n    optimizer=optimizer,\n    model=wine_model,\n    loss_fn=nn.MSELoss(),\n    train_x = train_t_un,\n    val_x = val_t_un,\n    train_y = train_t_c,\n    val_y = val_t_c)')


# * Cuanto tarda en entrenar comparado al dataset que hemos estado usando?
#     * Ahora que se tiene un batch de 3919 de entrenamiento y 979 de validation, todo el loop si tarda un poco mas al anterior. COn el anterior dataset se tardo 2.41 seg en entrenar 5000 epochs, mientras con este neuvo dataset se tardo 5.68 segs por 5000 epochs.
# * Pueden explicar que factores contribuyen a los tiempos de entrenamiento?
#     * El numero de capas y neuronas por capa ya que debe de encontrar los valores de los pesos de cada neurona y el termino bias de cada neurona. Otro factor es el numero del batch, el learning rate que tan grande sea tambien afecta.
# * Pueden hacer que el loss disminuya?
#     * Si se puede disminuir hasta cierto punto, pero no valdria la pena disminuirlo tanto por el tema de overfitting.
# * Graficas

# In[397]:


plt.plot(epochs_losses, training_losses)
plt.title('Training loss during epochs')
plt.show()


# In[398]:


plt.plot(epochs_losses, validation_losses)
plt.title('Validation loss during epochs')

