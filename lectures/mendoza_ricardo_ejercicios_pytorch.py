
import numpy as np
import torch
import torch.optim as optim

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] # Temperatura en grados celsios
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] # Unidades desconocidas
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)



# # Ejercicios

# 1. Redefinan el model a `w2 * t_u ** 2 + w1 * t_u + b`
#     * Que partes del training loop necesitaron cambiar para acomodar el nuevo modelo?
#     * Que partes se mantuvieron iguales?
#     * El _loss_ resultante es mas alto o bajo despues de entrenamiento?
#     * El resultado es mejor o peor?

# * Se cambio unicamente la funcion de model() para que aceptara otro parametro, en este caso w2.
# * El loop de entrenamiento esta igual, la funcoin de perdida tambien, al igual que el optimizador y el learning_rate.
# * Siguiendo con los mismo 2000 epochs, inicializando los pesos con 1 y el bias con 0 el loss despues del entrenamiento si bajo.
# * El resultado despues del entrenamiento es mejor, pero a comparacion del anterior modelo no es tan bueno, pues el error si quedo mas alto que el modelo anterior.

def training_loop(model, n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss {loss}")
            
    return params

def model(t_u, w1, w2, b):
    return w2 * t_u ** 2 + w1 * t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[33]:


params = torch.tensor([1.0 , 1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate) # Nuevo optimizador

training_loop(model,
              n_epochs=2000,
              optimizer=optimizer,
              params = params,
              t_u = t_u, # Regresamos a usar el t_u original como input
              t_c = t_c)

