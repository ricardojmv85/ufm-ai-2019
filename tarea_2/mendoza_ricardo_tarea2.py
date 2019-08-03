#!/usr/bin/env python
# coding: utf-8

# ## Ejercicios

# 1. Crear un tensor de `list(range(9))` e indicar cual es el `size`, `offset`, y `strides`
#     * Crear un tensor `b = a.view(3, 3)`. Cual es el valor de `b[1, 1]`
#     * crear un tensor `c = b[1:, 1:]`. Cual es el `size`, `offset`, `strides`?
# 2. Escogan una operacion matematica como cosine o sqrt. Hay una funcion correspondiente en PyTorch?
#     * Existe una version de esa operacion que opera `in-place`?

# 1. Crear un tensor 2D y luego agregar una dimension de tamanio 1 insertada en la dimension 0.
# 2. Eliminar la dimension extra que agrego en el tensor previo.
# 3. Crear un tensor aleatorio de forma $5x3$ en el intervalo $[3,7)$
# 4. Crear un tensor con valores de una distribucion normal ($\mu=0, \sigma=1$)
# 5. Recuperar los indices de todos los elementos no cero en el tensor `torch.Tensor([1,1,1,0,1])`.
# 6. Crear un tensor aleatorio de forma `(3,1)` y luego apilar cuatro copias horizontalmente.
# 7. Retornar el producto batch matrix-matrix de dos matrices 3D: (`a=torch.randn(3,4,5)`, `b=torch.rand(3,5,4)`)
# 8. Retornar el producto batch matrix-matrix de una matriz 3D y una matriz 2D: (`a=torch.rand(3,4,5)`, `b=torch.rand(5,4)`).

# In[7]:


import torch


# In[13]:


# EJERCICIO 1 SERIE 1
a=torch.tensor(list(range(9)))
print('Size: ', a.size(), ', Offset:', a.storage_offset(), ', Stride:', a.stride())

b = a.view(3,3)
print('Valor de b[1,1]: ', b[1,1])

c = b[1:, 1:]
print('Size: ', c.size(), ', Offset:', c.storage_offset(), ', Stride:', c.stride())


# In[16]:


# EJERCICIO 2 SERIE 1
# Operacion de suma, pytorch provee la funcion .add_() y esta misma funciona como in-place ya que no crea una copia de un tensor
sumando1 = torch.ones(3)
sumando2 = torch.ones(3)
sumando1.add_(sumando2)
print(sumando1)


# In[26]:


# EJERCICIO 1 SERIE 2
a = torch.ones([2, 2])
a.unsqueeze_(0)
print(a.size())


# In[34]:


# EJERCICIO 2 SERIE 2
a.resize_((2, 2))
print(a.size())


# In[57]:


# EJERCICIO 3 SERIE 2
a = torch.randint(size = [5, 3], low = 3, high = 7)
print(a)


# In[67]:


# EJERCICIO 4 SERIE 2
a = torch.normal(torch.zeros(4), torch.ones(4))
print(a)


# In[75]:


# EJERCICIO 5 SERIE 2
a = torch.Tensor([1, 1, 1, 0, 1])
print((a == 0).nonzero()[0])


# In[86]:


# EJERCICIO 6 SERIE 2
a = torch.rand(3, 1)
b = torch.stack([a, a, a, a])
print(b.size())


# In[88]:


# EJERCICIO 7 SERIE 2
a = torch.randn(3, 4, 5) 
b = torch.rand( 3, 5, 4)
producto = torch.bmm(a, b)
print(producto)


# In[90]:


# EJERCICIO 8 SERIE 2
a = torch.rand(3, 4, 5)
b = torch.rand(5, 4)
producto = torch.bmm(a, b.expand(3, 5, 4))
print(producto)

