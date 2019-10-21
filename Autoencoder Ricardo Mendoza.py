#!/usr/bin/env python
# coding: utf-8

# # Autoencoders

# * Una red neuronal compuesta por dos partes:
#     * Un _encoder_ network que comprime input data de alta dimension a una representacion vectorial de baja dimension
#     * Un _decoder_ network que decomprime una representacion vectorial dada de regreso al dominio original

# ![Diagrama de Autoencoder](../assets/autoencoder.png)

# * La red esta entrenada para encontrar los pesos para el encoder y decoder que minimizan el loss entre
#     * el input original
#     * la reconstruccion del input
# * El vector de representacion es una compresion de la imagen original a un espacio latente con baja dimensionalidad.
# * La idea es que al escoger _cualquier_ punto en el espacio latente, deberiamos poder generar imagenes nuevas al pasar este punto a traves del decoder.
#     * Porque el decoder aprendio el mapa: puntos en espacio latente -> imagenes viables

# Vamos a empezar construyendo un autoencoder simple para comprimir el dataset de MNIST. Con autoencoders, pasamos input data a traves del encoder que crea la representacion comprimida del input. Luego, esta representacion pasa a traves del decoder para reconstruir la data de input. Generalmente el encoder y decoder se construyen usando NNs, luego se entrenan en data de ejemplos.

# ## Representacion comprimida

# Una representacion comprimida puede ser buena para guardar y compartir cualquier tipo de data en una forma que sea mas eficiente que guardar la data cruda. En la practica, la representacion comprimida contiene informacion clave sobre la imagen de input y la podemos usar para reconstruir, denoising y otras transformaciones.
# 
# En este notebook vamos a construir una simple NN para el encoder y decoder.

# In[1]:


import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

# convertira data a torch.FloatTensor
transform = transforms.ToTensor()

# cargar los datasets de training y test
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)


# In[53]:


# Crear los dataloaders para training y test
# numero de subprocesses para usar para cargar la data
num_workers = 1
# numero de muestras por batch para cargar
batch_size = 20

# data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


# ### Visualizar la data

# In[54]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
    
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

img = np.squeeze(images[0])

fig = plt.figure(figsize = (5,5)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')


# In[55]:


prueba = torch.tensor(img)
prueba = prueba.view(-1)
prueba.shape


# ## Autoencoder lineal
# 
# Vamos a entrenar un autoencoder con estas imagenes aplanandolas a vectores de largo 784. Las imagenes de este dataset ya estan normalizadas para que sus valores esten entre 1 y 0. El encoder y decoder deberian estar hechos de **una capa lineal**. Las unidades que conectan el encoder y decoder van a ser la _representacion comprimida_.
# 
# Como las imagenes estan normalizadas, necesitamos usar una activacion **sigmoid** en la capa de output para obtener valores que se encuentren en el mismo rango que el input.
# 
# **TODO: Construir el autoencoder en la celda de abajo**
# > Las imagenes de input deben ser aplanadas a vectores de 784. Los targets son los mismos que los inputs. El encoder y el decoder van a estar hechos de dos capas lineales cada uno. La profundidad de las dimensiones deben cambiar de la siguiente forma: 784 inputs -> **encoding_dims** -> 784 outputs. Todas las capas deben tener activaciones ReLu aplicadas excepto por la capa final que debe ser un sigmoid
# 
# **La representacion comprimida debe ser un vector con dimension `encoding_dim=32`**

# In[56]:


import torch.nn as nn
import torch.nn.functional as F

# Definir arquitectura
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        ## encoder ##
        self.encoder1 = nn.Linear(784,32)
        self.decoder1 = nn.Linear(32,784)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x se convierte a vector plano de 784
        # definir feedforward
        output = F.relu(self.encoder1(x))
        output2 = F.relu(self.decoder1(output))
        
        # aplicar sigmoid al output layer
        output3 = self.sigmoid(output2)
        
        return output3

# inicializar NN
encoding_dim = 32
model = Autoencoder(encoding_dim)
print(model)


# ### Entrenamiento

# El resto es codigo para entrenamiento, deberia resultarles familiar. En este caso no nos interesa mucho la validacion, asi que solo vamos a monitorear el training loss y el test loss.
# 
# Tampoco nos preocupamos por los labels en este caso. Como estamos comparando valores de pixeles en las imagenes de input y output, lo mejor es usar una los que sea util para tareas de regresion. La regresion se utiliza para comparar _cantidades_ en vez de valores probabilisticos. En este caso vamos a usar `MSELoss`. 

# In[57]:


# especificar loss function
criterion = nn.MSELoss()

# especificar optimizador
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[58]:


# numero de epochs para entrenar el modelo
n_epochs = 20

for epoch in range(1, n_epochs+1):
    # training loss
    train_loss = 0.0
    
    ######################
    # entrenar el modelo #
    ######################
    for data in train_loader:
        # _ para los labels (como no nos interesan)
        images, _ = data
        # aplanar imagenes
        images = images.view(images.size(0), -1)
        # limpiar las gradientes de todas las variables optimizadas
        optimizer.zero_grad()
        # forward pass: calcular valores predecidos
        outputs = model(images)
        # calcular el loss
        loss = criterion(outputs, images)
        # backward pass: calcular gradiente del los con respecto al modelo
        loss.backward()
        # realizar un unico paso de optimizacion (actualizar parametros)
        optimizer.step()
        # actualizar training loss acumulado
        train_loss += loss.item()*images.size(0)
            
    # print estadisticas de entrenamiento
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))


# ### Revisar los resultados

# El codigo de abajo son plots de las imagenes de entrenamiento con sus reconstrucciones. 

# In[60]:


# obtener un batch del test set
dataiter = iter(test_loader)
images, labels = dataiter.next()

images_flatten = images.view(images.size(0), -1)
# sample outputs
output = model(images_flatten)
# preparar imagenes para display
images = images.numpy()

# output es redimensionado a un batch de imagenes
output = output.view(batch_size, 1, 28, 28)
# usar detach cuando es un output que requires_grad=False
output = output.detach().numpy()

# plot las primeras 10 input images y luego reconstruir las imagenes
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

# inputs arriba, reconstrucciones abajo
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


# Estamos tratando con imagenes, por lo que (usualmente) obtendriamos mejor performance usando convolution layers.

# ## Convolutional Autoencoder

# Siguiendo con el dataset de MNIST, vamos a mejorar el performance del autoencoder usando convolutional layers. Vamos a construir un autoencoder para comprimir el dataset de MNIST.
# > La porcion del encoder estara compuesta de convolutional y pooling layers y el decoder va a estar hecho de **transpose convolutional layers** que aprenden a "upsample" la representacion comprimida.

# In[61]:


import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)


# In[62]:


num_workers = 0
batch_size = 20

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


# In[63]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
    
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

img = np.squeeze(images[0])

fig = plt.figure(figsize = (5,5)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')


# ### Arquitectura

# **Encoder**
# 
# La parte de la red para el encoder va a ser una tipica piramide de convolutions. Cada convolutional layer va a ser seguida por un layer de max-pooling para reducir las dimensiones de los layers.
# 
# **Decoder**
# 
# El decoder si es algo nuevo. El decoder necesita convertir una representacion estrecha a una imagen ancha reconstruida. Por ejemplo, la representacion puede ser un max-pool layer de 7x7x4. Este es el outpu del encoder, pero tambien el input para el encoder. Queremos obtener una imagen de 28x28x1 de este decoder. 

# ![autoencoder architecture](../assets/conv_enc_1.png)

# Nuestra ultima layer del encoder tiene un tamanio de 7x7x4 = 196. Las imagenes originales tienen tamanio 28x28 = 784, asi que el encoded vector es 25% el tamanio de la imagen original. Estos son solamente tamanios sugeridos para cada layer. Intenten cambiar las depths y sizes y agregar layers adicionales para hacer esta representacion hasta mas pequenia. Nuestra meta aqui es encontrar una representacion pequenia del input data.

# ### Transpose convolutions, Decoder

# Este decoder usa **transposed convolutional** layers para incrementar el ancho y altura de las input layers. Funcionan casi exactamente igual que las convolutional layers, pero en reversa. Un stride en el input layer resulta en un stride mas largo en el transposed convolution layer. Por ejemplo, si tenemos un kernel de 3x3, un patch de 3x3 en el input layer va a ser reducido a una unidad en el convolutional layer. Asimismo, una unidad en el input layer se va a expandir a un patch de 3x3 en un transposed convolution layer. PyTorch nos provee con una forma facil de crear los layers, [`nn.ConvTranspose2d`](https://pytorch.org/docs/stable/nn.html#convtranspose2d).
# 
# Es importante notar que los transpose convolution layers pueden introducir artefactos en las imagenes finales, como patrones cuadriculados. Esto se debe al overlap en los kernels los cuales pueden ser evitados configurando el tamanio del stride y el kernel para que sean del mismo tamanio. En este [articulo de Distill](http://distill.pub/2016/deconv-checkerboard/) de Augustus Odena, _et. al_, los autores demuestran como estos artefactos pueden ser evitados.
# 
# **TODO: Desarrollar la red**
# > Construir el encoder compuesto de una serie de convolutional y pooling layers. Cuando construyan el decoder, recuerden que los transpose convolutional layers pueden upsample un input por un factor de 2 usando un stride y un kernel_size de 2.

# In[89]:


# pool of square window of size=3, stride=2
m = nn.MaxPool2d(1, stride=2)
# pool of non-square window
# m = nn.MaxPool2d((3, 2), stride=(2, 1))
input = torch.randn(4, 4, 1)
print(input.shape)
output = m(input)
output.shape


# In[97]:


import torch.nn as nn
import torch.nn.functional as F

# definir la arquitectura
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv_encoder1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.max_encoder1 = nn.MaxPool2d(2, 2)
        self.conv_encoder2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.max_encoder2 = nn.MaxPool2d(2, 2)
                                      
        ## decoder layers ##
        ## un kernel de 2 y un stride de 2 van a incrementar las dims por 2
        self.conv_decoder1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.conv_decoder2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
        
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        ## encode ##
        out = F.relu(self.conv_encoder1(x))
        out = F.relu(self.max_encoder1(out))
        out = F.relu(self.conv_encoder2(out))
        out = F.relu(self.max_encoder2(out))
        out = F.relu(self.conv_decoder1(out))
        out = F.relu(self.conv_decoder2(out))
        out = self.sigmoid(out)
        
        ## decode ##
        ## aplicar ReLu a todas las hidden layers excepto el output layer
        ## aplicar sigmoid al output layer
        
                
        return out

# inicializar el NN
model = ConvAutoencoder()
print(model)


# In[98]:


criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[99]:


n_epochs = 30

for epoch in range(1, n_epochs+1):

    train_loss = 0.0

    for data in train_loader:
        images, _ = data
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)
            
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))


# In[101]:


dataiter = iter(test_loader)
images, labels = dataiter.next()

output = model(images)
images = images.numpy()

output = output.view(batch_size, 1, 28, 28)
output = output.detach().numpy()

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))


for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

