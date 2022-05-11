
import os, sys
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )
# import PUJ_ML.Analisis.matrizConfusion as matrizConfusion
import PUJ_ML.Model.NeuralNetwork.FeedForward as FeedForward

#Carga de la informacion
tf.keras.datasets.mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

def crear_x(datos):
  view = datos.view()
  return np.reshape(view,(datos.shape[0],784))

def convertirY(y):
  z=np.array(512/2**y, dtype=np.int16)
  bin_nums = ((z.reshape(-1,1) & (2**np.arange(10))) != 0).astype(int)
  return bin_nums[:,::-1]

x_train2 = crear_x(x_train)
x_test2 = crear_x(x_test)
y_train2 = convertirY(y_train)
y_test2 = convertirY(y_test)

model = FeedForward( input_size = 784 )
model.addLayer( 'ReLU', size = 32 )
model.addLayer( 'ReLU', size = 16 )
model.addLayer( 'SoftMax', size = 10 )

# #incializacion del clasificador
# cls=clasificadorNumero()
# npzfile = np.load(sys.argv[ 1 ])
# cls.agregarModelos(npzfile)

# #verificacion sobre conjunto de prueba
# view = x_test.view()
# res=cls.evaluar(np.reshape(view,(x_test.shape[0],784)))
# y_pred=np.expand_dims(res, axis = 1)
# y_real=np.expand_dims(y_test, axis = 1)
# mtz=matrizConfusion(10)
# print("\n", "Carga de clasificadores exitosa:")
# print("\n", "El desempeño sobre el conjunto de prueba es: \n")
# mtz.calcular(y_real, y_pred)
# print("la exactitud es: ", mtz.exactitud())
# print("el f1 es: ", mtz.f1_score())
# print("la presicion promedio es: ", mtz.presicionProm())
# print("el recall promedio es: ", mtz.recallProm())

# # -- Read all data
# X = np.concatenate( ( x_train, x_test ), axis = 0 )
# Y = np.concatenate( (y_train, y_test ), axis = 0 )
# m = X.shape[ 0 ]

# # -- Play with the user
# print("\n", "Hora de jugar: \n")
# i = int( input( 'Type a number between 0 and ' + str( m - 1 ) + ': ' ) )
# while i >= 0 and i < m:
#   image = X[ i ]
#   x = image.reshape( ( 1, image.shape[ 0 ] * image.shape[ 1 ] ) )
#   idx=cls.evaluarImagen(x)
#   print( '**********************' )
#   print( '* Detected label :', idx )
#   print( '* Real label     :', Y[ i ] )
#   print( '**********************' )
#   plt.imshow( image, cmap = 'gray' )
#   plt.show( )
#   i = int( input( 'Type a number between 0 and ' + str( m - 1 ) + ': ' ) )
# # end while
