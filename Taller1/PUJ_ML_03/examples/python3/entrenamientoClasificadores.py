
import tensorflow as tf
import numpy as np
import os, sys

sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )
import PUJ
from PUJ.Optimizer.Analisis.matrizConfusion import matrizConfusion

def crear_y(digit:int,datos):
  y=np.copy(datos)
  for i in range (y.shape[0]):
    if(datos[i]==digit):
      y[i]=1
    else:
      y[i]=0
  return np.expand_dims(y, axis = 1)

def crear_x(datos):
  view = datos.view()
  return np.reshape(view,(datos.shape[0],784))

def guardarPesos(archivo,pesos):
  np.save(archivo+".npy",pesos)



tf.keras.datasets.mnist.load_data(path="mnist.npz")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

size=x_train.shape[1]*x_train.shape[2]

xtrain_rows=crear_x(x_train)
xtest_rows=crear_x(x_test)
ytrain_0=crear_y(9,y_train)
ytest_0=crear_y(9,y_test)

m = PUJ.Model.Logistic( )
m.setParameters( [ 0 for i in range( size +1 ) ] )
print( 'Initial model = ' + str( m ) )

# Configure cost
J = PUJ.Model.Logistic.Cost( m, xtrain_rows, ytrain_0 )
# Debugger
debugger = PUJ.Optimizer.Debug.Simple

# Fit using an optimization algorithm
opt = PUJ.Optimizer.GradientDescent( J )
opt.setDebugFunction( debugger )
opt.setLearningRate( 1e-5 )
opt.setNumberOfIterations( 15000 )
opt.setNumberOfDebugIterations( 100 )
opt.setLambda(6)
opt.Fit( )

# Show results
print( 'Resultados en el datset de entrenamiento' )
print( '===========================================' )
print( '= Iterations       :', opt.realIterations( ) )
print( '= Fitted model     :', m )
print( '===========================================' )

y_real = ytrain_0
y_est = m.threshold( xtrain_rows )

mtz = matrizConfusion(2)
print(mtz.calcular(y_real, y_est))
print("la exactitud es: ", mtz.exactitud())
print("el f1 es: ", mtz.f1_score())
print("la presicion promedio es: ", mtz.presicionProm())
print("el recall promedio es: ", mtz.recallProm())


print( 'Resultados en el datset de prueba' )
print( '===========================================' )
y_real = ytest_0
y_est = m.threshold( xtest_rows )

mtz =matrizConfusion(2)
print(mtz.calcular(y_real, y_est))
print("la exactitud es: ", mtz.exactitud())
print("el f1 es: ", mtz.f1_score())
print("la presicion promedio es: ", mtz.presicionProm())
print("el recall promedio es: ", mtz.recallProm())

guardarPesos("entrenador9",m.parameters())