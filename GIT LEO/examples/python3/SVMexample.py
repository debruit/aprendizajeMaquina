## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import random
import numpy as np
import matplotlib.pyplot as plt
import sys
#sys.path.append('/Users/debruit/Library/CloudStorage/OneDrive-PontificiaUniversidadJaveriana/Noveno/ML/GIT/aprendizajeMaquina/GIT LEO/libPy/python3')
sys.path.append(r'C:\Users\Estudiante\Documents\GitHub\aprendizajeMaquina\GIT LEO\libPy\python3')


import PUJ

## ----------
## -- Main --
## ----------

# Some parameters
numP = 1000
numN = 1000


# Load data

def graficarInfo(x,y,lbl):
  plt.scatter(x, y,
              c=lbl, edgecolor='none', alpha=0.5,
              cmap=plt.cm.get_cmap('Paired'))
  plt.xlabel('component 1')
  plt.ylabel('component 2')
  plt.colorbar()
  plt.show()


def dispersor(y,largo, limite):
  tam=y.shape[0]
  alt=np.array([random.randint(largo, limite)*(-1)**i for i in range(tam)])
  return y+alt

def crearDataLinealX(minX, maxX, minY, maxY, tam, pendiente,sezgo):
  x=np.array([random.randint(minX, maxX) for i in range(tam)])
  y=x*pendiente+sezgo
  y=dispersor(y,minY,maxY)
  res=np.array([x,y])
  return (res.T)

def crearDataLinealY(tam):
   return np.array([ (-1)**i for i in range(tam)]).reshape(tam,1)

tam = 1000

X = crearDataLinealX(-20,20,15,25,tam,1,0)
Y = crearDataLinealY(tam)



# print(Y.shape)

# Configure model
m = PUJ.Model.SVM( )
m.setParameters(
  #[-0.000, 0.020, -5.215]
  [ random.uniform( -1, 1 ) for i in range( X.shape[ 1 ] + 1 ) ]
  )
print( 'Initial model = ' + str( m ) )



# Configure cost
J = PUJ.Model.SVM.Cost( m, X, Y, 1)

# Debugger
debugger = PUJ.Optimizer.Debug.Simple
#debugger = PUJ.Optimizer.Debug.PlotPolynomialCost( X, Y )
# debugger = PUJ.Optimizer.Debug.Labeling( X, Y, 0.5 )

# Fit using an optimization algorithm
opt = PUJ.Optimizer.GradientDescent( J )
opt.setDebugFunction( debugger )
opt.setLearningRate( 1e-8 )
opt.setNumberOfIterations( 10000 )
opt.setNumberOfDebugIterations( 100 )
opt.setLambda(2)
#opt.setRegularizationToLASSO()
opt.Fit( )

# Show results
print( '===========================================' )
print( '= Iterations       :', opt.realIterations( ) )
print( '= Fitted model     :', m )
print( '===========================================' )

Y_est = np.array(m.threshold( X ))

#print(Y_est)
#print(Y)

K = np.zeros( ( 2, 2 ) )
for i in range( Y.shape[ 0 ] ):
  if int( Y[ i, 0 ] ) == 1 and int( Y_est[ i, 0 ] ) == 1:
    K[ 0, 0 ] += 1
  if int( Y[ i, 0 ] ) == -1 and int( Y_est[ i, 0 ] ) == 1:
    K[ 1, 0 ] += 1
  if int( Y[ i, 0 ] ) == 1 and int( Y_est[ i, 0 ] ) == -1:
    K[ 0, 1 ] += 1
  if int( Y[ i, 0 ] ) == -1 and int( Y_est[ i, 0 ] ) == -1:
    K[ 1, 1 ] += 1
  if int( Y[ i, 0 ] ) == -1 and int(Y_est[i, 0]) == 0:
    K[1, 0] += 1
  if int(Y[i, 0]) == 1 and int(Y_est[i, 0]) == 0:
    K[0, 1] += 1
# end for
print( K )

print("La exactitud es :", (K[0, 0]+ K[1, 1])/ (K[0, 0]+ K[1, 1]+ K[0, 1] +K[1, 0]) )

graficarInfo(X[:, 0], X[:, 1], Y)
graficarInfo(X[:, 0], X[:, 1], Y_est )

print("No se le olvide ordenar")

# debugger.KeepFigures( )

## eof - $RCSfile$
