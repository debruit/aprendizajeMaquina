## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import random
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/debruit/Library/CloudStorage/OneDrive-PontificiaUniversidadJaveriana/Noveno/ML/GIT/aprendizajeMaquina/GIT LEO/libPy/python3')

import PUJ

## ----------
## -- Main --
## ----------

# Some parameters
numP = 1000
numN = 1000

# Load data
# data = np.loadtxt( open( "/Users/debruit/Library/CloudStorage/OneDrive-PontificiaUniversidadJaveriana/Noveno/ML/GIT/aprendizajeMaquina/GIT LEO/examples/data/binary_data_01.csv", 'rb' ), delimiter=',' )
# P = data[ data[ : , 2 ] == 1 ]
# N = data[ data[ : , 2 ] == 0 ]
# np.random.shuffle( P )
# np.random.shuffle( N )
# data = np.concatenate( ( P[ : numP , : ], N[ : numN , : ] ), axis = 0 )
# np.random.shuffle( data )


# X = data[ : ,  0 : -1 ]
# Y = data[ : , -1 : ]

# print(Y.shape)


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

X = crearDataLinealX(-20,20,40,65,tam,1,0)
Y = crearDataLinealY(tam)

# print(Y.shape)

# Configure model
m = PUJ.Model.SVM( )
m.setParameters(
  [ random.uniform( -1, 1 ) for i in range( X.shape[ 1 ] + 1 ) ]
  )
print( 'Initial model = ' + str( m ) )

# Configure cost
J = PUJ.Model.SVM.Cost( m, X, Y, 32 )

# Debugger
debugger = PUJ.Optimizer.Debug.Simple
# debugger = PUJ.Optimizer.Debug.PlotPolynomialCost( X, Y )
# debugger = PUJ.Optimizer.Debug.Labeling( X, Y, 0.5 )

# Fit using an optimization algorithm
opt = PUJ.Optimizer.GradientDescent( J )
opt.setDebugFunction( debugger )
opt.setLearningRate( 1e-5 ) 
opt.setNumberOfIterations( 100000 )
opt.setNumberOfDebugIterations( 100 )
opt.setLambda( 4 )
opt.Fit( )

# Show results
print( '===========================================' )
print( '= Iterations       :', opt.realIterations( ) )
print( '= Fitted model     :', m )
print( '===========================================' )

Y_est = m.threshold( X )
K = np.zeros( ( 2, 2 ) )

for i in range( Y.shape[ 0 ] ):
  K[ int( Y[ i, 0 ] ), int( Y_est[ i, 0 ] ) ] += 1
# end for

print( K )

# debugger.KeepFigures( )

## eof - $RCSfile$
