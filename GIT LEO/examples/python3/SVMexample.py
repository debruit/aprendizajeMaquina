## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import random
import numpy, sys
sys.path.append('/Users/debruit/Library/CloudStorage/OneDrive-PontificiaUniversidadJaveriana/Noveno/ML/GIT/aprendizajeMaquina/GIT LEO/lib/python3')

import PUJ

## ----------
## -- Main --
## ----------

# Some parameters
numP = 1000
numN = 1000

# Load data
data = numpy.loadtxt( open( "/Users/debruit/Library/CloudStorage/OneDrive-PontificiaUniversidadJaveriana/Noveno/ML/GIT/aprendizajeMaquina/GIT LEO/examples/data/binary_data_01.csv", 'rb' ), delimiter=',' )
P = data[ data[ : , 2 ] == 1 ]
N = data[ data[ : , 2 ] == 0 ]
numpy.random.shuffle( P )
numpy.random.shuffle( N )
data = numpy.concatenate( ( P[ : numP , : ], N[ : numN , : ] ), axis = 0 )
numpy.random.shuffle( data )

X = data[ : ,  0 : -1 ]
Y = data[ : , -1 : ]

# Configure model
m = PUJ.Model.SVM( )
m.setParameters(
  [ random.uniform( -1, 1 ) for i in range( X.shape[ 1 ] + 1 ) ]
  )
print( 'Initial model = ' + str( m ) )

# Configure cost
J = PUJ.Model.SVM.Cost( m, X, Y )

# Debugger
debugger = PUJ.Optimizer.Debug.Simple
## debugger = PUJ.Optimizer.Debug.PlotPolynomialCost( X, Y )
# debugger = PUJ.Optimizer.Debug.Labeling( X, Y, 0.5 )

# Fit using an optimization algorithm
opt = PUJ.Optimizer.GradientDescent( J )
opt.setDebugFunction( debugger )
opt.setLearningRate( 1e-10 ) 
opt.setNumberOfIterations( 100000000 )
opt.setNumberOfDebugIterations( 10000 )
opt.setLambda( 5 )
opt.Fit( )

# Show results
print( '===========================================' )
print( '= Iterations       :', opt.realIterations( ) )
print( '= Fitted model     :', m )
print( '===========================================' )

Y_est = m.threshold( X )
K = numpy.zeros( ( 2, 2 ) )

for i in range( Y.shape[ 0 ] ):
  K[ int( Y[ i, 0 ] ), int( Y_est[ i, 0 ] ) ] += 1
# end for

print( K )

# debugger.KeepFigures( )

## eof - $RCSfile$
