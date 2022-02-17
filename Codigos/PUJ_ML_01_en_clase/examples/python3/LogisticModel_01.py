## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys
sys.path.insert( 0, '../../lib/python3' )

from PUJ_ML_01_en_clase.lib.python3.PUJ.Model.Logistic import  Logistic
from PUJ_ML_01_en_clase.lib.python3.PUJ.Optimizer import *

# Load data
data = numpy.loadtxt( open( sys.argv[ 1 ], 'rb' ), delimiter = ',' )

m = Logistic( )
m.setParameters( [ 0 for i in range( data.shape[ 1 ] ) ] )
print( 'Initial model = ' + str( m ) )

# Configure cost
J = Logistic.Cost( m, data[ : , 0 : -1 ], data[ : , -1 : ] )

# Debugger
debugger = Debug.Simple

# Fit using an optimization algorithm
opt = GradientDescent( J )
opt.setDebugFunction( debugger )
opt.setLearningRate( 1e-2 )
opt.setNumberOfIterations( 1000 )
opt.setNumberOfDebugIterations( 10 )
opt.Fit( )

# Show results
print( '===========================================' )
print( '= Iterations       :', opt.realIterations( ) )
print( '= Fitted model     :', m )
print( '===========================================' )

"""
m.setParameters( [ 0, -1, 0 ] )
print( m )

print( m.evaluate( [ 2, 2 ] ) )
print( m.threshold( [ 2, 2 ] ) )

print( m.evaluate( [ -2, -1 ] ) )
print( m.threshold( [ -2, -1 ] ) )

print( m.evaluate( [ -10, 30 ] ) )
print( m.threshold( [ -10, 30 ] ) )
"""



## eof - $RCSfile$
