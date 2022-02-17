## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys
sys.path.insert( 0, '../../lib/python3' )

from PUJ_ML_01.lib.python3.PUJ.Model.Linear import Linear
from PUJ_ML_01.lib.python3.PUJ.Optimizer import *

## ----------
## -- Main --
## ----------

# Load data
data = numpy.loadtxt( open( sys.argv[ 1 ], 'rb' ), delimiter = ',' )

# Configure model
m = Linear( )
m.setParameters( [ 0 for i in range( data.shape[ 1 ] ) ] )
print( 'Initial model = ' + str( m ) )

# Configure cost
J = Linear.Cost( m, data[ : , 0 : -1 ], data[ : , -1 : ] )

# Analytical model
a = Linear( numpy.matrix( data ) )

# Debugger
## debugger = PUJ.Optimizer.Debug.Simple
debugger = PlotPolynomialCost( data[ : , 0 : -1 ], data[ : , -1 : ] )

# Fit using an optimization algorithm
opt = GradientDescent( J )
opt.setDebugFunction( debugger )
opt.setLearningRate( 1e-4 )
opt.setNumberOfIterations( 10000 )
opt.setNumberOfDebugIterations( 10 )
opt.Fit( )

# Show results
print( '===========================================' )
print( '= Iterations       :', opt.realIterations( ) )
print( '= Fitted model     :', m )
print( '= Analytical model :', a )
print( '===========================================' )

## eof - $RCSfile$
