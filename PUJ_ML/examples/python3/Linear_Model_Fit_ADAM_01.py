## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import PUJ_ML.Model.Linear, PUJ_ML.Optimizer.ADAM
import PUJ_ML.Optimizer.Debug.Simple

# Some parameters to generate sample data
M = 10
minX = -100
maxX = 100

# Some model
dm = PUJ_ML.Model.Linear( )
dm.setParameters( [ 0.3, 2, 0.5 ] )
N = dm.numberOfParameters( )

# Some random data
X = numpy.random.uniform( minX, maxX, ( M, dm.inputSize( ) ) )
Y = dm.evaluate( X )

# Model to be optimized
om = PUJ_ML.Model.Linear( )
om.setParameters( [ 0 for i in range( N ) ] )

# Cost
cost = PUJ_ML.Model.Linear.Cost( om, X, Y )

# Debugger
debugger = PUJ_ML.Optimizer.Debug.Simple

# Optimizer
opt = PUJ_ML.Optimizer.ADAM( cost )
opt.setDebug( debugger )
opt.fit( )

# Show results
print( '******************************' )
print( '* Original model  :', dm )
print( '* Optimized model :', om )
print( '******************************' )

## eof - $RCSfile$
