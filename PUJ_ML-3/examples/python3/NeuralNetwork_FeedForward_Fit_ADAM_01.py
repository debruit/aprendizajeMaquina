## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import PUJ_ML

# Read data
D = numpy.loadtxt( open( sys.argv[ 1 ], 'rb' ), delimiter = ',' )

# Keep some data
m = 100
Z = D[ D[ : , -1 ] == 0 ]
O = D[ D[ : , -1 ] == 1 ]
numpy.random.shuffle( Z )
numpy.random.shuffle( O )
D = numpy.concatenate( ( O[ 0 : m , : ], Z[ 0 : m , : ] ) )
numpy.random.shuffle( D )
X = D[ : , : -1 ]
Y = D[ : , -1 : ]

# Standardize data
C = X - X.mean( axis = 0 )
S = ( C.T @ C ) / float( X.shape[ 0 ] - 1 )
X = C / numpy.sqrt( S.diagonal( ) )

# Prepare model
opt_model = PUJ_ML.Model.NeuralNetwork.FeedForward( input_size = X.shape[ 1 ] )
opt_model.addLayer( 'ReLU', size = 16 )
opt_model.addLayer( 'ReLU', size = 8 )
opt_model.addLayer( 'ReLU', size = 4 )
opt_model.addLayer( 'ReLU', size = 2 )
opt_model.addLayer( 'Sigmoid', size = 1 )

# Prepare cost
cost = PUJ_ML.Model.NeuralNetwork.FeedForward.Cost( opt_model, X, Y )

# Debugger
debugger = PUJ_ML.Optimizer.Debug.Labeling( X, Y, 0.5 )

# Optimizer
opt = PUJ_ML.Optimizer.ADAM( cost )
opt.setMaximumNumberOfIterations( 10000 )
opt.setNumberOfDebugIterations( 1000 )
opt.setDebug( debugger )
opt.fit( )

# Show results
print( '******************************' )
print( '* Iterations : ' + str( opt.iteration( ) ) )
print( '* Optimized model :\n' + str( opt_model ) )
print( '******************************' )

# Keep showing figures
debugger.KeepFigures( )

## eof - $RCSfile$
