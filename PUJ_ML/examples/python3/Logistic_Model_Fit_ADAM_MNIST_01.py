## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import io, numpy, os, random, requests, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import PUJ_ML.Model.Logistic, PUJ_ML.Optimizer.ADAM
import PUJ_ML.Optimizer.Debug.Simple

## -------------------------------------------------------------------------
def LoadModels( filename ):
  models = {}
  fstr = open( filename, 'r' )
  lines = fstr.readlines( )
  fstr.close( )
  for line in lines:
    toks = line.split( )
    d = int( toks[ 0 ] )
    n = int( toks[ 1 ] )
    models[ d ] = PUJ_ML.Model.Logistic( )
    models[ d ].setParameters( [ float( toks[ i ] ) for i in range( 2, len( toks ) ) ] )
  # end for
  return models
# end def

## -------------------------------------------------------------------------
def SaveModels( models, filename ):
  buf_str = ''
  for d in models:
    buf_str += str( d ) + ' ' + str( models[ d ] ) + '\n'
  # end for
  fstr = open( filename, 'w' )
  fstr.write( buf_str )
  fstr.close( )
# end def

# -- Read MNIST (hand-written digits) database
if not os.path.exists( 'mnist.npz' ):
  dataset_url = \
    'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
  response = requests.get( dataset_url, allow_redirects = True )
  fstr = open( 'mnist.npz', 'wb' )
  fstr.write( response.content )
  fstr.close( )
# end if
data = numpy.load( 'mnist.npz' )
x_test = data[ 'x_test' ]
y_test = data[ 'y_test' ]
x_train = data[ 'x_train' ]
y_train = data[ 'y_train' ]
n = x_test.shape[ 1 ] * x_test.shape[ 2 ]
x_test = numpy.matrix( numpy.reshape( x_test, ( x_test.shape[ 0 ], n ) ) )
y_test = numpy.matrix( numpy.reshape( y_test, ( y_test.shape[ 0 ], 1 ) ) )
x_train = numpy.matrix( numpy.reshape( x_train, ( x_train.shape[ 0 ], n ) ) )
y_train = numpy.matrix( numpy.reshape( y_train, ( y_train.shape[ 0 ], 1 ) ) )

# -- Read models
models = {}
if not os.path.exists( 'logistic_mnist_models.txt' ):
  buf_str = ''
  for d in numpy.unique( y_train, axis = 0 ).flatten( ):
    models[ int( d ) ] = PUJ_ML.Model.Logistic( )
    models[ int( d ) ].setParameters( [ random.uniform( -1, 1 ) for i in range( n + 1 ) ] )
  # end for
  SaveModels( models, 'logistic_mnist_models.txt' )
else:
  models = LoadModels( 'logistic_mnist_models.txt' )
# end if

# -- Train models
for d in models:

  print( 'Training label: "' + str( d ) + '"' )
  
  # Cost
  cost = PUJ_ML.Model.Logistic.Cost(
    models[ d ],
    x_train, ( y_train == float( d ) ).astype( y_train.dtype )
    )

  # Debugger
  debugger = PUJ_ML.Optimizer.Debug.Simple

  # Optimizer
  opt = PUJ_ML.Optimizer.ADAM( cost )
  opt.setMaximumNumberOfIterations( 100 )
  opt.setDebug( debugger )
  opt.fit( )

  print( 'Training label: "' + str( d ) + '" --> done!' )

# end for

SaveModels( models, 'logistic_mnist_models.txt' )

## eof - $RCSfile$
