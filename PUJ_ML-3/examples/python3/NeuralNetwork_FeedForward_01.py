## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, os, sys
sys.path.append( os.path.join( os.getcwd( ), '../../lib/python3' ) )

import PUJ_ML.Model.NeuralNetwork.FeedForward

model = PUJ_ML.Model.NeuralNetwork.FeedForward( input_size = 2 )
model.addLayer( 'ReLU', size = 8 )
model.addLayer( 'ReLU', size = 4 )
model.addLayer( 'ReLU', size = 2 )
model.addLayer( 'Sigmoid', size = 1 )

print( model )
print( model.evaluate( [ 10, 20 ] ) )
print( model.evaluate( numpy.matrix( [ 10, 20, -10, -20, 34, -100 ] ).reshape( ( 3, 2 ) ) ) )
print( model.threshold( numpy.matrix( [ 10, 20, -10, -20, 34, -100 ] ).reshape( ( 3, 2 ) ) ) )


## eof - $RCSfile$
