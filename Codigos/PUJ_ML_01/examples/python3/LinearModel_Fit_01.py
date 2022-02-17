## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, sys
sys.path.insert( 0, '../../lib/python3' )

from PUJ_ML_01.lib.python3.PUJ.Model.Linear import Linear

data = numpy.loadtxt( open( sys.argv[ 1 ], 'rb' ), delimiter = ',' )
m = Linear( numpy.matrix( data ) )
print( m )

## eof - $RCSfile$
