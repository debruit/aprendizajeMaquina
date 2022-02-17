## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys
sys.path.insert( 0, '../../lib/python3' )

import PUJ.Model.Linear

m = PUJ.Model.Linear( )
m.setParameters( [ -1.2, 2, 3, 1.62, 3.4 ] )
print( m )

print( m.evaluate( [ 2, 4, 5, 6 ] ) )
print( m.threshold( [ 2, 4, 1, 2 ] ) )

## eof - $RCSfile$
