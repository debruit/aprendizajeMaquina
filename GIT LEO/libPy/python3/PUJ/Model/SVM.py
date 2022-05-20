## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

from .Base import *

'''
'''
class SVM( Base ):

  ## -----------------------------------------------------------------------
  ## Initialize an object witha zero-sized parameters vector
  ## -----------------------------------------------------------------------
  def __init__( self, data = None ):
    if not data is None:
      if not isinstance( data, numpy.matrix ):
        raise ValueError(
          'Input data is not a numpy.matrix (' + str( type( data ) ) + ')'
          )
      # end if

      # # Get input matrices
      # m = data.shape[ 0 ]
      # n = data.shape[ 1 ] - 1
      # X = data[ : , 0 : n ]
      # y = data[ : , n : ]

      # # Build vector
      # b = numpy.zeros( ( n + 1, 1 ) )
      # b[ 0 , 0 ] = y.mean( )
      # b[ 1 : , : ] = numpy.multiply( X, y ).mean( axis = 0 ).T

      # # Build matrix
      # A = numpy.identity( n + 1 )
      # A[ 1 : , 1 : ] = ( X.T @ X ) / float( m )
      # A[ 0 : 1 , 1 : ] = X.mean( axis = 0 )
      # A[ 1 : , 0 : 1 ] = A[ 0 : 1 , 1 : ].T

      # # Solve system
      # self.m_P = numpy.linalg.inv( A ).T @ b
    # end if
  # end if

  ## -----------------------------------------------------------------------
  ## Final-user methods.
  ## -----------------------------------------------------------------------
  def evaluate( self, x ):
    rx = super( ).evaluate( x )
    return ( rx @ self.m_P[ 1 : , : ] ) - self.m_P[ 0 , 0 ]
  # end def
  
  def threshold( self, x ):
    z = self.evaluate( x )
    #z[z <= -1] = -1
    #z[z >= 1] = 1
    #return z
    return ( z >= 1 ).astype( z.dtype ) - ( z <= -1 ).astype( z.dtype )

  ## -----------------------------------------------------------------------
  '''
  MSE-based cost function for linear regressions.
  '''
  class Cost( Base.Cost ):

    ## ---------------------------------------------------------------------
    ## Initialize an object witha zero-sized parameters vector
    ## ---------------------------------------------------------------------
    def __init__( self, model, X, y, batch_size = 0 ):
      super( ).__init__( model, X, y, batch_size )
      # self.m_XtX = ( X.T @ X ) / float( X.shape[ 0 ] )
      # self.m_mX = numpy.matrix( X.mean( axis = 0 ) )
      # self.m_mY = y.mean( )
      # self.m_XhY = numpy.matrix( numpy.multiply( X, y ).mean( axis = 0 ) ).T
    # end def

    ## ---------------------------------------------------------------------
    ## Evaluate cost with gradient (if needed)
    ## ---------------------------------------------------------------------
    def _evaluate( self, samples, need_gradient = False ):
      #print("**********")
      X = samples[ 0 ]
      Y = samples[ 1 ]
      num_x = float(X.shape[ 0 ])
      num_param = self.m_Model.numberOfParameters( )
      
      x_evaluate= numpy.multiply(Y, self.m_Model.evaluate( X ))

      x_ev1 = numpy.where(x_evaluate < 1)
      x_filtro = X[x_ev1[0]]
      y_filtro = Y[x_ev1[0]]

      
      cost_bisagra = 1-(numpy.multiply(y_filtro,self.m_Model.evaluate( x_filtro )))

      if(cost_bisagra.shape[0] ==0):
        J=0
      else:
        J = cost_bisagra.mean()
      
      if need_gradient:
        g = numpy.zeros( self.m_Model.parameters( ).shape )
        b = g[ 0 , 0 ]

          
        g[ 0 , 0 ] = (y_filtro * b).sum()/ num_x

        a=numpy.multiply( X, Y )
        b=a[ x_ev1[0]]
        c=b.sum( axis = 0 )
        d=c.reshape( num_param - 1, 1 )
        e=-d/ num_x

        g[ 1 : , : ] =e

        
        return [ J, g ]
      else:
        return [ J, None ]
      # end if
    # end def
  # end class
# end class

## eof - $RCSfile$
