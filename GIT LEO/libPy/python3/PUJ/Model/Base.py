## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import io, math, numpy

'''
'''
class Base:

  ## -----------------------------------------------------------------------
  '''
  Paramateres for this model.
  @type numpy.matrix (nx1 column vector)
  '''
  m_P = None

  ## -----------------------------------------------------------------------
  ## Initialize an object witha zero-sized parameters vector
  ## -----------------------------------------------------------------------
  def __init__( self ):
    pass
  # end if

  ## -----------------------------------------------------------------------
  ## Parameters-related methods.
  ## -----------------------------------------------------------------------
  def clear( self ):
    self.m_P = None
  # end def

  def inputSize( self ):
    if not self.m_P is None:
      return self.m_P.shape[ 0 ] - 1
    else:
      return 0
    # end if
  # end def

  def outputSize( self ):
    return 0
  # end def

  def numberOfParameters( self ):
    if not self.m_P is None:
      return self.m_P.shape[ 0 ]
    else:
      return 0
    # end if
  # end def

  def parameters( self ):
    return self.m_P
  # end def

  def setParameters( self, p ):
    m = numpy.matrix( p )
    self.m_P = numpy.reshape( m, ( m.size, 1 ) ).astype( float )
  # end def
  
  def moveParameters( self, g ):
    self.m_P += g
  # end def

  ## -----------------------------------------------------------------------
  ## Final-user methods.
  ## -----------------------------------------------------------------------
  def evaluate( self, x ):
    if self.m_P is None:
      raise ValueError( 'Parameters should be defined first!' )
    # end if

    rx = None
    if not isinstance( x, numpy.matrix ):
      rx = numpy.matrix( x )
    else:
      rx = x
    # end if

    if rx.shape[ 1 ] != self.m_P.shape[ 0 ] - 1:
      raise ValueError(
        'Input size (=' + str( rx.shape[ 1 ] ) +
        ') differs from parameters (=' + str( self.m_P.shape[ 0 ] - 1 )
        + ')'
        )
    # end if

    return rx
  # end def

  def threshold( self, x ):
    return self.evaluate( x )
  # end def

  ## -----------------------------------------------------------------------
  ## Streaming methods.
  ## -----------------------------------------------------------------------
  def __str__( self ):
    b = io.BytesIO( )
    numpy.savetxt( b, self.m_P.T, fmt = '%.3f' )
    return str( self.m_P.size ) + ' ' + \
           b.getvalue( ).decode( 'latin1' )[ 0 : -1 ]
  # end def

  ## -----------------------------------------------------------------------
  '''
  Base class for costs
  '''
  class Cost:

    ## ---------------------------------------------------------------------
    '''
    Model associated to this cost
    @type Something derived from PUJ.Model.Base
    '''
    m_Model = None
    m_X = None
    m_Y = None
    m_BSize = 0

    ## ---------------------------------------------------------------------
    ## Initialize an object witha zero-sized parameters vector
    ## ---------------------------------------------------------------------
    def __init__( self, model, X, Y, batch_size = 0 ):
      self.m_Model = model
      self.m_X = X
      self.m_Y = Y

      self.m_BSize = batch_size
      if self.m_BSize <= 0 or self.m_BSize > X.shape[ 0 ]:
        self.m_BSize = X.shape[ 0 ]
      # end if
    # end def

    def numberOfBatches( self ):
      return int( math.ceil( float( self.m_X.shape[ 0 ] ) / float( self.m_BSize ) ) )
    # end def

    def batchSize( self ):
      return self.m_BSize
    # end def

    def batch( self, bId ):
      i = self.m_BSize * bId
      j = self.m_BSize * ( bId + 1 )
      if j > self.m_X.shape[ 0 ]:
        j = self.m_X.shape[ 0 ]
      # end if
      return [ self.m_X[ i : j , : ], self.m_Y[ i : j , : ] ]
    # end def

    def shuffle( self ):
      pass
    # end def

    ## ---------------------------------------------------------------------
    ## Model access
    ## ---------------------------------------------------------------------
    def model( self ):
      return self.m_Model
    # end def

    ## ---------------------------------------------------------------------
    ## Move parameters
    ## ---------------------------------------------------------------------
    # def move( self, d ):
    #   self.m_Model.m_P += d
    # # end def
    
    
    def updateModel( self, d ):
      self.m_Model.moveParameters( d )
    # end def

    ## ---------------------------------------------------------------------
    ## Evaluate cost with gradient (if needed)
    ## ---------------------------------------------------------------------
    def evaluate( self, batch_id, need_gradient = False ):
      if batch_id >= 0 and batch_id < self.numberOfBatches( ):
        return self._evaluate( self.batch( batch_id ), need_gradient )
      else:
        return self._evaluate( [ self.m_X, self.m_Y ], need_gradient )
      # end if
    # end def

    def _evaluate( self, samples, need_gradient ):
      return [ None, None ]
    # end def

  # end class
# end class

## eof - $RCSfile$




# ## =========================================================================
# ## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
# ## =========================================================================

# import io, numpy

# '''
# '''
# class Base:

#   ## -----------------------------------------------------------------------
#   '''
#   Paramateres for this model.
#   @type numpy.matrix (nx1 column vector)
#   '''
#   m_P = None

#   ## -----------------------------------------------------------------------
#   ## Initialize an object witha zero-sized parameters vector
#   ## -----------------------------------------------------------------------
#   def __init__( self ):
#     pass
#   # end if

#   ## -----------------------------------------------------------------------
#   ## Parameters-related methods.
#   ## -----------------------------------------------------------------------
#   def parameters( self ):
#     return self.m_P
#   # end def

#   def numberOfParameters( self ):
#     if not self.m_P is None:
#       return self.m_P.shape[ 0 ]
#     else:
#       return -1
#     # end if
#   # end def

#   def setParameters( self, p ):
#     m = numpy.matrix( p )
#     self.m_P = numpy.reshape( m, ( m.size, 1 ) ).astype( float )
#   # end def

  # def moveParameters( self, g ):
  #   self.m_P += g
  # # end def

#   ## -----------------------------------------------------------------------
#   ## Streaming methods.
#   ## -----------------------------------------------------------------------
#   def __str__( self ):
#     b = io.BytesIO( )
#     numpy.savetxt( b, self.m_P.T, fmt = '%.3f' )
#     return str( self.m_P.size ) + ' ' + \
#            b.getvalue( ).decode( 'latin1' )[ 0 : -1 ]
#   # end def

#   ## -----------------------------------------------------------------------
#   ## Final-user methods.
#   ## -----------------------------------------------------------------------
#   def evaluate( self, x ):
#     if self.m_P is None:
#       raise ValueError( 'Parameters should be defined first!' )
#     # end if

#     rx = None
#     if not isinstance( x, numpy.matrix ):
#       rx = numpy.matrix( x )
#     else:
#       rx = x
#     # end if

#     if rx.shape[ 1 ] != self.m_P.shape[ 0 ] - 1:
#       raise ValueError(
#         'Input size (=' + str( rx.shape[ 1 ] ) +
#         ') differs from parameters (=' + str( self.m_P.shape[ 0 ] - 1 )
#         + ')'
#         )
#     # end if

#     return rx
#   # end def

#   def threshold( self, x ):
#     return self.evaluate( x )
#   # end def

#   ## -----------------------------------------------------------------------
#   '''
#   Base class for costs
#   '''
  
  
  
#   # class Cost:

#   #   ## ---------------------------------------------------------------------
#   #   '''
#   #   Model associated to this cost
#   #   @type Something derived from PUJ.Model.Base
#   #   '''
#   #   m_Model = None
#   #   m_X = None
#   #   m_Y = None

#   #   ## ---------------------------------------------------------------------
#   #   ## Initialize an object witha zero-sized parameters vector
#   #   ## ---------------------------------------------------------------------
#   #   def __init__( self, model, X, Y ):
#   #     self.m_Model = model
#   #     self.m_X = X
#   #     self.m_Y = Y
#   #   # end def

#   #   ## ---------------------------------------------------------------------
#   #   ## Evaluate cost with gradient (if needed)
#   #   ## ---------------------------------------------------------------------
#   #   def evaluate( self, need_gradient = False ):
#   #     return [ None, None ]
#   #   # end def

#   #   ## ---------------------------------------------------------------------
#   #   ## Model access
#   #   ## ---------------------------------------------------------------------
#   #   def model( self ):
#   #     return self.m_Model
#   #   # end def

#   #   ## ---------------------------------------------------------------------
#   #   ## Move parameters
#   #   ## ---------------------------------------------------------------------
#   #   def updateModel( self, d ):
#   #     self.m_Model.moveParameters( d )
#   #   # end def
#   # # end class
# # end class

# ## eof - $RCSfile$
