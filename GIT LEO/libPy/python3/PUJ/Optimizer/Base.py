## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

class Base:

  '''
  '''
  m_Cost = None
  m_Alpha = 1e-2
  m_Lambda = 0
  m_Epsilon = 1e-6
  m_RegularizationType = 'ridge'
  m_NumberOfIterations = 1000
  m_RealIterations = 0
  m_NumberOfDebugIterations = 10
  m_DebugFunction = None

  '''
  '''
  def __init__( self, cost ):
    self.m_Cost = cost
  # end def

  def setLearningRate( self, a ):
    self.m_Alpha = a
  # end def

  def setLambda( self, l ):
    self.m_Lambda = l
  # end def

  def setRegularizationToRidge( self ):
    self.m_RegularizationType = 'ridge'
  # end def

  def setRegularizationToLASSO( self ):
    self.m_RegularizationType = 'LASSO'
  # end def

  def setEpsilon( self, e ):
    self.m_Epsilon = e
  # end def

  def setNumberOfIterations( self, i ):
    self.m_NumberOfIterations = i
  # end def

  def setNumberOfDebugIterations( self, i ):
    self.m_NumberOfDebugIterations = i
  # end def

  def setDebugFunction( self, f ):
    self.m_DebugFunction = f
  # end def

  def realIterations( self ):
    return self.m_RealIterations
  # end def

  # def regularize( self ):
  #   J = float( 0 )
  #   g = numpy.zeros( self.m_Cost.m_Model.parameters( ).shape )
  #   if self.m_Lambda != 0:
  #     if self.m_RegularizationType == 'ridge':
  #       g = self.m_Cost.m_Model.parameters( )
  #       J = numpy.power( g, 2 ).sum( )
  #       g *= 2
  #     elif self.m_RegularizationType == 'LASSO':
  #       J = numpy.absolute( self.m_Cost.m_Model.parameters( ) ).sum( )
  #       g  = ( self.m_Cost.m_Model.parameters( ) > 0 ).astype( g.dtype )
  #       g -= ( self.m_Cost.m_Model.parameters( ) < 0 ).astype( g.dtype )
  #     # end if
  #   # end if
  #   return [ J * self.m_Lambda, g * self.m_Lambda ]
  # # end def
  
  def evaluate( self, batch_id ):
    [ J, g ] = self.m_Cost.evaluate( batch_id , need_gradient = True )
    if self.m_Lambda != 0:
      if self.m_RegularizationType == 'ridge':
        gr = self.m_Cost.model( ).parameters( ).copy( )
        J += numpy.power( g, 2 ).sum( ) * self.m_Lambda
        g += gr * ( 2 * self.m_Lambda )
      elif self.m_RegularizationType == 'LASSO':
        Jr = numpy.absolute( self.m_Cost.model( ).parameters( ) ).sum( )
        gr  = ( self.m_Cost.model( ).parameters( ) > 0 ).astype( g.dtype )
        gr -= ( self.m_Cost.model( ).parameters( ) < 0 ).astype( g.dtype )
        J += Jr * self.m_Lambda
        g += gr * self.m_Lambda
      # end if
    # end if
    return [ J, g ]
  # end def

  def Fit( self ):
    pass
  # end def

## eof - $RCSfile$
