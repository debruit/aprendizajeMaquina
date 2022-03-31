// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <PUJ_ML/Model/Logistic.h>

using TScalar = double;
using TModel  = PUJ_ML::Model::Logistic< TScalar >;

int main( int argc, char** argv )
{
  // Curly initialization
  TModel m1( { 10, 1, 2, 3 } );
  std::cout << m1 << std::endl;
  TModel::TRow x1( m1.inputSize( ) );
  x1 << -1, -2, -3;
  std::cout << m1.evaluate( x1 ) << std::endl;
  std::cout << m1.threshold( x1 ) << std::endl;

  // Stream-based initialization
  TModel m2;
  m2 << 4 << 10 << 1 << 2 << 3;
  std::cout << m2 << std::endl;
  TModel::TRow x2( m2.inputSize( ) );
  x2 << -1, -2, -3;
  std::cout << m2.evaluate( x2 ) << std::endl;
  std::cout << m2.threshold( x2 ) << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
