// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <deque>
#include <iostream>
#include <vector>
#include <PUJ_ML/Model/Linear.h>

using TScalar = double;
using TModel  = PUJ_ML::Model::Linear< TScalar >;

int main( int argc, char** argv )
{
  // Curly initialization
  TModel m1;
  m1.setParameters( { 10, 1, 2, 3 } );
  std::cout << m1 << std::endl;

  /* TODO
     TModel m1( { 10, 1, 2, 3 } );
     TModel::TRow x1( m1.inputSize( ) );
     x1 << -1, -2, -3;
     std::cout << m1.evaluate( x1 ) << std::endl;
  */

  // Stream-based initialization
  TModel m2;
  m2 << 10 << 1 << 2 << 3 << TModel::end;
  std::cout << m2 << std::endl;
  /* TODO
     TModel::TRow x2( m2.inputSize( ) );
     x2 << -1, -2, -3;
     std::cout << m2.evaluate( x2 ) << std::endl;
  */

  // Eigen::Row based initalization
  Eigen::Matrix< float, 1, Eigen::Dynamic > r( 4 );
  r << 10, 1, 2, 3;
  TModel m3;
  m3.setParameters( r );
  std::cout << m3 << std::endl;

  // Eigen::Column based initalization
  TModel::TColumn c( 4 );
  c << 10, 1, 2, 3;
  TModel m4;
  m4.setParameters( c );
  std::cout << m4 << std::endl;

  // Eigen::Matrix based initalization
  TModel::TMatrix m( 2, 2 );
  m << 10, 1, 2, 3;
  TModel m5;
  m5.setParameters( m );
  std::cout << m5 << std::endl;

  // std::vector based initalization
  std::vector< TScalar > v;
  v.push_back( 10 );
  v.push_back( 1 );
  v.push_back( 2 );
  v.push_back( 3 );
  TModel m6;
  m6.setParameters( v );
  std::cout << m6 << std::endl;

  // std::deque based initalization
  std::deque< TScalar > d;
  d.push_back( 10 );
  d.push_back( 1 );
  d.push_back( 2 );
  d.push_back( 3 );
  TModel m7;
  m7.setParameters( d );
  std::cout << m7 << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
