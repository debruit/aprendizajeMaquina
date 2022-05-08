// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <PUJ_ML/Helpers/CSVIO.h>
#include <PUJ_ML/Model/Logistic.h>
#include <PUJ_ML/Optimizer/ADAM.h>

using TScalar = double;
using TModel  = PUJ_ML::Model::Logistic< TScalar >;
using TOptimizer = PUJ_ML::Optimizer::ADAM< TScalar >;

int main( int argc, char** argv )
{
  if( argc < 2 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ] << " input.csv"
      << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string input = argv[ 1 ];

  // Read data
  auto D = PUJ_ML::Helpers::CSVIO::Read< TModel::TMatrix >( input );
  TModel::TMatrix X = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );
  TModel::TMatrix Y = D.block( 0, D.cols( ) - 1, D.rows( ), 1 );

  auto C = X.rowwise( ) - X.colwise( ).mean( );
  TModel::TMatrix S = ( C.transpose( ) * C ).array( ) / TScalar( X.rows( ) - 1 );
  TModel::TRow d( X.cols( ) );
  for( unsigned int i = 0; i < X.cols( ); ++i )
    d( i ) = S( i, i );
  X = C.array( ).rowwise( ) / d.array( );

  // Model to be optimized
  TModel om( std::vector< TScalar >( X.cols( ) + 1, 0 ) );

  // Cost
  TModel::Cost cost( &om, X, Y );

  // Optimizer
  auto opt = TOptimizer( &cost );
  opt.setMaximumNumberOfIterations( 10000 );
  opt.setNumberOfDebugIterations( 2000 );
  opt.Fit( );

  // Show results
  std::cout << "******************************" << std::endl;
  std::cout << "* Optimized model : " << om << std::endl;
  std::cout << "******************************" << std::endl;

  TModel::TMatrix Yreal( Y.rows( ), 2 );
  Yreal.block( 0, 0, Y.rows( ), 1 ) = Y;
  Yreal.block( 0, 1, Y.rows( ), 1 ) = 1 - Y.array( );

  TModel::TMatrix Z = om.threshold( X );
  TModel::TMatrix Yest( Z.rows( ), 2 );
  Yest.block( 0, 0, Z.rows( ), 1 ) = Z;
  Yest.block( 0, 1, Z.rows( ), 1 ) = 1 - Z.array( );

  TModel::TMatrix K = Yreal.transpose( ) * Yest;
  std::cout << "******** confussion ********" << std::endl;
  std::cout << K << std::endl;
  std::cout << "****************************" << std::endl;
  std::cout << "Accuracy: " << 100 * ( K( 0, 0 ) + K( 1, 1 ) ) / K.sum( ) << "%" << std::endl;
  std::cout << "****************************" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$

