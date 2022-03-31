// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <PUJ_ML/Helpers/CSVIO.h>
#include <PUJ_ML/Model/Logistic.h>
#include <PUJ_ML/Optimizer/GradientDescent.h>

using TScalar = double;
using TModel  = PUJ_ML::Model::Logistic< TScalar >;
using TOptimizer = PUJ_ML::Optimizer::GradientDescent< TScalar >;

int main( int argc, char** argv )
{
  if( argc < 3 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ] << " input.csv learning_rate"
      << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string input = argv[ 1 ];
  TScalar learning_rate = std::atof( argv[ 2 ] );

  // Read data
  auto D = PUJ_ML::Helpers::CSVIO::Read< TModel::TMatrix >( input );
  TModel::TMatrix X = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );
  TModel::TMatrix Y = D.block( 0, D.cols( ) - 1, D.rows( ), 1 );

  // Model to be optimized
  TModel om( std::vector< TScalar >( X.cols( ) + 1, 0 ) );

  // Cost
  TModel::Cost cost( &om, X, Y );

  // Optimizer
  auto opt = TOptimizer( &cost );
  opt.setLearningRate( learning_rate );
  opt.Fit( );

  // Show results
  std::cout << "******************************" << std::endl;
  std::cout << "* Optimized model : " << om << std::endl;
  std::cout << "******************************" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$

