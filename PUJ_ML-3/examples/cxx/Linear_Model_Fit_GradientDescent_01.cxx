// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <random>
#include <PUJ_ML/Model/Linear.h>
#include <PUJ_ML/Optimizer/GradientDescent.h>

using TScalar = double;
using TModel  = PUJ_ML::Model::Linear< TScalar >;
using TOptimizer = PUJ_ML::Optimizer::GradientDescent< TScalar >;

int main( int argc, char** argv )
{
  // Some parameters to generate sample data
  unsigned long long M = 10;
  TScalar minX = -100;
  TScalar maxX = 100;

  // Some model
  TModel dm( { 0.3, 2, 0.5 } );
  unsigned long long N = dm.numberOfParameters( );

  // Some random data
  std::random_device rd;
  std::mt19937 gen( rd( ) );
  std::uniform_real_distribution< TScalar > dis( minX, maxX );
  TModel::TMatrix X =
    TModel::TMatrix::Zero( M, dm.inputSize( ) ).unaryExpr(
      [&]( const TScalar& z ) -> TScalar { return( dis( gen ) ); }
      );
  auto Y = dm.evaluate( X );

  // Model to be optimized
  TModel om( std::vector< TScalar >( N, 0 ) );

  // Cost
  TModel::Cost cost( &om, X, Y );

  // Optimizer
  auto opt = TOptimizer( &cost );
  opt.setLearningRate( 1e-4 );
  opt.Fit( );

  // Show results
  std::cout << "******************************" << std::endl;
  std::cout << "* Original model  : " << dm << std::endl;
  std::cout << "* Optimized model : " << om << std::endl;
  std::cout << "******************************" << std::endl;

  return( EXIT_SUCCESS );
}

// eof - $RCSfile$

