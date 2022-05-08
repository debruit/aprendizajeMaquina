## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

set(
  _src
  Linear_Model_01
  Linear_Model_Fit_ADAM_01
  Linear_Model_Fit_GradientDescent_01
  Logistic_Model_01
  Logistic_Model_Fit_GradientDescent_01
  Logistic_Model_Fit_ADAM_01
  )

include_directories(
  ${PROJECT_BINARY_DIR}/lib/cxx
  ${PROJECT_SOURCE_DIR}/lib/cxx
  )

foreach(_e ${_src})
  add_executable(PUJ_ML_example_${_e} ${_e}.cxx)
  target_link_libraries(PUJ_ML_example_${_e} PUJ_ML)
endforeach()

## eof - $RCSfile$
