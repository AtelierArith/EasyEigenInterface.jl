#include <jlcxx/jlcxx.hpp>

#include <Eigen/Dense>


namespace jleigen
{
  struct WrapEigenMatrixX
  {
    template<typename TypeWrapperT>
    void operator()(TypeWrapperT&& wrapped)
    {
      using WrappedT = typename TypeWrapperT::type;
      using ScalarT = typename WrappedT::Scalar;
      wrapped.template constructor<Eigen::Index, Eigen::Index>();
      wrapped.method("cols", &WrappedT::cols);
      wrapped.method("rows", &WrappedT::rows);

      wrapped.module().set_override_module(jl_base_module);
      wrapped.method("resize!", [](WrappedT& m, int_t i, int_t j) {
        m.resize(i, j);
      });
      wrapped.module().method("getindex", [](const WrappedT& m, int_t i, int_t j) { return m(i-1,j-1); });
      wrapped.module().method("setindex!", [](WrappedT& m, ScalarT value, int_t i, int_t j) { m(i-1,j-1) = value; });

      wrapped.module().method("getindex", [](const WrappedT& m, int_t i) { return m(i-1); });
      wrapped.module().method("setindex!", [](WrappedT& m, ScalarT value, int_t i) { m(i-1) = value; });

      wrapped.module().unset_override_module();
    }
  };

  struct WrapEigenMatrixStaticSized
  {
    template<typename TypeWrapperT>
    void operator()(TypeWrapperT&& wrapped)
    {
      using WrappedT = typename TypeWrapperT::type;
      using ScalarT = typename WrappedT::Scalar;
      wrapped.method("cols", &WrappedT::cols);
      wrapped.method("rows", &WrappedT::rows);

      wrapped.module().set_override_module(jl_base_module);

      wrapped.module().method("getindex", [](const WrappedT& m, int_t i, int_t j) { return m(i-1,j-1); });
      wrapped.module().method("setindex!", [](WrappedT& m, ScalarT value, int_t i, int_t j) { m(i-1,j-1) = value; });

      wrapped.module().method("getindex", [](const WrappedT& m, int_t i) { return m(i-1); });
      wrapped.module().method("setindex!", [](WrappedT& m, ScalarT value, int_t i) { m(i-1) = value; });

      wrapped.module().unset_override_module();
    }
  };

  struct WrapEigenVectorX
  {
    template<typename TypeWrapperT>
    void operator()(TypeWrapperT&& wrapped)
    {
      using WrappedT = typename TypeWrapperT::type;
      using ScalarT = typename WrappedT::Scalar;
      wrapped.template constructor<Eigen::Index>();
      wrapped.method("cols", &WrappedT::cols);
      wrapped.method("rows", &WrappedT::rows);

      wrapped.module().set_override_module(jl_base_module);
      wrapped.module().method("getindex", [](const WrappedT& v, int_t i) { return v(i-1); });
      wrapped.module().method("setindex!", [](WrappedT& v, ScalarT value, int_t i) { v(i-1) = value; });
      wrapped.module().unset_override_module();
    }
  };

  struct WrapEigenVectorStaticSized
  {
    template<typename TypeWrapperT>
    void operator()(TypeWrapperT&& wrapped)
    {
      using WrappedT = typename TypeWrapperT::type;
      using ScalarT = typename WrappedT::Scalar;
      wrapped.method("cols", &WrappedT::cols);
      wrapped.method("rows", &WrappedT::rows);

      wrapped.module().set_override_module(jl_base_module);
      wrapped.module().method("getindex", [](const WrappedT& v, int_t i) { return v(i-1); });
      wrapped.module().method("setindex!", [](WrappedT& v, ScalarT value, int_t i) { v(i-1) = value; });
      wrapped.module().unset_override_module();
    }
  };
}

namespace jlcxx
{

// MatrixXd
template<typename T>
struct BuildParameterList<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
{
  typedef ParameterList<T> type;
};

// VectorXd
template<typename T>
struct BuildParameterList<Eigen::Matrix<T, Eigen::Dynamic, 1>>
{
  typedef ParameterList<T> type;
};

// Matrix
template<typename T>
struct BuildParameterList<Eigen::Matrix<T, 2, 2>>
{
  typedef ParameterList<T> type;
};

template<typename T>
struct BuildParameterList<Eigen::Matrix<T, 3, 3>>
{
  typedef ParameterList<T> type;
};

template<typename T>
struct BuildParameterList<Eigen::Matrix<T, 4, 4>>
{
  typedef ParameterList<T> type;
};

// Vector
template<typename T>
struct BuildParameterList<Eigen::Matrix<T, 2, 1>>
{
  typedef ParameterList<T> type;
};

template<typename T>
struct BuildParameterList<Eigen::Matrix<T, 3, 1>>
{
  typedef ParameterList<T> type;
};

template<typename T>
struct BuildParameterList<Eigen::Matrix<T, 4, 1>>
{
  typedef ParameterList<T> type;
};
}

Eigen::MatrixXd example1(Eigen::MatrixXd x){
  return 3 * x;
}

Eigen::Matrix2d example2(Eigen::Matrix2d x){
  return 3 * x;
}

JLCXX_MODULE easy_eigen_interface(jlcxx::Module& mod)
{
  using jlcxx::Parametric;
  using jlcxx::TypeVar;

  // Matrix
  mod.add_type<Parametric<TypeVar<1>>>("EigenMatrixX", jlcxx::julia_type("AbstractEigenMatrix"))
    .apply<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(jleigen::WrapEigenMatrixX());

  // Vectoor
  mod.add_type<Parametric<TypeVar<1>>>("EigenVectorX", jlcxx::julia_type("AbstractEigenVector"))
    .apply<Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::Matrix<float, Eigen::Dynamic, 1>>(jleigen::WrapEigenVectorX());

  #define ADD_EIGEN_MATRIX_TYPE(N) \
    mod.add_type<Parametric<TypeVar<1>>>("EigenMatrix" #N, jlcxx::julia_type("AbstractEigenMatrix")) \
      .apply<Eigen::Matrix<double, N, N>, Eigen::Matrix<float, N, N>>(jleigen::WrapEigenMatrixStaticSized());

  #define ADD_EIGEN_VECTOR_TYPE(N) \
    mod.add_type<Parametric<TypeVar<1>>>("EigenVector" #N, jlcxx::julia_type("AbstractEigenVector")) \
      .apply<Eigen::Matrix<double, N, 1>, Eigen::Matrix<float, N, 1>>(jleigen::WrapEigenVectorStaticSized());

  ADD_EIGEN_MATRIX_TYPE(2)
  ADD_EIGEN_MATRIX_TYPE(3)
  ADD_EIGEN_MATRIX_TYPE(4)

  ADD_EIGEN_VECTOR_TYPE(2)
  ADD_EIGEN_VECTOR_TYPE(3)
  ADD_EIGEN_VECTOR_TYPE(4)

  mod.method("example1", &example1);
  mod.method("example2", &example2);
}