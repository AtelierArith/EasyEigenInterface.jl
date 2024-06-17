#include <jlcxx/jlcxx.hpp>

#include <Eigen/Dense>


namespace jleigen
{
  struct WrapEigenMatrix
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

  struct WrapEigenVector
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
}

namespace jlcxx
{

template<typename T>
struct BuildParameterList<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
{
  typedef ParameterList<T> type;
};

template<typename T>
struct BuildParameterList<Eigen::Matrix<T, Eigen::Dynamic, 1>>
{
  typedef ParameterList<T> type;
};

}

Eigen::MatrixXd example1(Eigen::MatrixXd x){
  return 3 * x;
}

JLCXX_MODULE easy_eigen_interface(jlcxx::Module& mod)
{
  using jlcxx::Parametric;
  using jlcxx::TypeVar;

  mod.add_type<Parametric<TypeVar<1>>>("EigenMatrix", jlcxx::julia_type("AbstractEigenMatrix"))
    .apply<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(jleigen::WrapEigenMatrix());
  mod.add_type<Parametric<TypeVar<1>>>("EigenVector", jlcxx::julia_type("AbstractEigenVector"))
    .apply<Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::Matrix<float, Eigen::Dynamic, 1>>(jleigen::WrapEigenVector());

  mod.method("example1", &example1);
}