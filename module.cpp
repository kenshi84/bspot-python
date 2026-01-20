#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <omp.h>
#include "BSP-OT_header_only.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Eigen;
using namespace BSPOT;

int num_threads = 0;
void set_num_threads(int n) { num_threads = n; }

template<int dim>
VectorXi compute_partial_dim(const Matrix<scalar,-1,-1> &A,
                             const Matrix<scalar,-1,-1> &B,
                             int num_plans,
                             bool orthogonal) {
    const Points<dim> A_fixed = A;
    const Points<dim> B_fixed = B;
    const cost_function cost = [&A_fixed, &B_fixed](size_t i, size_t j) {
        return (A_fixed.col(i) - B_fixed.col(j)).squaredNorm();
    };
    PartialBSPMatching<dim> BSP(A_fixed, B_fixed, cost);
    std::vector<InjectiveMatching> plans(num_plans);
    #pragma omp parallel for
    for (int i = 0; i < num_plans; i++){
        if (orthogonal) {
            const Matrix<scalar,dim,dim> Q = sampleUnitGaussianMat(dim, dim).fullPivHouseholderQr().matrixQ();
            plans[i] = BSP.computePartialMatching(Q, false);
        } else {
            plans[i] = BSP.computePartialMatching();
        }
    }
    InjectiveMatching T{};
    InjectiveMatching result = MergePlans(plans, cost, T);
    return Map<const VectorXi>(result.getPlan().data(), result.getPlan().size());
}

VectorXi compute_partial(const Matrix<scalar,-1,-1> &A,
                         const Matrix<scalar,-1,-1> &B,
                         int num_plans = 16,
                         bool orthogonal = false) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    if (A.rows() != B.rows()) {
        throw std::runtime_error("Source and target points must have the same dimension");
    }
    if (A.cols() > B.cols()) {
        throw std::runtime_error("Number of source points must be less than or equal to number of target points");
    }
    switch (A.rows()) {
    case 2:
        return compute_partial_dim<2>(A, B, num_plans, orthogonal);
    case 3:
        return compute_partial_dim<3>(A, B, num_plans, orthogonal);
    case 4:
        return compute_partial_dim<4>(A, B, num_plans, orthogonal);
    case 5:
        return compute_partial_dim<5>(A, B, num_plans, orthogonal);
    case 6:
        return compute_partial_dim<6>(A, B, num_plans, orthogonal);
    default:
        throw std::runtime_error("Dimension higher than 6 is not supported");
    }
}

#ifndef BSPOT_SINGLE_PRECISION
NB_MODULE(pybspot, m)
#else
NB_MODULE(pybspot_f32, m)
#endif
{
    m.def("compute_partial", &compute_partial, "A"_a, "B"_a, "num_plans"_a = 16, "orthogonal"_a = false,
          "Computes partial matching between two point clouds in 2<=d<=6 dimension.");
    m.def("set_num_threads", &set_num_threads, "n"_a,
          "Sets the number of threads used in computation. If n<=0, uses default number of threads.");
    m.doc() = "A Python binding to BSP-OT";
}
