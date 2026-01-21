#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <omp.h>
#include "BSP-OT_header_only.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Eigen;
using namespace BSPOT;

int num_threads = 0;
void set_num_threads(int n) { num_threads = n; }

template <int dim>
SparseMatrix<scalar> compute_coupling_dim(const nb::DRef<const Matrix<scalar,-1,-1>> &A,
                                          const nb::DRef<const Eigen::Vector<scalar,-1>> &mu,
                                          const nb::DRef<const Matrix<scalar,-1,-1>> &B,
                                          const nb::DRef<const Eigen::Vector<scalar,-1>> &nu) {
    const Points<dim> A_fixed = A;
    const Points<dim> B_fixed = B;
    const Atoms mu_atoms = FromMass(mu);
    const Atoms nu_atoms = FromMass(nu);
    GeneralBSPMatching<dim> BSP(A_fixed, mu_atoms, B_fixed, nu_atoms);
    return BSP.computeCoupling();
}

SparseMatrix<scalar> compute_coupling(const nb::DRef<const Matrix<scalar,-1,-1>> &A,
                                      const nb::DRef<const Eigen::Vector<scalar,-1>> &mu,
                                      const nb::DRef<const Matrix<scalar,-1,-1>> &B,
                                      const nb::DRef<const Eigen::Vector<scalar,-1>> &nu) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    if (A.rows() != B.rows()) {
        throw std::runtime_error("Source and target points must have the same dimension");
    }
    if (mu.size() != A.cols()) {
        throw std::runtime_error("Size of mu must be equal to number of source points");
    }
    if (nu.size() != B.cols()) {
        throw std::runtime_error("Size of nu must be equal to number of target points");
    }
    switch (A.rows()) {
    case 2:
        return compute_coupling_dim<2>(A, mu, B, nu);
    case 3:
        return compute_coupling_dim<3>(A, mu, B, nu);
    case 4:
        return compute_coupling_dim<4>(A, mu, B, nu);
    case 5:
        return compute_coupling_dim<5>(A, mu, B, nu);
    case 6:
        return compute_coupling_dim<6>(A, mu, B, nu);
    default:
        throw std::runtime_error("Dimension higher than 6 is not supported");
    }
}

template <int dim>
VectorXi compute_matching_dim(const nb::DRef<const Matrix<scalar,-1,-1>> &A,
                              const nb::DRef<const Matrix<scalar,-1,-1>> &B,
                              int num_plans,
                              bool orthogonal,
                              bool gaussian) {
    const Points<dim> A_fixed = A;
    const Points<dim> B_fixed = B;
    const cost_function cost = [&A_fixed, &B_fixed](size_t i, size_t j) {
        return (A_fixed.col(i) - B_fixed.col(j)).squaredNorm();
    };
    std::vector<BijectiveMatching> plans(num_plans);
    #pragma omp parallel for
    for (int i = 0; i < num_plans; i++) {
        BijectiveBSPMatching<dim> BSP(A_fixed, B_fixed);
        if (orthogonal) {
            const Matrix<scalar,dim,dim> Q = sampleUnitGaussianMat(dim, dim).fullPivHouseholderQr().matrixQ();
            plans[i] = BSP.computeOrthogonalMatching(Q);
        } else if (gaussian) {
            plans[i] = BSP.computeGaussianMatching();
        } else {
            plans[i] = BSP.computeMatching();
        }
    }
    BijectiveMatching T{};
    BijectiveMatching result = MergePlans(plans, cost, T);
    return Map<const VectorXi>(result.getPlan().data(), result.getPlan().size());
}

VectorXi compute_matching(const nb::DRef<const Matrix<scalar,-1,-1>> &A,
                          const nb::DRef<const Matrix<scalar,-1,-1>> &B,
                          int num_plans = 16,
                          bool orthogonal = false,
                          bool gaussian = false) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    if (A.rows() != B.rows()) {
        throw std::runtime_error("Source and target points must have the same dimension");
    }
    if (A.cols() != B.cols()) {
        throw std::runtime_error("Source and target point clouds must have the same number of points");
    }
    if (orthogonal && gaussian) {
        throw std::runtime_error("Only one of orthogonal and gaussian options can be true");
    }
    switch (A.rows()) {
    case 2:
        return compute_matching_dim<2>(A, B, num_plans, orthogonal, gaussian);
    case 3:
        return compute_matching_dim<3>(A, B, num_plans, orthogonal, gaussian);
    case 4:
        return compute_matching_dim<4>(A, B, num_plans, orthogonal, gaussian);
    case 5:
        return compute_matching_dim<5>(A, B, num_plans, orthogonal, gaussian);
    case 6:
        return compute_matching_dim<6>(A, B, num_plans, orthogonal, gaussian);
    default:
        throw std::runtime_error("Dimension higher than 6 is not supported");
    }
}

template <int dim>
Matrix<scalar,-1,-1> compute_transport_gradient_dim(const nb::DRef<const Matrix<scalar,-1,-1>> &A,
                                                    const nb::DRef<const Eigen::Vector<scalar,-1>> &mu,
                                                    const nb::DRef<const Matrix<scalar,-1,-1>> &B,
                                                    const nb::DRef<const Eigen::Vector<scalar,-1>> &nu,
                                                    int num_plans) {
    Points<dim> Grad = Points<dim>::Zero(dim, A.cols());
    const Points<dim> A_fixed = A;
    const Points<dim> B_fixed = B;
    const Atoms mu_atoms = FromMass(mu);
    const Atoms nu_atoms = FromMass(nu);
    #pragma omp parallel for
    for (int i = 0; i < num_plans; i++) {
        GeneralBSPMatching<dim> BSP(A_fixed, mu_atoms, B_fixed, nu_atoms);
        const Points<dim> Grad_i = BSP.computeTransportGradient();
        #pragma omp critical
        {
            Grad += Grad_i / num_plans;
        }
    }
    return Grad;
}

Matrix<scalar,-1,-1> compute_transport_gradient(const nb::DRef<const Matrix<scalar,-1,-1>> &A,
                                                const nb::DRef<const Eigen::Vector<scalar,-1>> &mu,
                                                const nb::DRef<const Matrix<scalar,-1,-1>> &B,
                                                const nb::DRef<const Eigen::Vector<scalar,-1>> &nu,
                                                int num_plans) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    if (A.rows() != B.rows()) {
        throw std::runtime_error("Source and target points must have the same dimension");
    }
    if (A.cols() != mu.size()) {
        throw std::runtime_error("Source points and source masses must have the same size");
    }
    if (B.cols() != nu.size()) {
        throw std::runtime_error("Target points and target masses must have the same size");
    }
    switch (A.rows()) {
    case 2:
        return compute_transport_gradient_dim<2>(A, mu, B, nu, num_plans);
    case 3:
        return compute_transport_gradient_dim<3>(A, mu, B, nu, num_plans);
    case 4:
        return compute_transport_gradient_dim<4>(A, mu, B, nu, num_plans);
    case 5:
        return compute_transport_gradient_dim<5>(A, mu, B, nu, num_plans);
    case 6:
        return compute_transport_gradient_dim<6>(A, mu, B, nu, num_plans);
    default:
        throw std::runtime_error("Dimension higher than 6 is not supported");
    }
}

template <int dim>
VectorXi compute_partial_matching_dim(const nb::DRef<const Matrix<scalar,-1,-1>> &A,
                                      const nb::DRef<const Matrix<scalar,-1,-1>> &B,
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
    for (int i = 0; i < num_plans; i++) {
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

VectorXi compute_partial_matching(const nb::DRef<const Matrix<scalar,-1,-1>> &A,
                                  const nb::DRef<const Matrix<scalar,-1,-1>> &B,
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
        return compute_partial_matching_dim<2>(A, B, num_plans, orthogonal);
    case 3:
        return compute_partial_matching_dim<3>(A, B, num_plans, orthogonal);
    case 4:
        return compute_partial_matching_dim<4>(A, B, num_plans, orthogonal);
    case 5:
        return compute_partial_matching_dim<5>(A, B, num_plans, orthogonal);
    case 6:
        return compute_partial_matching_dim<6>(A, B, num_plans, orthogonal);
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
    m.def("compute_matching", &compute_matching, "A"_a, "B"_a, "num_plans"_a = 16, "orthogonal"_a = false, "gaussian"_a = false,
          "Computes matching between two point clouds in 2<=d<=6 dimension.");
    m.def("compute_partial_matching", &compute_partial_matching, "A"_a, "B"_a, "num_plans"_a = 16, "orthogonal"_a = false,
          "Computes partial matching between two point clouds in 2<=d<=6 dimension.");
    m.def("compute_coupling", &compute_coupling, "A"_a, "mu"_a, "B"_a, "nu"_a,
          "Computes coupling between two point clouds in 2<=d<=6 dimension.");
    m.def("compute_transport_gradient", &compute_transport_gradient, "A"_a, "mu"_a, "B"_a, "nu"_a, "num_plans"_a = 16,
          "Computes transport gradient between two point clouds in 2<=d<=6 dimension.");
    m.def("set_num_threads", &set_num_threads, "n"_a,
          "Sets the number of threads used in computation. If n<=0, uses default number of threads.");
    m.doc() = "A Python binding to BSP-OT";
}
