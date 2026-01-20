#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include "BSP-OT_header_only.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Eigen;
using namespace BSPOT;

VectorXi compute_partial_2d(const Matrix2Xd &A,
                            const Matrix2Xd &B,
                            int num_plans = 16,
                            bool orthogonal = false) {
    const cost_function cost = [&A, &B](size_t i, size_t j) {
        return (A.col(i) - B.col(j)).squaredNorm();
    };
    PartialBSPMatching<2> BSP(A, B, cost);
    std::vector<InjectiveMatching> plans(num_plans);
#pragma omp parallel for
    for (int i = 0; i < num_plans; i++){
        if (orthogonal) {
            const Matrix2Xd Q = sampleUnitGaussianMat(2, 2).fullPivHouseholderQr().matrixQ();
            plans[i] = BSP.computePartialMatching(Q, false);
        } else {
            plans[i] = BSP.computePartialMatching();
        }
    }
    InjectiveMatching T{};
    InjectiveMatching result = MergePlans(plans, cost, T);
    return Map<const VectorXi>(result.getPlan().data(), result.getPlan().size());
}

NB_MODULE(pybspot, m) {
    m.def("compute_partial_2d", &compute_partial_2d, "A"_a, "B"_a, "num_plans"_a = 16, "orthogonal"_a = false,
          "Computes partial matching in 2D ");
    m.doc() = "A Python binding to BSP-OT";
}
