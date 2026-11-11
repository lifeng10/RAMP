#pragma once
#include "common.hpp"
#include <rabitqlib/quantization/rabitq.hpp>
#include <rabitqlib/quantization/pack_excode.hpp>

namespace ramp {
namespace rq = rabitqlib;

struct MetricCodes {
    U n, dim, bits;
    size_t rowbytes;
    std::vector<uint8_t> packed;
    std::vector<float> add, rescale;
    std::vector<double> residual, normerr;
    rq::ex_ipfunc ip;

    MetricCodes(const std::vector<float>& z, U d, U b)
        : n(d ? z.size() / d : 0), dim(d), bits(b), rowbytes(size_t(d) * b / 8),
          packed(n * rowbytes), add(n), rescale(n), residual(n), normerr(n), ip(nullptr) {
        check(dim > 0 && z.size() % dim == 0, "metric code dimensions");
        check(bits >= 1 && bits <= 8, "metric code bits in [1,8]");
        const U block = (bits == 1 || bits == 4 || bits == 8) ? 16 : 64;
        check(dim % block == 0, "metric code dimension for packed kernel");
        ip = rq::select_excode_ipfunc(bits);
        auto start = Clock::now();

#pragma omp parallel for schedule(dynamic, 64) num_threads(8)

        for (int i = 0; i < int(n); ++i) {
            std::vector<uint8_t> code(dim);
            float extra = 0;
            const double own = normsq(z.data() + size_t(i) * dim, dim);
            if (own > 0) {
                rq::quant::quantize_full_single(
                    z.data() + size_t(i) * dim,
                    dim,
                    bits,
                    code.data(),
                    add[i],
                    rescale[i],
                    extra,
                    rq::METRIC_L2,
                    rq::quant::RabitqConfig());
            } else {
                add[i] = rescale[i] = 0;
            }
            rq::quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
                code.data(), packed.data() + size_t(i) * rowbytes, dim, bits);
            double e = 0;
            for (U j = 0; j < dim; ++j) {
                double original = z[size_t(i) * dim + j];
                double reconstructed =
                    -.5 * rescale[i] * (double(code[j]) - ((1U << bits) - 1) * .5);
                e += (original - reconstructed) * (original - reconstructed);
            }
            residual[i] = std::sqrt(e);
            normerr[i] = std::abs(own - add[i]);
            if (i < 100) {
                double sum = 0;
                auto q = z.data();
                for (U j = 0; j < dim; ++j) {
                    sum += double(q[j]) * code[j];
                }
                check(
                    std::abs(sum - ip(q, packed.data() + size_t(i) * rowbytes, dim)) <
                        1e-4 * (1 + std::abs(sum)),
                    "metric packed dot");
                double qsum = 0, refdot = 0, exact = 0;
                for (U j = 0; j < dim; ++j) {
                    qsum += q[j];
                    double v = double(q[j]) - z[size_t(i) * dim + j];
                    exact += v * v;
                    refdot += double(q[j]) * (double(code[j]) - double((1U << bits) - 1) * .5);
                }
                double qn = normsq(q, dim);
                double reference = add[i] + qn + rescale[i] * refdot;
                double actual = score(i, q, qn, -double((1U << bits) - 1) * .5 * qsum);
                check(
                    std::abs(reference - actual) < 1e-4 * (1 + std::abs(reference)),
                    "complete metric estimator vs independent coordinate formula");
                check(
                    std::abs(actual - exact) <= error(i, qn) + 1e-4,
                    "metric estimator residual bound");
            }
        }
        std::cerr << "metric codes bits=" << bits << " seconds=" << ms(start) / 1000 << std::endl;
    }

    float score(U id, const float* q, double qnorm, double offset) const {
        return float(
            add[id] + qnorm +
            rescale[id] * (ip(q, packed.data() + size_t(id) * rowbytes, dim) + offset));
    }

    // Error bounds are construction checks and are discarded from the server index.
    void discard_error_bounds() {
        std::vector<double>().swap(residual);
        std::vector<double>().swap(normerr);
    }

    double error(U id, double qnorm) const {
        check(
            id < normerr.size() && id < residual.size(), "diagnostic error bounds were discarded");
        return normerr[id] + 2 * std::sqrt(qnorm) * residual[id] + 1e-5;
    }
};
} // namespace ramp
