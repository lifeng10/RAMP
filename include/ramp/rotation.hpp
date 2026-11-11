#pragma once
#include "common.hpp"

namespace ramp {
std::vector<float> rotated(const float* x, U n) {
    check(n > 0, "positive rotation dimension");
    std::vector<float> a(x, x + n);
    // The sign pattern is public and fixed, not fresh privacy randomness.
    static thread_local std::unordered_map<U, std::vector<uint8_t>> patterns;
    auto it = patterns.find(n);
    if (it == patterns.end()) {
        std::mt19937 rng(817263);
        std::vector<uint8_t> signs(n);
        for (auto& v : signs) {
            v = rng() & 1;
        }
        it = patterns.emplace(n, std::move(signs)).first;
    }
    for (U j = 0; j < n; ++j) {
        if (it->second[j]) {
            a[j] = -a[j];
        }
    }
    // Block-diagonal signed Hadamard for arbitrary dimensions. Power-of-two dimensions use a single orthogonal block.
    for (U first = 0; first < n;) {
        U count = 1;
        while (count <= (n - first) / 2) {
            count *= 2;
        }
        for (U step = 1; step < count; step *= 2) {
            for (U base = first; base < first + count; base += 2 * step) {
                for (U j = 0; j < step; ++j) {
                    float u = a[base + j], v = a[base + j + step];
                    a[base + j] = u + v;
                    a[base + j + step] = u - v;
                }
            }
        }
        for (U j = first; j < first + count; ++j) {
            a[j] /= std::sqrt(float(count));
        }
        first += count;
    }
    return a;
}
} // namespace ramp
