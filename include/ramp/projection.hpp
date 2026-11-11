#pragma once
#include "crypto.hpp"
#include "rotation.hpp"

namespace ramp {
struct PrivateNoiseRng {
    using result_type = uint64_t;
    EVP_CIPHER_CTX* ctx = nullptr;
    std::array<unsigned char, 4096> buffer{}, zero{};
    size_t position = 4096;

    static constexpr result_type min() {
        return 0;
    }

    static constexpr result_type max() {
        return UINT64_MAX;
    }

    explicit PrivateNoiseRng(const er::Key& key) {
        ctx = EVP_CIPHER_CTX_new();
        std::array<unsigned char, 16> iv{};
        check(
            ctx && EVP_EncryptInit_ex(ctx, EVP_aes_256_ctr(), nullptr, key.data(), iv.data()) == 1,
            "private noise RNG");
    }

    ~PrivateNoiseRng() {
        EVP_CIPHER_CTX_free(ctx);
    }

    result_type operator()() {
        if (position == buffer.size()) {
            int used = 0;
            check(
                EVP_EncryptUpdate(ctx, buffer.data(), &used, zero.data(), zero.size()) == 1 &&
                    used == int(buffer.size()),
                "private RNG block");
            position = 0;
        }
        result_type value;
        std::memcpy(&value, buffer.data() + position, 8);
        position += 8;
        return value;
    }
};

std::vector<float> secret_private_vectors(
    const std::vector<float>& raw, U dim, double lambda, const er::Key& key) {
    PrivateNoiseRng rng(key);
    std::normal_distribution<double> normal;
    std::gamma_distribution<double> gamma(dim, 1 / lambda);
    std::vector<float> z(raw.size());
    std::vector<double> g(dim);
    for (size_t off = 0; off < raw.size(); off += dim) {
        auto point = rotated(raw.data() + off, dim);
        double squared = 0;
        for (auto& value : g) {
            value = normal(rng);
            squared += value * value;
        }
        double multiple = gamma(rng) / std::sqrt(squared);
        for (U j = 0; j < dim; ++j) {
            z[off + j] = float(point[j] + multiple * g[j]);
        }
    }
    return z;
}

struct PublicProjection {
    U d, columns;
    double scale;
    std::vector<float> mean, basis;

    PublicProjection(const std::string& path, U r) : columns(r) {
        std::ifstream f(path, std::ios::binary);
        read(f, &d, 1);
        read(f, &scale, 1);
        mean.resize(d);
        std::vector<float> full(size_t(d) * d);
        read(f, mean.data(), d);
        read(f, full.data(), full.size());
        check(bool(f) && r <= d, "public projection file");
        check(r > 0 && std::isfinite(scale) && scale > 0, "public projection parameters");
        basis.resize(size_t(d) * r);
        for (U j = 0; j < d; ++j) {
            std::copy_n(full.data() + size_t(j) * d, r, basis.data() + size_t(j) * r);
        }
    }

    std::vector<float> operator()(const std::vector<float>& raw, U r) const {
        check(raw.size() % d == 0 && r == columns, "projection dimensions");
        std::vector<float> out(raw.size() / d * r);
        for (size_t row = 0; row < raw.size() / d; ++row) {
            float* dest = out.data() + row * r;
            for (U j = 0; j < d; ++j) {
                float v = float(raw[row * d + j] / scale - mean[j]);
                const float* b = basis.data() + size_t(j) * r;
                for (U k = 0; k < r; ++k) {
                    dest[k] += v * b[k];
                }
            }
        }
        return out;
    }
};
} // namespace ramp
