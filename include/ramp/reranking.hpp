#pragma once
#include "crypto.hpp"
#include "rotation.hpp"
#include <rabitqlib/quantization/rabitq.hpp>

namespace ramp {
struct Codec {
    const Data& d;
    er::Key key = er::random_key();
    std::unordered_map<U, Bytes> stored;
    std::vector<float> rq;

    explicit Codec(const Data& data) : d(data) {
        check(d.dim > 0, "reranking dimension");
        rq.resize(size_t(d.nq) * d.dim);
        for (U i = 0; i < d.nq; ++i) {
            auto a = rotated(d.y.data() + size_t(i) * d.dim, d.dim);
            std::copy(a.begin(), a.end(), rq.begin() + size_t(i) * d.dim);
        }
    }

    Bytes pack(U id) const {
        auto h = rotated(d.x.data() + size_t(id) * d.dim, d.dim);
        std::vector<uint8_t> codes(d.dim);
        float add = 0, rescale = 0, error = 0;
        double squared = 0;
        for (float v : h) {
            squared += double(v) * v;
        }
        if (squared > 0) {
            rabitqlib::quant::quantize_full_single(
                h.data(),
                d.dim,
                8,
                codes.data(),
                add,
                rescale,
                error,
                rabitqlib::METRIC_L2,
                rabitqlib::quant::RabitqConfig());
        }
        Bytes out;
        append_float(out, add);
        append_float(out, rescale);
        out.insert(out.end(), codes.begin(), codes.end());
        return out;
    }

    std::array<U, 7> aad(U id) const {
        return {1, 1, id, d.dim, 8, 0, 6};
    }

    const Bytes& ciphertext(U id) {
        auto found = stored.find(id);
        if (found != stored.end()) {
            return found->second;
        }
        Bytes plain = pack(id), ct(12 + plain.size() + 16);
        uint64_t serial = id;
        std::memcpy(ct.data() + 4, &serial, 8);
        // Unique nonce per record per independently sampled format key.
        auto ctx = EVP_CIPHER_CTX_new();
        int used, tail;
        auto ad = aad(id);
        check(
            EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, key.data(), ct.data()) == 1,
            "encrypt init");
        check(
            EVP_EncryptUpdate(
                ctx, nullptr, &used, reinterpret_cast<unsigned char*>(ad.data()), sizeof(ad)) == 1,
            "encrypt AAD");
        check(
            EVP_EncryptUpdate(ctx, ct.data() + 12, &used, plain.data(), plain.size()) == 1,
            "encrypt");
        check(EVP_EncryptFinal_ex(ctx, ct.data() + 12 + used, &tail) == 1, "encrypt final");
        check(
            EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, ct.data() + 12 + plain.size()) == 1,
            "tag");
        EVP_CIPHER_CTX_free(ctx);
        return stored.emplace(id, std::move(ct)).first->second;
    }

    double decode_score_view(
        U id, const unsigned char* ct, size_t bytes, U qi, EVP_CIPHER_CTX* ctx, Bytes& plain)
        const {
        check(bytes >= 28, "ciphertext size");
        size_t n = bytes - 28;
        plain.resize(n);
        int used, tail;
        auto ad = aad(id);
        check(
            EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, key.data(), ct) == 1,
            "decrypt init");
        check(
            EVP_DecryptUpdate(
                ctx, nullptr, &used, reinterpret_cast<unsigned char*>(ad.data()), sizeof(ad)) == 1,
            "decrypt AAD");
        check(EVP_DecryptUpdate(ctx, plain.data(), &used, ct + 12, n) == 1, "decrypt");
        check(
            EVP_CIPHER_CTX_ctrl(
                ctx, EVP_CTRL_GCM_SET_TAG, 16, const_cast<unsigned char*>(ct + 12 + n)) == 1,
            "set tag");
        check(EVP_DecryptFinal_ex(ctx, plain.data() + used, &tail) == 1, "authenticate");
        const float* q = rq.data() + size_t(qi) * d.dim;
        double dot = 0;
        check(n == size_t(8) + d.dim, "Extended RaBitQ reranking plaintext length");
        double factor = -.5 * get_float(plain.data() + 4);
        for (U j = 0; j < d.dim; ++j) {
            dot += ((double(plain[8 + j]) - 127.5) * factor) * q[j];
        }
        return double(get_float(plain.data())) - 2 * dot;
    }
};

Bytes gather(const std::vector<Bytes>& store, const std::vector<U>& ids) {
    Bytes out;
    size_t row = store.front().size();
    out.reserve(4 + ids.size() * (4 + row));
    append_u32(out, ids.size());
    for (U id : ids) {
        append_u32(out, id);
        out.insert(out.end(), store.at(id).begin(), store.at(id).end());
    }
    return out;
}

std::vector<U> project_refine(Codec& client, const Bytes& response, U qi, U k = 10) {
    size_t at = 0;
    U count = take_u32(response, at);
    size_t row = 36 + client.d.dim;
    check(response.size() == 4 + count * (4 + row), "candidate response framing");
    std::vector<std::pair<double, U>> rank;
    auto* ctx = EVP_CIPHER_CTX_new();
    Bytes plain;
    rank.reserve(count);
    for (U j = 0; j < count; ++j) {
        U id = take_u32(response, at);
        rank.push_back(
            {client.decode_score_view(id, response.data() + at, row, qi, ctx, plain), id});
        at += row;
    }
    EVP_CIPHER_CTX_free(ctx);
    size_t take = std::min(size_t(k), rank.size());
    std::partial_sort(rank.begin(), rank.begin() + take, rank.end());
    std::vector<U> ids;
    for (size_t j = 0; j < take; ++j) {
        ids.push_back(rank[j].second);
    }
    return ids;
}
} // namespace ramp
