#pragma once
#include "common.hpp"

namespace ramp::er {
using Key = std::array<unsigned char, 32>;

struct KeyHash {
    size_t operator()(const Key& key) const {
        std::uint64_t a, b;
        std::memcpy(&a, key.data(), 8);
        std::memcpy(&b, key.data() + 16, 8);
        return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
    }
};

inline Bytes field(const std::string& value) {
    return {value.begin(), value.end()};
}

inline Bytes integer(std::uint64_t value, U width = 8) {
    Bytes data(width);
    for (U i = 0; i < width; ++i) {
        data[width - 1 - i] = value >> (8 * i);
    }
    return data;
}

inline Key prf(const Key& key, std::initializer_list<Bytes> parts) {
    Bytes message;
    for (const auto& part : parts) {
        auto size = integer(part.size(), 4);
        message.insert(message.end(), size.begin(), size.end());
        message.insert(message.end(), part.begin(), part.end());
    }
    Key result;
    unsigned used = 0;
    check(
        HMAC(EVP_sha256(), key.data(), 32, message.data(), message.size(), result.data(), &used) &&
            used == 32,
        "range PRF");
    return result;
}

inline Key random_key() {
    Key key;
    check(RAND_bytes(key.data(), key.size()) == 1, "private randomness");
    return key;
}
} // namespace ramp::er
