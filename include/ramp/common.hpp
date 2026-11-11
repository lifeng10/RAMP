#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <omp.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>

namespace ramp {
using U = std::uint32_t;
using Bytes = std::vector<unsigned char>;
using Graph = std::vector<std::vector<U>>;
using FPair = std::pair<float, U>;
using Clock = std::chrono::steady_clock;

inline void check(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

inline double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

template <class T>
inline void read(std::ifstream& file, T* data, size_t count) {
    file.read(reinterpret_cast<char*>(data), sizeof(T) * count);
    check(bool(file), "short input");
}

inline void append_u32(Bytes& data, U value) {
    for (U i = 0; i < 4; ++i) {
        data.push_back(value >> (8 * i));
    }
}

inline U take_u32(const Bytes& data, size_t& offset) {
    check(offset + 4 <= data.size(), "truncated integer");
    U value = 0;
    for (U i = 0; i < 4; ++i) {
        value |= U(data[offset++]) << (8 * i);
    }
    return value;
}

inline void append_float(Bytes& data, float value) {
    const auto* ptr = reinterpret_cast<const unsigned char*>(&value);
    data.insert(data.end(), ptr, ptr + sizeof(float));
}

inline float get_float(const unsigned char* data) {
    float value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

inline double normsq(const float* data, U dimension) {
    double sum = 0;
    for (U i = 0; i < dimension; ++i) {
        sum += double(data[i]) * data[i];
    }
    return sum;
}

// Owner inputs and client queries. Neither is part of the server index.
struct Data {
    U n = 0, nq = 0, dim = 0;
    std::vector<float> x, y;
};

struct MetricQuery {
    std::vector<float> q;
    U seed = 0, width = 256, budget = 128;
};

struct MetricRun {
    std::vector<FPair> visited;
    std::uint64_t edges = 0;
};
} // namespace ramp
