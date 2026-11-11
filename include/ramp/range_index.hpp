#pragma once
#include "crypto.hpp"

namespace ramp {
namespace opaque_range {
struct Prefix {
    U depth;
    uint64_t value;

    bool operator<(const Prefix& b) const {
        return std::tie(depth, value) < std::tie(b.depth, b.value);
    }
};

std::vector<Prefix> cover(U lo, U hi) {
    check(lo <= hi, "inclusive range");
    std::vector<Prefix> out;
    uint64_t first = lo, last = uint64_t(hi) + 1;
    while (first < last) {
        uint64_t size = first ? first & (~first + 1) : 1ULL << 32;
        while (size > last - first) {
            size >>= 1;
        }
        U power = __builtin_ctzll(size);
        out.push_back({32 - power, first >> power});
        first += size;
    }
    return out;
}

struct Token {
    er::Key tag, key;
};

using Tokens = std::vector<Token>;

struct Client {
    er::Key address = er::random_key(), encryption = er::random_key();

    Token pair(Prefix p) const {
        return {
            er::prf(
                address,
                {er::field("dyadic-range-v1"), er::integer(p.depth), er::integer(p.value)}),
            er::prf(
                encryption,
                {er::field("dyadic-range-v1"), er::integer(p.depth), er::integer(p.value)})};
    }

    Tokens token(U lo, U hi) const {
        Tokens out;
        auto prefixes = cover(lo, hi);
        out.reserve(prefixes.size());
        // Exactly the original length-prefixed HMAC message. Reuse keyed
        // contexts within this call, without caching queries or tokens.
        Bytes msg;
        auto domain = er::field("dyadic-range-v1");
        auto length = er::integer(domain.size(), 4);
        msg.insert(msg.end(), length.begin(), length.end());
        msg.insert(msg.end(), domain.begin(), domain.end());
        auto eight = er::integer(8, 4);
        msg.insert(msg.end(), eight.begin(), eight.end());
        size_t depth_at = msg.size();
        msg.resize(msg.size() + 8);
        msg.insert(msg.end(), eight.begin(), eight.end());
        size_t value_at = msg.size();
        msg.resize(msg.size() + 8);
        using Context = std::unique_ptr<HMAC_CTX, decltype(&HMAC_CTX_free)>;
        Context a(HMAC_CTX_new(), HMAC_CTX_free), e(HMAC_CTX_new(), HMAC_CTX_free);
        check(a && e, "range MAC contexts");
        check(
            HMAC_Init_ex(a.get(), address.data(), 32, EVP_sha256(), nullptr) == 1 &&
                HMAC_Init_ex(e.get(), encryption.data(), 32, EVP_sha256(), nullptr) == 1,
            "range MAC keys");
        for (auto p : prefixes) {
            for (U j = 0; j < 8; ++j) {
                msg[depth_at + 7 - j] = uint64_t(p.depth) >> (8 * j);
                msg[value_at + 7 - j] = p.value >> (8 * j);
            }
            Token item;
            unsigned used = 0;
            auto digest = [&](HMAC_CTX* ctx, er::Key& dest) {
                check(
                    HMAC_Init_ex(ctx, nullptr, 0, nullptr, nullptr) == 1 &&
                        HMAC_Update(ctx, msg.data(), msg.size()) == 1 &&
                        HMAC_Final(ctx, dest.data(), &used) == 1 && used == 32,
                    "range MAC digest");
            };
            digest(a.get(), item.tag);
            digest(e.get(), item.key);
            out.push_back(item);
        }
        // Sorting by pseudorandom address conceals numerical cover order.
        std::sort(out.begin(), out.end(), [](auto& a, auto& b) {
            return a.tag < b.tag;
        });
        return out;
    }
};

Bytes pack(const Tokens& t) {
    Bytes b;
    b.reserve(4 + 64 * t.size());
    append_u32(b, t.size());
    for (auto& x : t) {
        b.insert(b.end(), x.tag.begin(), x.tag.end());
        b.insert(b.end(), x.key.begin(), x.key.end());
    }
    return b;
}

Tokens parse(const Bytes& b) {
    size_t at = 0;
    U n = take_u32(b, at);
    check(n <= 64 && b.size() == 4 + 64 * n, "opaque token frame");
    Tokens t(n);
    for (auto& x : t) {
        std::memcpy(x.tag.data(), b.data() + at, 32);
        std::memcpy(x.key.data(), b.data() + at + 32, 32);
        at += 64;
    }
    return t;
}

struct MaskStorage {
    std::vector<U> stamp;
    U epoch = 0;
    std::vector<uint64_t> bitmap;

    explicit MaskStorage(U n) : stamp(n) {
    }
};

struct Mask {
    U n;
    bool full = false;
    std::shared_ptr<MaskStorage> storage;
    U epoch = 0;
    std::vector<U> ids;
    bool packed = false;

    explicit Mask(U count) : n(count) {
    }

    size_t count() const {
        return full ? n : ids.size();
    }

    U sample(U draw) const {
        return full ? draw % n : ids[draw % ids.size()];
    }

    bool allowed(U id) const {
        check(id < n, "mask ID");
        if (packed) {
            return storage && ((storage->bitmap[id / 64] >> (id % 64)) & 1);
        }
        return full || (storage && storage->stamp[id] == epoch);
    }
};

struct Directory {
    U n;
    size_t stored_bytes = 0, posting_count = 0;
    std::unordered_map<er::Key, Bytes, er::KeyHash> rows;
    // Worker-local pool. Live masks retain their own workspace. Setup pays the
    // O(n) allocation; a query marks only the IDs in the opened postings.
    mutable std::vector<std::shared_ptr<MaskStorage>> workspaces;

    Directory(const std::vector<U>& attrs, const Client& client) : n(attrs.size()) {
        for (U i = 0; i < 2; ++i) {
            workspaces.push_back(std::make_shared<MaskStorage>(n));
        }
        std::map<Prefix, std::vector<U>> plain;
        for (U id = 0; id < n; ++id) {
            for (U d = 0; d <= 32; ++d) {
                plain[{d, uint64_t(attrs[id]) >> (32 - d)}].push_back(id);
            }
        }
        auto* ctx = EVP_CIPHER_CTX_new();
        check(ctx, "range encrypt context");
        for (auto& [p, ids] : plain) {
            auto token = client.pair(p);
            Bytes raw;
            raw.reserve(4 * ids.size());
            for (U id : ids) {
                append_u32(raw, id);
            }
            Bytes ct(12 + raw.size() + 16);
            check(RAND_bytes(ct.data(), 12) == 1, "range nonce");
            int len, tail;
            check(
                EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, token.key.data(), ct.data()) ==
                    1,
                "range init");
            check(EVP_EncryptUpdate(ctx, nullptr, &len, token.tag.data(), 32) == 1, "range aad");
            check(
                EVP_EncryptUpdate(ctx, ct.data() + 12, &len, raw.data(), raw.size()) == 1,
                "range seal");
            check(EVP_EncryptFinal_ex(ctx, ct.data() + 12 + len, &tail) == 1, "range final");
            check(
                EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, ct.data() + ct.size() - 16) == 1,
                "range tag");
            posting_count += ids.size();
            stored_bytes += 32 + ct.size();
            check(rows.emplace(token.tag, std::move(ct)).second, "unique dyadic row");
        }
        EVP_CIPHER_CTX_free(ctx);
        // Canonical upload order is by pseudorandom tag, not plaintext prefix.
        // Rebuild the server dictionary in that order as well, so hash-bucket
        // insertion order is independent of the owner's plaintext iteration.
        std::vector<std::pair<er::Key, Bytes>> upload;
        upload.reserve(rows.size());
        for (auto& entry : rows) {
            upload.emplace_back(entry.first, std::move(entry.second));
        }
        std::sort(upload.begin(), upload.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        rows.clear();
        for (auto& entry : upload) {
            rows.emplace(entry.first, std::move(entry.second));
        }
    }
};
} // namespace opaque_range

namespace opaque_range {
struct DecodedGroups {
    const std::vector<U>* partitions = nullptr;
    U count = 0;
    std::vector<uint8_t> labels;
    std::vector<U> counts;
    bool ready = false;
};
} // namespace opaque_range

namespace opaque_range {
Mask evaluate_bitmap(const Directory& dir, const Tokens& tokens, DecodedGroups* groups = nullptr) {
    Mask result(dir.n);
    size_t cardinality = 0;
    for (const auto& token : tokens) {
        auto it = dir.rows.find(token.tag);
        if (it != dir.rows.end()) {
            check(it->second.size() >= 28 && (it->second.size() - 28) % 4 == 0, "range row frame");
            cardinality += (it->second.size() - 28) / 4;
        }
    }
    if (cardinality == dir.n) {
        result.full = true;
        return result;
    }
    for (auto& state : dir.workspaces) {
        if (state.use_count() == 1) {
            result.storage = state;
            break;
        }
    }
    if (!result.storage) {
        dir.workspaces.push_back(std::make_shared<MaskStorage>(dir.n));
        result.storage = dir.workspaces.back();
    }
    if (++result.storage->epoch == 0) {
        std::fill(result.storage->stamp.begin(), result.storage->stamp.end(), 0);
        ++result.storage->epoch;
    }
    result.epoch = result.storage->epoch;
    result.ids.resize(cardinality);
    const bool packed = cardinality >= (size_t(dir.n) + 63) / 64;
    result.packed = packed;
    if (packed) {
        result.storage->bitmap.assign((size_t(dir.n) + 63) / 64, 0);
    }
    auto* bits = result.storage->bitmap.data();
    if (groups) {
        check(
            groups->partitions && groups->partitions->size() == dir.n && groups->count <= 256,
            "public IVF labels");
        groups->labels.resize(cardinality);
        groups->counts.assign(groups->count, 0);
    }
    U* marks = result.storage->stamp.data();
    using Context = std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)>;
    Context ctx(EVP_CIPHER_CTX_new(), EVP_CIPHER_CTX_free);
    check(bool(ctx), "range decrypt context");
    size_t written = 0;
    for (const auto& token : tokens) {
        auto it = dir.rows.find(token.tag);
        if (it == dir.rows.end()) {
            continue;
        }
        const auto& ct = it->second;
        size_t size = ct.size() - 28, count = size / 4;
        if (!count) {
            continue;
        }
        auto* destination = reinterpret_cast<unsigned char*>(result.ids.data() + written);
        int used = 0, tail = 0;
        check(
            EVP_DecryptInit_ex(
                ctx.get(), EVP_aes_256_gcm(), nullptr, token.key.data(), ct.data()) == 1,
            "range open init");
        check(
            EVP_DecryptUpdate(ctx.get(), nullptr, &used, token.tag.data(), 32) == 1,
            "range open aad");
        check(
            EVP_DecryptUpdate(ctx.get(), destination, &used, ct.data() + 12, size) == 1,
            "range open");
        check(
            EVP_CIPHER_CTX_ctrl(
                ctx.get(),
                EVP_CTRL_GCM_SET_TAG,
                16,
                const_cast<unsigned char*>(ct.data() + 12 + size)) == 1,
            "range open tag");
        check(EVP_DecryptFinal_ex(ctx.get(), destination + used, &tail) == 1, "range authenticate");
        check(size_t(used + tail) == size, "range plaintext length");
        for (size_t at = written; at < written + count; ++at) {
            U id = result.ids[at];
            if constexpr (std::endian::native != std::endian::little) {
                id = __builtin_bswap32(id);
                result.ids[at] = id;
            }
            check(id < dir.n, "mask ID");
            if (packed) {
                auto& word = bits[id / 64];
                uint64_t bit = 1ULL << (id % 64);
                check(!(word & bit), "disjoint cover postings");
                word |= bit;
            } else {
                check(marks[id] != result.epoch, "disjoint cover postings");
                marks[id] = result.epoch;
            }
            if (groups) {
                U label = (*groups->partitions)[id];
                check(label < groups->count, "IVF label");
                groups->labels[at] = uint8_t(label);
                ++groups->counts[label];
            }
        }
        written += count;
    }
    check(written == cardinality, "range result length");
    if (groups) {
        groups->ready = true;
    }
    return result;
}
} // namespace opaque_range
} // namespace ramp
