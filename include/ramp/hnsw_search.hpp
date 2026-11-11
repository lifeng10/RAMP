#pragma once
#include "search_codes.hpp"
#include "range_index.hpp"
#include <hnswlib/hnswlib.h>

namespace ramp {
struct LayerGraph {
    std::vector<Graph> layers;
    U entry = 0;
    size_t bytes = 0;
};

// A separate, per-worker epoch tracks bottom-layer discovery. Scoring a
// vertex on an upper layer must not prevent its later bottom-layer admission.
struct BottomDiscovery {
    std::vector<U> stamp;
    U epoch = 0;

    void begin(size_t n) {
        if (stamp.size() != n) {
            stamp.assign(n, 0);
            epoch = 0;
        }
        if (++epoch == 0) {
            std::fill(stamp.begin(), stamp.end(), 0);
            epoch = 1;
        }
    }

    bool insert(U id) {
        if (stamp[id] == epoch) {
            return false;
        }
        stamp[id] = epoch;
        return true;
    }
};

LayerGraph hnsw_graph(const std::vector<float>& z, U d, const std::string& path) {
    U n = z.size() / d;
    hnswlib::L2Space space(d);
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    const bool cached = std::filesystem::exists(path);
    if (cached) {
        index = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, path);
        check(
            index->max_elements_ >= n && index->cur_element_count == n,
            "HNSW cache record count differs from current vectors");
        check(
            index->label_offset_ >= index->offsetData_ &&
                index->label_offset_ - index->offsetData_ == size_t(d) * sizeof(float),
            "HNSW cache dimension differs from current vectors");
    } else {
        index = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, n, 16, 200, 912);
        auto start = Clock::now();
        for (U i = 0; i < n; ++i) {
            index->addPoint(z.data() + size_t(i) * d, i);
            if ((i + 1) % 25000 == 0) {
                std::cerr << "HNSW build " << i + 1 << " seconds=" << ms(start) / 1000 << std::endl;
            }
        }
        index->saveIndex(path);
    }
    LayerGraph out;
    out.entry = index->enterpoint_node_;
    out.layers.resize(index->maxlevel_ + 1, Graph(n));
    for (U i = 0; i < n; ++i) {
        check(index->getExternalLabel(i) == i, "sequential graph labels");
        if (cached) {
            check(
                std::memcmp(
                    index->getDataByInternalId(i),
                    z.data() + size_t(i) * d,
                    size_t(d) * sizeof(float)) == 0,
                "HNSW cache contains different vectors; rebuild it for the current projection and perturbation");
        }
        for (int l = 0; l <= index->element_levels_[i]; ++l) {
            auto* links = index->get_linklist_at_level(i, l);
            U count = index->getListCount(links);
            auto* ids = reinterpret_cast<U*>(links + 1);
            out.layers[l][i].assign(ids, ids + count);
            out.bytes += 4 * (1 + count);
        }
    }
    return out;
}

MetricRun graph_search(
    const MetricCodes& codes,
    const LayerGraph& graph,
    const opaque_range::Mask& mask,
    const MetricQuery& q,
    std::vector<U>& stamp,
    U epoch,
    std::vector<float>& cache,
    U b0 = 8,
    U tau = 32) {
    MetricRun run;
    double qn = normsq(q.q.data(), codes.dim);
    double offset =
        -double((1U << codes.bits) - 1) * .5 * std::accumulate(q.q.begin(), q.q.end(), 0.);
    auto score = [&](U id) {
        if (stamp[id] != epoch) {
            stamp[id] = epoch;
            cache[id] = codes.score(id, q.q.data(), qn, offset);
            if (mask.allowed(id)) {
                run.visited.push_back({cache[id], id});
            }
        }
        return cache[id];
    };
    if (mask.count() == 0) {
        return run;
    }
    U entry = graph.entry;
    float best = score(entry);
    // Upper-level navigation uses only the same public private-vector scores.
    for (int l = int(graph.layers.size()) - 1; l > 0; --l) {
        bool change = true;
        while (change) {
            change = false;
            for (U v : graph.layers[l][entry]) {
                ++run.edges;
                float dist = score(v);
                if (FPair(dist, v) < FPair(best, entry)) {
                    entry = v;
                    best = dist;
                    change = true;
                }
            }
        }
    }

    struct Entry {
        float score;
        U id;
        bool expanded;
    };

    auto before = [](auto& a, auto& b) {
        return FPair(a.score, a.id) < FPair(b.score, b.id);
    };
    std::vector<Entry> beam;
    beam.reserve(q.width + 1);
    static thread_local BottomDiscovery discovered;
    discovered.begin(codes.n);
    auto put = [&](U id) {
        if (!discovered.insert(id)) {
            return beam.size();
        }
        Entry item{score(id), id, false};
        auto it = std::lower_bound(beam.begin(), beam.end(), item, before);
        if (beam.size() == q.width && it == beam.end()) {
            return beam.size();
        }
        size_t pos = it - beam.begin();
        beam.insert(it, item);
        if (beam.size() > q.width) {
            beam.pop_back();
        }
        return pos;
    };
    if (mask.allowed(entry)) {
        put(entry);
    }
    {
        // Sample min(b0, m) distinct matching IDs without copying all postings.
        // Floyd sampling needs O(b0) draws, including when b0 is at least m.
        std::mt19937 rng(q.seed);
        const size_t count = mask.count(), take = std::min(size_t(b0), count);
        std::unordered_set<size_t> selected;
        selected.reserve(take);
        for (size_t last = count - take; last < count; ++last) {
            size_t pick = std::uniform_int_distribution<size_t>(0, last)(rng);
            if (!selected.insert(pick).second) {
                selected.insert(last);
                pick = last;
            }
            put(mask.full ? U(pick) : mask.ids[pick]);
        }
    }
    size_t pos = 0;
    auto& g = graph.layers[0];
    while (pos < beam.size()) {
        if (beam[pos].expanded) {
            ++pos;
            continue;
        }
        U id = beam[pos].id;
        beam[pos].expanded = true;
        size_t restart = pos + 1;
        U fresh = 0;
        auto visit = [&](U v) {
            ++run.edges;
            if (!mask.allowed(v)) {
                return;
            }
            const bool unscored = stamp[v] != epoch;
            restart = std::min(restart, put(v));
            if (unscored) {
                ++fresh;
            }
        };
        for (U v : g[id]) {
            visit(v);
        }
        // Direct matching neighbors are always processed first. Only newly
        // computed matching distances count toward the extra two-hop threshold.
        if (mask.count() < codes.n && fresh < tau) {
            for (U v : g[id]) {
                for (U w : g[v]) {
                    visit(w);
                    if (fresh >= tau) {
                        break;
                    }
                }
                if (fresh >= tau) {
                    break;
                }
            }
        }
        pos = restart;
    }
    return run;
}

// Ensure the result-count contract even when filtering disconnects the graph.
// No new report is requested. Only previously unscored eligible records are read.
void complete_candidates(
    MetricRun& run,
    const MetricCodes& codes,
    const opaque_range::Mask& mask,
    const MetricQuery& q,
    U k,
    std::vector<U>& stamp,
    U epoch,
    std::vector<float>& cache) {
    const size_t target = std::min(size_t(k), mask.count());
    if (run.visited.size() >= target) {
        return;
    }
    double qn = normsq(q.q.data(), codes.dim), offset = -double((1U << codes.bits) - 1) * .5 *
                                                        std::accumulate(q.q.begin(), q.q.end(), 0.);
    auto add = [&](U id) {
        if (stamp[id] != epoch) {
            stamp[id] = epoch;
            cache[id] = codes.score(id, q.q.data(), qn, offset);
            run.visited.push_back({cache[id], id});
        }
    };
    if (mask.full) {
        for (U id = 0; id < codes.n && run.visited.size() < target; ++id) {
            add(id);
        }
    } else {
        for (U id : mask.ids) {
            if (run.visited.size() >= target) {
                break;
            }
            add(id);
        }
    }
    check(run.visited.size() >= target, "eligible candidate count");
}
} // namespace ramp
