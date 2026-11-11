#pragma once
#include "search_codes.hpp"
#include "range_index.hpp"
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

namespace ramp {
struct Partition {
    U c, r;
    std::vector<float> centers;
    std::vector<U> label;
    std::vector<U> all_ids, all_offsets;
    double build_ms;

    Partition(const std::vector<float>& x, U n, U dim, U cells) : c(cells), r(dim), label(n) {
        auto start = Clock::now();

        faiss::ClusteringParameters cp;
        cp.niter = 15;
        cp.nredo = 1;
        cp.seed = 20261009;
        cp.max_points_per_centroid = 40;
        cp.min_points_per_centroid = 1;
        faiss::Clustering clustering(r, c, cp);
        faiss::IndexFlatL2 index(r);
        clustering.train(n, x.data(), index);
        centers = clustering.centroids;
        std::vector<faiss::idx_t> ids(n);
        std::vector<float> distances(n);
        index.search(n, x.data(), 1, distances.data(), ids.data());

        for (U i = 0; i < n; ++i) {
            check(ids[i] >= 0 && ids[i] < c, "partition label");
            label[i] = ids[i];
        }

        all_offsets.assign(c + 1, 0);

        for (U j : label) {
            ++all_offsets[j + 1];
        }

        std::partial_sum(all_offsets.begin(), all_offsets.end(), all_offsets.begin());
        auto cursor = all_offsets;
        all_ids.resize(n);

        for (U i = 0; i < n; ++i) {
            all_ids[cursor[label[i]]++] = i;
        }
        
        build_ms = ms(start);
    }
};

MetricRun decoded_groups_ivf(
    const Partition& p,
    const MetricCodes& codes,
    const opaque_range::Mask& mask,
    const std::vector<float>& report,
    U probes,
    size_t budget,
    U& actual,
    opaque_range::DecodedGroups& groups,
    double& group_ms,
    double& centers_ms,
    double& score_ms) {
    actual = 0;
    budget = std::min(budget, mask.count());
    auto start = Clock::now();
    std::vector<U> offsets, ids;
    const std::vector<U>* op = &p.all_offsets;
    const std::vector<U>* ip = &p.all_ids;
    if (!mask.full) {
        if (!groups.ready) {
            groups.labels.resize(mask.ids.size());
            groups.counts.assign(p.c, 0);
            for (size_t i = 0; i < mask.ids.size(); ++i) {
                U label = p.label[mask.ids[i]];
                groups.labels[i] = uint8_t(label);
                ++groups.counts[label];
            }
            groups.ready = true;
        }
        offsets.resize(p.c + 1);
        offsets[0] = 0;
        for (U j = 0; j < p.c; ++j) {
            offsets[j + 1] = offsets[j] + groups.counts[j];
        }
        auto cursor = offsets;
        ids.resize(mask.ids.size());
        for (size_t i = 0; i < mask.ids.size(); ++i) {
            ids[cursor[groups.labels[i]]++] = mask.ids[i];
        }
        op = &offsets;
        ip = &ids;
    }
    const auto& offset = *op;
    const auto& selected_ids = *ip;
    group_ms = ms(start);
    start = Clock::now();
    std::vector<FPair> order;
    order.reserve(p.c);
    for (U j = 0; j < p.c; ++j) {
        if (offset[j] != offset[j + 1]) {
            float dist = 0;
            const float* center = p.centers.data() + size_t(j) * p.r;
            for (U d = 0; d < p.r; ++d) {
                float delta = report[d] - center[d];
                dist += delta * delta;
            }
            order.emplace_back(dist, j);
        }
    }
    std::sort(order.begin(), order.end());
    double qn = normsq(report.data(), p.r);
    double off =
        -double((1U << codes.bits) - 1) * .5 * std::accumulate(report.begin(), report.end(), 0.);
    MetricRun run;
    size_t needed = 0;
    U selected = 0;
    for (auto [dist, j] : order) {
        if (selected >= probes && needed >= budget) {
            break;
        }
        ++selected;
        needed += offset[j + 1] - offset[j];
    }
    run.visited.reserve(needed);
    centers_ms = ms(start);
    start = Clock::now();
    for (auto [dist, j] : order) {
        if (actual >= probes && run.visited.size() >= budget) {
            break;
        }
        ++actual;
        for (U pos = offset[j]; pos < offset[j + 1]; ++pos) {
            U id = selected_ids[pos];
            if (pos + 16 < offset[j + 1]) {
                U next = selected_ids[pos + 16];
                __builtin_prefetch(codes.packed.data() + size_t(next) * codes.rowbytes, 0, 1);
                __builtin_prefetch(codes.add.data() + next, 0, 1);
                __builtin_prefetch(codes.rescale.data() + next, 0, 1);
            }
            run.visited.emplace_back(codes.score(id, report.data(), qn, off), id);
        }
    }
    score_ms = ms(start);
    return run;
}
} // namespace ramp
