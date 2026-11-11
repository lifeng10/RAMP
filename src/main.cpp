#include "ramp/hnsw_search.hpp"
#include "ramp/ivf_search.hpp"
#include "ramp/projection.hpp"
#include "ramp/reranking.hpp"

using namespace ramp;

namespace {
struct Request {
    U query_id, lower, upper, k, budget, width, probes;
};

static_assert(sizeof(Request) == 7 * sizeof(U));

struct Input {
    Data data;
    U r;
    double lambda_d, lambda_q;
    std::vector<U> attributes;
    std::vector<Request> requests;

    explicit Input(const std::filesystem::path& folder) {
        check(
            std::endian::native == std::endian::little,
            "input format requires little-endian storage");
        std::ifstream file(folder / "input.bin", std::ios::binary);
        char magic[8];
        read(file, magic, 8);
        check(std::memcmp(magic, "RAMPIN01", 8) == 0, "input format/version mismatch");
        U header[5];
        read(file, header, 5);
        data.n = header[0];
        data.nq = header[1];
        data.dim = header[2];
        r = header[4];
        read(file, &lambda_d, 1);
        read(file, &lambda_q, 1);
        check(
            data.n >= 256 && data.nq && data.dim && r && r <= data.dim && r % 32 == 0,
            "invalid vector dimensions/counts");
        check(
            std::isfinite(lambda_d) && lambda_d > 0 && std::isfinite(lambda_q) && lambda_q > 0,
            "invalid privacy parameters");
        const std::uint64_t expected = 44ULL + 4ULL * data.n +
                                       4ULL * (std::uint64_t(data.n) + data.nq) * data.dim +
                                       28ULL * header[3];
        check(
            std::filesystem::file_size(folder / "input.bin") == expected,
            "input file size mismatch");
        attributes.resize(data.n);
        data.x.resize(size_t(data.n) * data.dim);
        data.y.resize(size_t(data.nq) * data.dim);
        requests.resize(header[3]);
        read(file, attributes.data(), attributes.size());
        read(file, data.x.data(), data.x.size());
        read(file, data.y.data(), data.y.size());
        read(file, requests.data(), requests.size());
        for (const auto& request : requests) {
            check(
                request.query_id < data.nq && request.lower <= request.upper && request.k > 0 &&
                    request.k <= request.budget && request.width > 0 && request.probes > 0 &&
                    request.probes <= 256,
                "invalid request");
        }
        for (const auto* values : {&data.x, &data.y}) {
            for (float value : *values) {
                check(std::isfinite(value), "nonfinite input vector");
            }
        }
    }
};

struct Options {
    std::filesystem::path input, output;
    U scan_threshold = 0, hnsw_threshold = 0;
    U starts = 8, tau = 32, repeats = 1;
    bool calibrate = false, have_scan = false, have_hnsw = false;
};

void help() {
    std::cout << R"(RAMP: local execution of owner, server and client operations.

Prepare vectors and requests with prepare.py, then run:
  ramp --input PREPARED --output NEW_RUN_DIRECTORY \
       --scan-threshold S --hnsw-threshold H

For held-out calibration requests, measure RAMP's three search paths:
  ramp --input CALIBRATION_INPUT --output NEW_CALIBRATION_DIRECTORY --calibrate --repeats 3
  python select_thresholds.py NEW_CALIBRATION_DIRECTORY/queries.csv --records N
Use the returned thresholds for evaluation with the same search settings.
The program does not automatically tune width, probes, or candidate budget.
These are supplied per request, independently of evaluation results.

Options:
  --starts N      HNSW random starting nodes (default 8)
  --tau N         Two-hop stopping threshold (default 32)
  --repeats N     Repetitions per fixed perturbed request (default 1)

The output directory must not exist. Generated graphs and result CSVs belong
outside the source folder. Privacy/encryption keys are freshly generated in
memory and are not written to the source tree or printed. The HNSW cache in
the run directory contains perturbed vectors and is a local build artifact.
This executable is a local harness, not a network service.
)";
}

U integer_option(const std::string& text) {
    check(!text.empty() && text.front() != '-', "expected unsigned integer");
    size_t used = 0;
    auto value = std::stoull(text, &used);
    check(used == text.size() && value <= UINT32_MAX, "unsigned integer out of range");
    return U(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string name = argv[i];
        if (name == "--calibrate") {
            options.calibrate = true;
            continue;
        }
        check(i + 1 < argc, "missing option value");
        const std::string value = argv[++i];
        if (name == "--input") {
            options.input = value;
        } else if (name == "--output") {
            options.output = value;
        } else if (name == "--scan-threshold") {
            options.scan_threshold = integer_option(value);
            options.have_scan = true;
        } else if (name == "--hnsw-threshold") {
            options.hnsw_threshold = integer_option(value);
            options.have_hnsw = true;
        } else if (name == "--starts") {
            options.starts = integer_option(value);
        } else if (name == "--tau") {
            options.tau = integer_option(value);
        } else if (name == "--repeats") {
            options.repeats = integer_option(value);
        } else {
            throw std::runtime_error("unknown option: " + name);
        }
    }
    check(!options.input.empty() && !options.output.empty(), "--input and --output are required");
    check(
        options.starts && options.tau && options.repeats,
        "starts, tau and repeats must be positive");
    if (!options.calibrate) {
        check(
            options.have_scan && options.have_hnsw &&
                options.scan_threshold < options.hnsw_threshold,
            "supply independently calibrated thresholds with S < H");
    }
    check(!std::filesystem::exists(options.output), "output directory already exists");
    return options;
}

int choose_path(size_t matching, U budget, const Options& options) {
    if (matching <= budget || matching <= options.scan_threshold) {
        return 0;
    }
    return matching <= options.hnsw_threshold ? 1 : 2;
}

MetricRun scan(
    const MetricCodes& codes, const opaque_range::Mask& mask, const std::vector<float>& query) {
    MetricRun run;
    run.visited.reserve(mask.count());
    const double qnorm = normsq(query.data(), codes.dim);
    const double offset =
        -double((1U << codes.bits) - 1) * .5 * std::accumulate(query.begin(), query.end(), 0.);
    auto score = [&](U id) {
        run.visited.emplace_back(codes.score(id, query.data(), qnorm, offset), id);
    };
    if (mask.full) {
        for (U id = 0; id < codes.n; ++id) {
            score(id);
        }
    } else {
        for (U id : mask.ids) {
            score(id);
        }
    }
    return run;
}

std::vector<U> select_candidates(const MetricRun& run, U budget) {
    auto ranked = run.visited;
    const size_t count = std::min(size_t(budget), ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + count, ranked.end());
    std::vector<U> ids;
    ids.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        ids.push_back(ranked[i].second);
    }
    return ids;
}

void execute(const Options& options) {
    Input input(options.input);
    PublicProjection projection((options.input / "projection.bin").string(), input.r);
    check(projection.d == input.data.dim, "projection and database dimensions differ");
    check(
        options.calibrate || options.hnsw_threshold <= input.data.n,
        "threshold exceeds database size");
    std::filesystem::create_directories(options.output);
    const auto build_start = Clock::now();

    // Owner: one perturbation release supplies both vector indexes.
    auto z = secret_private_vectors(
        projection(input.data.x, input.r), input.r, input.lambda_d, er::random_key());
    auto graph = hnsw_graph(z, input.r, (options.output / "hnsw.bin").string());
    MetricCodes codes(z, input.r, 8);
    codes.discard_error_bounds();
    Partition partition(z, input.data.n, input.r, 256);
    std::vector<float>().swap(z);
    opaque_range::Client range_client;
    opaque_range::Directory directory(input.attributes, range_client);
    Codec owner_codec(input.data);
    std::vector<Bytes> ciphertexts(input.data.n);
    for (U id = 0; id < input.data.n; ++id) {
        ciphertexts[id] = owner_codec.ciphertext(id);
    }
    Data user;
    user.n = input.data.n;
    user.nq = input.data.nq;
    user.dim = input.data.dim;
    user.y = input.data.y;
    Codec client_codec(user);
    client_codec.key = owner_codec.key;
    std::vector<float>().swap(input.data.x);
    owner_codec.stored.clear();
    owner_codec.stored.rehash(0);
    std::cerr << "Index built in " << ms(build_start) << " ms; r=" << input.r << '\n';

    omp_set_num_threads(1);
    std::vector<U> stamps(input.data.n);
    std::vector<float> score_cache(input.data.n);
    U epoch = 0;
    const std::array<const char*, 3> names = {"Scan", "IVF", "HNSW"};
    std::ofstream measurements(options.output / "queries.csv");
    std::ofstream results(options.output / "results.csv");
    check(bool(measurements) && bool(results), "cannot create result files");
    measurements
        << std::setprecision(10)
        << "request,query_id,matching,path,budget,width,probes,repeat,visited,edges,returned,"
           "request_ms,range_ms,vector_ms,response_ms,rerank_ms,response_bytes\n";
    results << "request,path,repeat,rank,id\n";
    for (size_t index = 0; index < input.requests.size(); ++index) {
        const auto& request = input.requests[index];
        auto start = Clock::now();
        std::vector<float> raw(
            user.y.begin() + size_t(request.query_id) * user.dim,
            user.y.begin() + size_t(request.query_id + 1) * user.dim);
        // Independent fresh query noise; calibration paths reuse this same report.
        auto report = secret_private_vectors(
            projection(raw, input.r), input.r, input.lambda_q, er::random_key());
        const auto token = opaque_range::pack(range_client.token(request.lower, request.upper));
        const double request_ms = ms(start);
        std::vector<int> paths =
            options.calibrate ? std::vector<int>{0, 1, 2} : std::vector<int>{-1};
        if (options.calibrate) {
            std::mt19937 order(U(index) ^ 20261029U);
            std::shuffle(paths.begin(), paths.end(), order);
        }
        for (int requested_path : paths) {
            for (U repeat = 0; repeat < options.repeats; ++repeat) {
                start = Clock::now();
                opaque_range::DecodedGroups groups;
                groups.partitions = &partition.label;
                groups.count = partition.c;
                auto mask =
                    opaque_range::evaluate_bitmap(directory, opaque_range::parse(token), &groups);
                const double range_ms = ms(start);
                start = Clock::now();
                int path = requested_path < 0 ? choose_path(mask.count(), request.budget, options)
                                              : requested_path;
                MetricQuery query{
                    report, U(2026100102U + request.query_id), request.width, request.budget};
                MetricRun run;
                if (path == 0) {
                    run = scan(codes, mask, report);
                } else if (path == 1) {
                    U actual = 0;
                    double grouping_ms = 0, centers_ms = 0, scoring_ms = 0;
                    run = decoded_groups_ivf(partition, codes, mask, report, request.probes, request.budget, actual, groups, grouping_ms, centers_ms, scoring_ms);
                } else {
                    if (++epoch == 0) {
                        std::fill(stamps.begin(), stamps.end(), 0);
                        epoch = 1;
                    }
                    run = graph_search(codes, graph, mask, query, stamps, epoch, score_cache, options.starts, options.tau);

                    complete_candidates(
                        run, codes, mask, query, request.budget, stamps, epoch, score_cache);
                }
                auto candidates = select_candidates(run, request.budget);
                check(
                    candidates.size() == std::min(size_t(request.budget), mask.count()),
                    "candidate count");
                const double vector_ms = ms(start);
                start = Clock::now();
                auto response = gather(ciphertexts, candidates);
                const double response_ms = ms(start);
                start = Clock::now();
                auto final_ids =
                    project_refine(client_codec, response, request.query_id, request.k);
                const double rerank_ms = ms(start);
                check(
                    final_ids.size() == std::min(size_t(request.k), mask.count()), "result count");
                for (U id : final_ids) {
                    check(mask.allowed(id), "result violates range predicate");
                }
                measurements << index << ',' << request.query_id << ',' << mask.count() << ','
                             << names[path] << ',' << request.budget << ',' << request.width << ','
                             << request.probes << ',' << repeat << ',' << run.visited.size() << ','
                             << run.edges << ',' << candidates.size() << ',' << request_ms << ','
                             << range_ms << ',' << vector_ms << ',' << response_ms << ','
                             << rerank_ms << ',' << response.size() << '\n';
                for (size_t rank = 0; rank < final_ids.size(); ++rank) {
                    results << index << ',' << names[path] << ',' << repeat << ',' << rank + 1
                            << ',' << final_ids[rank] << '\n';
                }
            }
        }
    }
    check(bool(measurements) && bool(results), "result write failed");
    std::cout << "Completed " << input.requests.size()
              << " requests. Results: " << options.output.string() << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
            help();
            return 0;
        }
        execute(parse_options(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RAMP: " << error.what() << '\n';
        return 1;
    }
}
