#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace {

std::uint64_t edge_key(std::int32_t left, std::int32_t right) {
    if (left > right) {
        std::swap(left, right);
    }
    return (static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(left))
            << 32) |
        static_cast<std::uint32_t>(right);
}

std::uint64_t splitmix64(std::uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

}  // namespace

// Greedily edge-color one globally score-sorted candidate list into up to
// 64 edge-disjoint perfect matchings. Unlike the legacy Python routine, this
// scans the candidate list once rather than once per stage.
extern "C" int task_edge_color(
    const std::int32_t* sorted_edges,
    std::size_t edge_count,
    std::int32_t width,
    std::int32_t stages,
    std::uint64_t seed,
    std::int32_t* permutations,
    std::int32_t* candidate_pair_counts) {
    if (sorted_edges == nullptr || permutations == nullptr ||
        candidate_pair_counts == nullptr || width <= 0 ||
        (width % 2) != 0 || stages <= 0 || stages > 64) {
        return 1;
    }

    const std::uint64_t full_mask =
        stages == 64 ? ~std::uint64_t{0}
                     : ((std::uint64_t{1} << stages) - 1);
    const std::int32_t pairs_per_stage = width / 2;
    const std::int64_t target_pairs =
        static_cast<std::int64_t>(stages) * pairs_per_stage;
    std::vector<std::uint64_t> occupied(width, 0);
    std::vector<std::int32_t> counts(stages, 0);
    std::vector<std::uint8_t> is_candidate(
        static_cast<std::size_t>(target_pairs), 0);
    std::unordered_set<std::uint64_t> used_edges;
    used_edges.reserve(static_cast<std::size_t>(target_pairs * 2));
    std::fill(candidate_pair_counts, candidate_pair_counts + stages, 0);

    std::int64_t assigned_pairs = 0;
    for (std::size_t edge_index = 0;
         edge_index < edge_count && assigned_pairs < target_pairs;
         ++edge_index) {
        std::int32_t left = sorted_edges[2 * edge_index];
        std::int32_t right = sorted_edges[2 * edge_index + 1];
        if (left == right || left < 0 || right < 0 ||
            left >= width || right >= width) {
            continue;
        }
        if (left > right) {
            std::swap(left, right);
        }
        const auto key = edge_key(left, right);
        if (used_edges.find(key) != used_edges.end()) {
            continue;
        }
        std::uint64_t available =
            full_mask & ~(occupied[left] | occupied[right]);
        if (available == 0) {
            continue;
        }

        std::int32_t selected_stage = -1;
        std::int32_t selected_count = pairs_per_stage + 1;
        for (std::int32_t stage = 0; stage < stages; ++stage) {
            if ((available & (std::uint64_t{1} << stage)) != 0 &&
                counts[stage] < selected_count) {
                selected_stage = stage;
                selected_count = counts[stage];
            }
        }
        if (selected_stage < 0 ||
            counts[selected_stage] >= pairs_per_stage) {
            continue;
        }

        const std::int32_t pair_index = counts[selected_stage]++;
        const std::size_t output_offset =
            static_cast<std::size_t>(selected_stage) * width +
            2 * pair_index;
        permutations[output_offset] = left;
        permutations[output_offset + 1] = right;
        occupied[left] |= std::uint64_t{1} << selected_stage;
        occupied[right] |= std::uint64_t{1} << selected_stage;
        used_edges.insert(key);
        is_candidate[
            static_cast<std::size_t>(selected_stage) *
                pairs_per_stage +
            pair_index] = 1;
        ++candidate_pair_counts[selected_stage];
        ++assigned_pairs;
    }

    // Complete any unmatched vertices deterministically. The candidate graph
    // supplies almost all pairs in normal use; this path preserves the exact
    // perfect-matching and cross-stage unique-edge invariants.
    for (std::int32_t stage = 0; stage < stages; ++stage) {
        std::vector<std::int32_t> remaining;
        remaining.reserve(width);
        const std::uint64_t stage_bit = std::uint64_t{1} << stage;
        for (std::int32_t vertex = 0; vertex < width; ++vertex) {
            if ((occupied[vertex] & stage_bit) == 0) {
                remaining.push_back(vertex);
            }
        }
        std::uint64_t random_state =
            seed ^ (0x9e3779b97f4a7c15ULL *
                    static_cast<std::uint64_t>(stage + 1));
        for (std::size_t index = remaining.size();
             index > 1;
             --index) {
            const std::size_t swap_index =
                splitmix64(random_state) % index;
            std::swap(remaining[index - 1], remaining[swap_index]);
        }

        while (!remaining.empty()) {
            const std::int32_t left = remaining.back();
            remaining.pop_back();
            std::ptrdiff_t partner_index = -1;
            for (std::ptrdiff_t index =
                     static_cast<std::ptrdiff_t>(remaining.size()) - 1;
                 index >= 0;
                 --index) {
                if (used_edges.find(
                        edge_key(left, remaining[index])) ==
                    used_edges.end()) {
                    partner_index = index;
                    break;
                }
            }
            if (partner_index >= 0) {
                const std::int32_t right = remaining[partner_index];
                remaining.erase(remaining.begin() + partner_index);
                const std::int32_t pair_index = counts[stage]++;
                const std::size_t output_offset =
                    static_cast<std::size_t>(stage) * width +
                    2 * pair_index;
                permutations[output_offset] = left;
                permutations[output_offset + 1] = right;
                occupied[left] |= stage_bit;
                occupied[right] |= stage_bit;
                used_edges.insert(edge_key(left, right));
                ++assigned_pairs;
                continue;
            }

            // Repair a blocked leftover by replacing one prior pair with two
            // new unique edges.
            bool repaired = false;
            for (std::ptrdiff_t remaining_index =
                     static_cast<std::ptrdiff_t>(remaining.size()) - 1;
                 remaining_index >= 0 && !repaired;
                 --remaining_index) {
                const std::int32_t right =
                    remaining[remaining_index];
                for (std::int32_t pair_index = 0;
                     pair_index < counts[stage] && !repaired;
                     ++pair_index) {
                    const std::size_t pair_offset =
                        static_cast<std::size_t>(stage) * width +
                        2 * pair_index;
                    const std::int32_t prior_left =
                        permutations[pair_offset];
                    const std::int32_t prior_right =
                        permutations[pair_offset + 1];
                    const std::int32_t first_options[2] = {
                        prior_left, prior_right};
                    const std::int32_t second_options[2] = {
                        prior_right, prior_left};
                    for (int option = 0;
                         option < 2 && !repaired;
                         ++option) {
                        const auto first_key =
                            edge_key(left, first_options[option]);
                        const auto second_key =
                            edge_key(right, second_options[option]);
                        if (first_key == second_key ||
                            used_edges.find(first_key) !=
                                used_edges.end() ||
                            used_edges.find(second_key) !=
                                used_edges.end()) {
                            continue;
                        }
                        const auto prior_key =
                            edge_key(prior_left, prior_right);
                        used_edges.erase(prior_key);
                        used_edges.insert(first_key);
                        used_edges.insert(second_key);
                        permutations[pair_offset] = left;
                        permutations[pair_offset + 1] =
                            first_options[option];
                        const std::int32_t new_pair = counts[stage]++;
                        const std::size_t new_offset =
                            static_cast<std::size_t>(stage) * width +
                            2 * new_pair;
                        permutations[new_offset] = right;
                        permutations[new_offset + 1] =
                            second_options[option];
                        if (is_candidate[
                                static_cast<std::size_t>(stage) *
                                    pairs_per_stage +
                                pair_index] != 0) {
                            is_candidate[
                                static_cast<std::size_t>(stage) *
                                    pairs_per_stage +
                                pair_index] = 0;
                            --candidate_pair_counts[stage];
                        }
                        remaining.erase(
                            remaining.begin() + remaining_index);
                        occupied[left] |= stage_bit;
                        occupied[right] |= stage_bit;
                        ++assigned_pairs;
                        repaired = true;
                    }
                }
            }
            if (!repaired) {
                return 2;
            }
        }
        if (counts[stage] != pairs_per_stage) {
            return 3;
        }
    }
    return assigned_pairs == target_pairs ? 0 : 4;
}
