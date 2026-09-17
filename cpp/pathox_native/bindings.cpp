#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct Candidate {
    int x;
    int y;
    int target_class;
    float fraction;
};

py::array_t<float> score_tiles(
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> mask,
    int tile_size,
    int stride,
    float foreground_threshold,
    float class_threshold
) {
    if (mask.ndim() != 2) {
        throw std::runtime_error("mask must be a 2D uint8 array");
    }

    if (tile_size <= 0 || stride <= 0) {
        throw std::runtime_error("tile_size and stride must be positive");
    }

    const int height = static_cast<int>(mask.shape(0));
    const int width = static_cast<int>(mask.shape(1));

    if (height < tile_size || width < tile_size) {
        return py::array_t<float>(py::array::ShapeContainer{0, 4});
    }

    auto input = mask.unchecked<2>();

    const int nx = 1 + (width - tile_size) / stride;
    const int ny = 1 + (height - tile_size) / stride;
    const int total_tiles = nx * ny;

    std::vector<int> counts(total_tiles * 6, 0);
    std::vector<std::uint8_t> valid(total_tiles, 0);
    std::vector<std::uint8_t> chosen_class(total_tiles, 0);
    std::vector<float> chosen_fraction(total_tiles, 0.0f);

    #pragma omp parallel for schedule(dynamic)
    for (int tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
        const int ty = tile_idx / nx;
        const int tx = tile_idx % nx;

        const int x0 = tx * stride;
        const int y0 = ty * stride;

        int local_counts[6] = {0, 0, 0, 0, 0, 0};

        for (int y = y0; y < y0 + tile_size; ++y) {
            for (int x = x0; x < x0 + tile_size; ++x) {
                const std::uint8_t value = input(y, x);
                if (value <= 5) {
                    ++local_counts[value];
                }
            }
        }

        const int total_pixels = tile_size * tile_size;

        const float background_fraction =
            static_cast<float>(local_counts[0]) /
            static_cast<float>(total_pixels);

        if (background_fraction >= foreground_threshold) {
            valid[tile_idx] = 1;
            chosen_class[tile_idx] = 0;
            chosen_fraction[tile_idx] = background_fraction;
            continue;
        }

        int best_class = -1;
        int best_count = 0;

        for (int cls = 1; cls < 6; ++cls) {
            if (local_counts[cls] > best_count) {
                best_count = local_counts[cls];
                best_class = cls;
            }
        }

        if (best_class >= 1) {
            const float fraction =
                static_cast<float>(best_count) /
                static_cast<float>(total_pixels);

            if (fraction >= class_threshold) {
                valid[tile_idx] = 1;
                chosen_class[tile_idx] =
                    static_cast<std::uint8_t>(best_class);
                chosen_fraction[tile_idx] = fraction;
            }
        }
    }

    int valid_count = 0;

    for (int i = 0; i < total_tiles; ++i) {
        if (valid[i]) {
            ++valid_count;
        }
    }

    py::array_t<float> result({valid_count, 4});
    auto output = result.mutable_unchecked<2>();

    int row = 0;

    for (int i = 0; i < total_tiles; ++i) {
        if (!valid[i]) {
            continue;
        }

        const int ty = i / nx;
        const int tx = i % nx;

        output(row, 0) = static_cast<float>(tx * stride);
        output(row, 1) = static_cast<float>(ty * stride);
        output(row, 2) = static_cast<float>(chosen_class[i]);
        output(row, 3) = chosen_fraction[i];

        ++row;
    }

    return result;
}

PYBIND11_MODULE(pathox_native, m) {
    m.doc() = "Native OpenMP acceleration for PathoX tile processing";

    m.def(
        "score_tiles",
        &score_tiles,
        py::arg("mask"),
        py::arg("tile_size") = 512,
        py::arg("stride") = 512,
        py::arg("foreground_threshold") = 0.70f,
        py::arg("class_threshold") = 0.01f,
        "Parallel tile scoring over a uint8 segmentation mask."
    );

    m.def(
        "max_threads",
        []() {
            return omp_get_max_threads();
        }
    );
}
