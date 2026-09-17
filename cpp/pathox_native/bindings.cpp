#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

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
        throw std::runtime_error(
            "tile_size and stride must be positive"
        );
    }

    const int height = static_cast<int>(mask.shape(0));
    const int width = static_cast<int>(mask.shape(1));

    if (height < tile_size || width < tile_size) {
        return py::array_t<float>(
            py::array::ShapeContainer{0, 4}
        );
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

    py::array_t<float> result(
        py::array::ShapeContainer{valid_count, 4}
    );

    auto output = result.mutable_unchecked<2>();

    int row = 0;

    for (int i = 0; i < total_tiles; ++i) {
        if (!valid[i]) {
            continue;
        }

        const int ty = i / nx;
        const int tx = i % nx;

        output(row, 0) =
            static_cast<float>(tx * stride);

        output(row, 1) =
            static_cast<float>(ty * stride);

        output(row, 2) =
            static_cast<float>(chosen_class[i]);

        output(row, 3) =
            chosen_fraction[i];

        ++row;
    }

    return result;
}


py::array_t<float> score_tissue_tiles(
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> mask,
    int slide_width,
    int slide_height,
    int tile_size,
    int stride,
    float min_tissue_fraction
) {
    if (mask.ndim() != 2) {
        throw std::runtime_error("mask must be a 2D uint8 array");
    }

    if (slide_width <= 0 || slide_height <= 0) {
        throw std::runtime_error(
            "slide_width and slide_height must be positive"
        );
    }

    if (tile_size <= 0 || stride <= 0) {
        throw std::runtime_error(
            "tile_size and stride must be positive"
        );
    }

    if (min_tissue_fraction < 0.0f ||
        min_tissue_fraction > 1.0f) {
        throw std::runtime_error(
            "min_tissue_fraction must be in [0, 1]"
        );
    }

    const int thumb_height =
        static_cast<int>(mask.shape(0));

    const int thumb_width =
        static_cast<int>(mask.shape(1));

    if (slide_width < tile_size ||
        slide_height < tile_size) {
        return py::array_t<float>(
            py::array::ShapeContainer{0, 3}
        );
    }

    auto input = mask.unchecked<2>();

    const int nx =
        1 + (slide_width - tile_size) / stride;

    const int ny =
        1 + (slide_height - tile_size) / stride;

    const int total_tiles = nx * ny;

    std::vector<std::uint8_t> valid(
        total_tiles,
        0
    );

    std::vector<float> fractions(
        total_tiles,
        0.0f
    );

    #pragma omp parallel for schedule(dynamic)
    for (int tile_idx = 0;
         tile_idx < total_tiles;
         ++tile_idx) {

        const int ty = tile_idx / nx;
        const int tx = tile_idx % nx;

        const int x = tx * stride;
        const int y = ty * stride;

        // Exact equivalent of the Python reference:
        //
        // scale_x = thumb_w / slide_w
        // scale_y = thumb_h / slide_h
        //
        // tx0 = int(x * scale_x)
        // tx1 = max(tx0 + 1, int((x + tile_size) * scale_x))
        const int tx0 = static_cast<int>(
            (static_cast<double>(x) *
             static_cast<double>(thumb_width)) /
            static_cast<double>(slide_width)
        );

        const int ty0 = static_cast<int>(
            (static_cast<double>(y) *
             static_cast<double>(thumb_height)) /
            static_cast<double>(slide_height)
        );

        int tx1 = static_cast<int>(
            (static_cast<double>(x + tile_size) *
             static_cast<double>(thumb_width)) /
            static_cast<double>(slide_width)
        );

        int ty1 = static_cast<int>(
            (static_cast<double>(y + tile_size) *
             static_cast<double>(thumb_height)) /
            static_cast<double>(slide_height)
        );

        tx1 = std::max(tx0 + 1, tx1);
        ty1 = std::max(ty0 + 1, ty1);

        tx1 = std::min(tx1, thumb_width);
        ty1 = std::min(ty1, thumb_height);

        const int region_width = tx1 - tx0;
        const int region_height = ty1 - ty0;

        if (region_width <= 0 ||
            region_height <= 0) {
            continue;
        }

        int tissue_pixels = 0;

        for (int yy = ty0; yy < ty1; ++yy) {
            for (int xx = tx0; xx < tx1; ++xx) {
                if (input(yy, xx) > 0) {
                    ++tissue_pixels;
                }
            }
        }

        const int region_pixels =
            region_width * region_height;

        const float fraction =
            static_cast<float>(tissue_pixels) /
            static_cast<float>(region_pixels);

        if (fraction >= min_tissue_fraction) {
            valid[tile_idx] = 1;
            fractions[tile_idx] = fraction;
        }
    }

    int valid_count = 0;

    for (int i = 0; i < total_tiles; ++i) {
        if (valid[i]) {
            ++valid_count;
        }
    }

    py::array_t<float> result(
        py::array::ShapeContainer{
            valid_count,
            3
        }
    );

    auto output = result.mutable_unchecked<2>();

    int row = 0;

    for (int i = 0; i < total_tiles; ++i) {
        if (!valid[i]) {
            continue;
        }

        const int ty = i / nx;
        const int tx = i % nx;

        output(row, 0) =
            static_cast<float>(tx * stride);

        output(row, 1) =
            static_cast<float>(ty * stride);

        output(row, 2) =
            fractions[i];

        ++row;
    }

    return result;
}


PYBIND11_MODULE(pathox_native, m) {
    m.doc() =
        "Native OpenMP acceleration for PathoX tile processing";

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
        "score_tissue_tiles",
        &score_tissue_tiles,
        py::arg("mask"),
        py::arg("slide_width"),
        py::arg("slide_height"),
        py::arg("tile_size") = 512,
        py::arg("stride") = 512,
        py::arg("min_tissue_fraction") = 0.12f,
        "OpenMP tissue scoring using WSI-to-thumbnail coordinate mapping."
    );

    m.def(
        "max_threads",
        []() {
            return omp_get_max_threads();
        }
    );
}
