#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

#include "geometry.h"
#include "layer.h"
#include "paper.h"
#include "constants.h"

namespace Rasterizer {
    struct Edge {
        double y_max;
        double x;
        double dx;
        size_t layer_id;
    };

    inline Rect compute_bounds(const std::vector<Layer>& layers, double sin_t, double cos_t) {
        double min_x = std::numeric_limits<double>::max();
        double max_x = std::numeric_limits<double>::lowest();
        double min_y = std::numeric_limits<double>::max();
        double max_y = std::numeric_limits<double>::lowest();

        for (const Layer & l : layers) {
            for (const Vec2 & v : l.vertices) {
                const double x = v.x*cos_t - v.y*sin_t;
                const double y = v.x*sin_t + v.y*cos_t;
                min_x = std::min(min_x, x); max_x = std::max(max_x, x);
                min_y = std::min(min_y, y); max_y = std::max(max_y, y);
            }
        }
        return {min_x, max_x, min_y, max_y};
    }

    inline std::vector<std::vector<Edge>> build_edge_table(
        const std::vector<Layer> & layers, int rows, int cols,
        double sin_t, double cos_t, const Rect & bounds) {
        std::vector<std::vector<Edge>> table(rows);

        const double w = bounds.max_x - bounds.min_x;
        const double h = bounds.max_y - bounds.min_y;
        if (w < Origami::EPSILON || h < Origami::EPSILON) return table;

        double scale = std::min((cols - 1) / w, (rows - 1) / h);

        auto to_raster = [rows, cos_t, sin_t, &bounds, &scale](Vec2 v) {
            const double x = v.x*cos_t - v.y*sin_t;
            const double y = v.x*sin_t + v.y*cos_t;
            return Vec2{
                (x - bounds.min_x) * scale,
                (rows - 1) - (y - bounds.min_y) * scale
            };
        };

        for (size_t i = 0; i < layers.size(); ++i) {
            const std::vector<Vec2> & vertices = layers[i].vertices;
            size_t n = vertices.size();
            for (size_t j = 0; j < n; ++j) {
                Vec2 p1 = to_raster(vertices[j]);
                Vec2 p2 = to_raster(vertices[(j + 1) % n]);
                if (p1.y > p2.y) std::swap(p1, p2);

                const double dy = p2.y - p1.y;
                if (dy < Origami::EPSILON) continue;

                const int y_start = std::max(0, static_cast<int>(std::ceil(p1.y)));
                const int y_end   = std::min(rows-1, static_cast<int>(std::floor(p2.y)));

                if (y_start <= y_end) {
                    const double dx = (p2.x - p1.x) / dy;
                    const double x_start = p1.x + (y_start - p1.y) * dx;
                    table[y_start].push_back({p2.y, x_start, dx, i});
                }
            }
        }
        return table;
    }

    inline std::vector<uint8_t> render(
        const std::vector<Layer> & layers, int rows, int cols,
        double theta) {
        if (rows < 1 || cols < 1 || layers.empty()) return {};

        const double sin_t = std::sin(theta);
        const double cos_t = std::cos(theta);

        Rect bounds = compute_bounds(layers, sin_t, cos_t);
        std::vector<std::vector<Edge>> edge_table = build_edge_table(
                layers, rows, cols, sin_t, cos_t, bounds);

        std::vector<uint8_t> grid(rows * cols, 0);
        std::vector<Edge> active;
        active.reserve(layers.size() * 2);

        // Sentinel value for layer start X
        std::vector<double> layer_starts(layers.size(), -1.0);
        std::vector<std::pair<int, int>> intervals;

        for (int y = 0; y < rows; ++y) {
            auto row_it = grid.begin() + y * cols;
            // Add new edges starting at this row
            if (!edge_table[y].empty()) {
                active.insert(active.end(), edge_table[y].begin(), edge_table[y].end());
            }

            active.erase(std::remove_if(active.begin(), active.end(), [y](const Edge& e) {
                return y >= e.y_max;
            }), active.end());

            if (active.empty()) continue;

            // Extract Intervals (Pairing Left/Right edges per layer)
            intervals.clear();
            for (const Edge & e : active) {
                if (layer_starts[e.layer_id] < 0) {
                    layer_starts[e.layer_id] = e.x;
                } else {
                    double x1 = layer_starts[e.layer_id];
                    double x2 = e.x;

                    // Round towards exclusion
                    int L = std::max(0, static_cast<int>(std::floor(std::min(x1, x2))));
                    int R = std::min(cols - 1, static_cast<int>(std::ceil(std::max(x1, x2))));

                    if (L <= R) intervals.push_back({L, R});

                    layer_starts[e.layer_id] = -1.0; // Reset
                }
            }

            // Union Intervals & Draw
            if (!intervals.empty()) {
                std::sort(intervals.begin(), intervals.end());
                int curr_L = intervals[0].first;
                int curr_R = intervals[0].second;

                for (size_t i = 1; i < intervals.size(); ++i) {
                    if (intervals[i].first > curr_R + 1) {
                        std::fill(row_it + curr_L, row_it + curr_R + 1, 1);
                        curr_L = intervals[i].first;
                        curr_R = intervals[i].second;
                    } else {
                        curr_R = std::max(curr_R, intervals[i].second);
                    }
                }
                std::fill(row_it + curr_L, row_it + curr_R + 1, 1);
            }

            for (Edge & e : active) { e.x += e.dx; }
        }
        return grid;
    }
}
