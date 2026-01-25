#pragma once
#include <vector>
#include <unordered_set>
#include <queue>
#include <algorithm>

#include "geometry.h"
#include "layer.h"
#include "rasterizer.h"

// Temporary (maybe) implementation detail
struct EdgeConnection {
    size_t neighbor_layer;
    Vec2 v1, v2;
};

/**
 * A single (possibly folded) sheet of paper
 */
class Paper {
public:
    std::vector<Layer> layers;
    std::vector<std::vector<EdgeConnection>> adjacency;

    Paper() : layers({new_unit_square()}), adjacency(1) {}

    Paper copy() const {
        return *this;
    }

    bool global_fold(Segment s) {
        // Verify fold doesn't cause obvious rips
        for (const Layer & layer : layers) {
            if ((layer.contains_point(s.p1) && !layer.is_on_boundary(s.p1)) ||
                (layer.contains_point(s.p2) && !layer.is_on_boundary(s.p2))) {
                return false;
            }
        }

        const Line l(s);

        // Track which old layer each new layer came from
        std::vector<size_t> parent_indices;
        std::vector<Layer> new_layers;
        std::vector<std::vector<EdgeConnection>> new_adjacency;

        // Split all layers and group pieces by parent
        std::vector<std::vector<size_t>> parent_to_pieces(layers.size());
        
        for (size_t parent_idx = 0; parent_idx < layers.size(); ++parent_idx) {
            int intersections;
            std::vector<Layer> pieces = layers[parent_idx].split(s, intersections);
            
            for (const Layer & piece : pieces) {
                size_t new_idx = new_layers.size();
                parent_indices.push_back(parent_idx);
                parent_to_pieces[parent_idx].push_back(new_idx);
                new_layers.push_back(piece);
                new_adjacency.push_back({});
            }
        }

        // Connect pieces from the same split (they share the cut edge)
        for (size_t parent_idx = 0; parent_idx < layers.size(); ++parent_idx) {
            const std::vector<size_t> & pieces = parent_to_pieces[parent_idx];
            if (pieces.size() == 2) {
                // Find the shared edge along the fold line
                std::vector<Vec2> shared_verts;
                for (const Vec2 & v1 : new_layers[pieces[0]].vertices) {
                    for (const Vec2 & v2 : new_layers[pieces[1]].vertices) {
                        if ((v1 - v2).norm() < Origami::EPSILON) {
                            shared_verts.push_back(v1);
                            break;
                        }
                    }
                }
                
                if (shared_verts.size() >= 2) {
                    Vec2 edge_v1 = shared_verts[0];
                    Vec2 edge_v2 = shared_verts[1];
                    
                    // Canonicalize endpoint order
                    if (edge_v1.x > edge_v2.x || 
                        (std::abs(edge_v1.x - edge_v2.x) < Origami::EPSILON && edge_v1.y > edge_v2.y)) {
                        std::swap(edge_v1, edge_v2);
                    }
                    
                    new_adjacency[pieces[0]].push_back({pieces[1], edge_v1, edge_v2});
                    new_adjacency[pieces[1]].push_back({pieces[0], edge_v1, edge_v2});
                }
            }
        }

        // Inherit connections from parent layers
        for (size_t i = 0; i < new_layers.size(); ++i) {
            size_t parent_i = parent_indices[i];
            
            for (const EdgeConnection & old_conn : adjacency[parent_i]) {
                size_t old_neighbor = old_conn.neighbor_layer;
                
                // Find the actual edge segment on this piece that overlaps with old connection
                Vec2 my_v1, my_v2;
                if (!find_overlapping_segment(new_layers[i], old_conn.v1, old_conn.v2, my_v1, my_v2)) {
                    continue;
                }
                
                // Find which piece(s) of the neighbor also have overlapping edges
                for (size_t j : parent_to_pieces[old_neighbor]) {
                    Vec2 neighbor_v1, neighbor_v2;
                    if (find_overlapping_segment(new_layers[j], old_conn.v1, old_conn.v2, neighbor_v1, neighbor_v2)) {
                        // Compute canonical shared segment (intersection of both clips)
                        Vec2 shared_v1, shared_v2;
                        if (!compute_segment_intersection(my_v1, my_v2, neighbor_v1, neighbor_v2, shared_v1, shared_v2)) {
                            continue; // No overlap between the two clipped segments
                        }
                        
                        // Canonicalize endpoint order (smaller x first, or smaller y if x equal)
                        if (shared_v1.x > shared_v2.x || 
                            (std::abs(shared_v1.x - shared_v2.x) < Origami::EPSILON && shared_v1.y > shared_v2.y)) {
                            std::swap(shared_v1, shared_v2);
                        }
                        
                        // Store the same canonical geometry for both layers
                        new_adjacency[i].push_back({j, shared_v1, shared_v2});
                        new_adjacency[j].push_back({i, shared_v1, shared_v2});
                    }
                }
            }
        }

        // Remove duplicate connections
        for (auto & conns : new_adjacency) {
            // First canonicalize all edge endpoints
            for (EdgeConnection & conn : conns) {
                if (conn.v1.x > conn.v2.x || 
                    (std::abs(conn.v1.x - conn.v2.x) < Origami::EPSILON && conn.v1.y > conn.v2.y)) {
                    std::swap(conn.v1, conn.v2);
                }
            }
            
            std::sort(conns.begin(), conns.end(), [](const EdgeConnection & a, const EdgeConnection & b) {
                if (a.neighbor_layer != b.neighbor_layer) return a.neighbor_layer < b.neighbor_layer;
                if (std::abs(a.v1.x - b.v1.x) > Origami::EPSILON) return a.v1.x < b.v1.x;
                if (std::abs(a.v1.y - b.v1.y) > Origami::EPSILON) return a.v1.y < b.v1.y;
                if (std::abs(a.v2.x - b.v2.x) > Origami::EPSILON) return a.v2.x < b.v2.x;
                return a.v2.y < b.v2.y;
            });
            conns.erase(std::unique(conns.begin(), conns.end(), [](const EdgeConnection & a, const EdgeConnection & b) {
                return a.neighbor_layer == b.neighbor_layer &&
                       (a.v1 - b.v1).norm() < Origami::EPSILON &&
                       (a.v2 - b.v2).norm() < Origami::EPSILON;
            }), conns.end());
        }

        // Find layers touched by the fold segment (including interior points)
        std::unordered_set<size_t> touched_layers;
        for (size_t i = 0; i < new_layers.size(); ++i) {
            if (layer_touches_segment(new_layers[i], s)) {
                touched_layers.insert(i);
            }
        }

        // BFS to find connected component, stopping at static-side boundaries
        std::unordered_set<size_t> connected_component;
        if (!touched_layers.empty()) {
            std::queue<size_t> q;
            for (size_t idx : touched_layers) {
                if (new_layers[idx].should_reflect(l)) {
                    q.push(idx);
                    connected_component.insert(idx);
                }
            }

            while (!q.empty()) {
                size_t curr = q.front();
                q.pop();

                for (const EdgeConnection & conn : new_adjacency[curr]) {
                    size_t neighbor = conn.neighbor_layer;
                    
                    // Only traverse to neighbors on the active side
                    if (connected_component.count(neighbor) == 0 && 
                        new_layers[neighbor].should_reflect(l)) {
                        connected_component.insert(neighbor);
                        q.push(neighbor);
                    }
                }
            }
        }

        // Reflect all layers in connected component and update ALL edge records
        for (size_t i : connected_component) {
            new_layers[i].reflect(l);
        }
        
        // Update edge vertex positions in the entire graph
        for (size_t i = 0; i < new_adjacency.size(); ++i) {
            for (EdgeConnection & conn : new_adjacency[i]) {
                // If either endpoint was reflected, update the edge
                if (connected_component.count(i) > 0 || 
                    connected_component.count(conn.neighbor_layer) > 0) {
                    
                    // Reflect vertices if they were on a reflected layer
                    if (connected_component.count(i) > 0) {
                        // Check if vertices lie on the fold line (identity case)
                        double dist1 = conn.v1.dot(l.n) - l.d;
                        double dist2 = conn.v2.dot(l.n) - l.d;
                        
                        if (std::abs(dist1) > Origami::EPSILON) {
                            conn.v1 = conn.v1 - 2 * dist1 * l.n;
                        }
                        if (std::abs(dist2) > Origami::EPSILON) {
                            conn.v2 = conn.v2 - 2 * dist2 * l.n;
                        }
                    }
                }
            }
        }

        new_layers.swap(layers);
        new_adjacency.swap(adjacency);

        return true;
    }

    std::vector<double> compute_bounds() const {
        Rect temp = Rasterizer::compute_bounds(layers, 0, 1);
        return {temp.min_x, temp.max_x, temp.min_y, temp.max_y};
    }

    std::vector<uint8_t> rasterize(int rows, int cols, double theta) const {
        return Rasterizer::render(layers, rows, cols, theta);
    }

    std::vector<Vec2> compute_boundary_points(double max_dist) const {
        std::vector<Vec2> result;
        for (const Layer & layer : layers) {
            std::vector<Segment> temp;
            for (size_t i = 0; i < layer.vertices.size(); i++) {
                size_t j = (i+1) % layer.vertices.size();
                temp.push_back({layer.vertices[i], layer.vertices[j]});
            }

            for (const Layer & layer2 : layers) {
                if (&layer == &layer2) { continue; }
                std::vector<Segment> temp2;
                for (const Segment & s: temp) {
                    for (const Segment & s2 : layer2.subtract(s)) {
                        temp2.push_back(s2);
                    }
                }
                temp = std::move(temp2);
                if (temp.empty()) { break; }
            }

            for (const Segment & s : temp) {
                result.push_back(s.p1);
                result.push_back(s.p2);

                const double len = (s.p1 - s.p2).norm();
                if (len > max_dist) {
                    const int intermediates = std::ceil(len/max_dist);
                    const Vec2 dir = s.p2 - s.p1;
                    for (int i = 1; i < intermediates; i++) {
                        result.push_back(s.p1 + (dir*(static_cast<double>(i)/intermediates)));
                    }
                }
            }
        }
        std::sort(result.begin(), result.end(), [](Vec2 p1, Vec2 p2) {
            return p1.x < p2.x;
        });

        auto write = result.begin();
        for (auto it = result.begin() + 1; it != result.end(); ++it) {
            bool is_duplicate = false;
            for (auto checker = write; checker >= result.begin(); --checker) {
                if ((it->x - checker->x) > Origami::EPSILON) { break; }
                if ((*it - *checker).norm() <= Origami::EPSILON) {
                    is_duplicate = true;
                }
            }
            if (!is_duplicate) {
                ++write;
                *write = *it;
            }
        }
        result.erase(write+1, result.end());
        return result;
    }

private:
    // Compute the intersection of two collinear segments
    bool compute_segment_intersection(Vec2 a1, Vec2 a2, Vec2 b1, Vec2 b2, Vec2 & out_v1, Vec2 & out_v2) const {
        // Project onto a common direction to find overlap
        Vec2 dir = (a2 - a1);
        double len = dir.norm();
        if (len < Origami::EPSILON) {
            return false;
        }
        dir = dir / len;
        
        // Project all points onto this direction
        double t_a1 = 0.0;
        double t_a2 = len;
        double t_b1 = (b1 - a1).dot(dir);
        double t_b2 = (b2 - a1).dot(dir);
        
        if (t_b1 > t_b2) std::swap(t_b1, t_b2);
        
        // Find overlap interval
        double overlap_start = std::max(t_a1, t_b1);
        double overlap_end = std::min(t_a2, t_b2);
        
        if (overlap_end < overlap_start + Origami::EPSILON) {
            return false; // No overlap
        }
        
        out_v1 = a1 + overlap_start * dir;
        out_v2 = a1 + overlap_end * dir;
        return true;
    }

    // Find the segment on the layer that overlaps with v1-v2, return clipped endpoints
    bool find_overlapping_segment(const Layer & layer, Vec2 v1, Vec2 v2, Vec2 & out_v1, Vec2 & out_v2) const {
        double seg_len = (v2 - v1).norm();
        if (seg_len < Origami::EPSILON) {
            return false; // Degenerate segment
        }
        
        Vec2 seg_dir = (v2 - v1) / seg_len; // Normalized direction
        
        for (size_t i = 0; i < layer.vertices.size(); ++i) {
            Vec2 edge_v1 = layer.vertices[i];
            Vec2 edge_v2 = layer.vertices[(i + 1) % layer.vertices.size()];
            
            double edge_len = (edge_v2 - edge_v1).norm();
            if (edge_len < Origami::EPSILON) {
                continue; // Skip degenerate edges
            }
            
            // Check if edge is collinear with the segment (strict tolerance)
            Vec2 edge_dir = (edge_v2 - edge_v1) / edge_len;
            double parallelism = std::abs(seg_dir.dot(edge_dir));
            if (parallelism < 1 - Origami::EPSILON * 0.1) {
                continue; // Not parallel enough
            }
            
            // Check if edge_v1 lies on the line through v1-v2 (strict collinearity)
            Vec2 to_edge = edge_v1 - v1;
            double cross = std::abs(seg_dir.cross(to_edge));
            if (cross > Origami::EPSILON * 0.1) {
                continue; // Not collinear
            }
            
            // Project edge endpoints onto the v1-v2 line segment
            double t1 = (edge_v1 - v1).dot(seg_dir);
            double t2 = (edge_v2 - v1).dot(seg_dir);
            
            if (t1 > t2) std::swap(t1, t2);
            
            // Check if [t1, t2] overlaps with [0, seg_len]
            double overlap_start = std::max(0.0, t1);
            double overlap_end = std::min(seg_len, t2);
            
            if (overlap_end > overlap_start + Origami::EPSILON) {
                // Compute the clipped segment endpoints
                out_v1 = v1 + overlap_start * seg_dir;
                out_v2 = v1 + overlap_end * seg_dir;
                return true;
            }
        }
        return false;
    }

    // Check if a layer has an edge with the given vertices (in either order)
    bool layer_has_edge(const Layer & layer, Vec2 v1, Vec2 v2) const {
        for (size_t i = 0; i < layer.vertices.size(); ++i) {
            Vec2 edge_v1 = layer.vertices[i];
            Vec2 edge_v2 = layer.vertices[(i + 1) % layer.vertices.size()];
            
            bool forward = (edge_v1 - v1).norm() < Origami::EPSILON && 
                          (edge_v2 - v2).norm() < Origami::EPSILON;
            bool backward = (edge_v1 - v2).norm() < Origami::EPSILON && 
                           (edge_v2 - v1).norm() < Origami::EPSILON;
            
            if (forward || backward) {
                return true;
            }
        }
        return false;
    }

    bool layer_touches_segment(const Layer & layer, const Segment & seg) const {
        // Check if segment endpoints are in or on the layer
        if (layer.contains_point(seg.p1) || layer.contains_point(seg.p2)) {
            return true;
        }

        // Check for edge intersections
        for (size_t i = 0; i < layer.vertices.size(); ++i) {
            Vec2 v1 = layer.vertices[i];
            Vec2 v2 = layer.vertices[(i + 1) % layer.vertices.size()];
            Segment edge{v1, v2};
            
            Vec2 intersection;
            if (seg.intersect(edge, intersection)) {
                return true;
            }
        }

        return false;
    }
};