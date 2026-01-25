#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "constants.h"
#include "geometry.h"


/**
 * One layer of paper, MUST be a convex counterclockwise polygon
 */
class Layer {
public:
    std::vector<Vec2> vertices; // MUST be convex and counterclockwise

    Layer(std::vector<Vec2> vertices_in) : vertices(std::move(vertices_in)) {}

    bool should_reflect(const Line & seg_line) const {
        return seg_line.n.dot(centroid(vertices)) < seg_line.d - Origami::EPSILON;
    }

    std::vector<Layer> split(const Segment & seg, int & intersections) const {
        std::vector<Vec2> vertices1;
        std::vector<Vec2> vertices2;
        intersections = 0;
        bool is_first = true;

        const size_t N = vertices.size();

        for (size_t i = 0; i < N; i++) {
            const Vec2 p1 = vertices[i];
            const Vec2 p2 = vertices[(i + 1) % N];
            const Segment edge_seg{p1, p2};

            if (is_first) {
                vertices1.push_back(p1);
            } else {
                vertices2.push_back(p1);
            }

            Vec2 intersection;
            if (seg.intersect(edge_seg, intersection)) {
                if ((intersection - p1).norm() < Origami::EPSILON) {
                    continue;
                } else if ((intersection - p2).norm() < Origami::EPSILON) {
                    if (is_first) {
                        vertices1.push_back(p2);
                    } else {
                        vertices2.push_back(p2);
                    }
                } else {
                    vertices1.push_back(intersection);
                    vertices2.push_back(intersection);
                }
                is_first = !is_first;
                intersections++;
            }
        }

        std::vector<Layer> ans;
        ans.reserve(2);

        if (!is_first) {
            ans.emplace_back(vertices);
        } else {
            if (area(vertices1) >= Origami::MIN_AREA) {
                ans.emplace_back(std::move(vertices1));
            }

            if (area(vertices2) >= Origami::MIN_AREA) {
                ans.emplace_back(std::move(vertices2));
            }
        }

        return ans;
    }

    static Layer create(std::vector<Vec2> v) {
        if (v.size() < 3) {
            throw std::invalid_argument("Layer must have at least 3 vertices.");
        }

        const double area = signed_area(v);
        if (std::abs(area) < Origami::MIN_AREA) {
            throw std::invalid_argument("Layer area is too small.");
        }

        if (area < 0) {
            std::reverse(v.begin(), v.end());
        }

        if (!is_convex(v)) {
            throw std::invalid_argument("Layer polygon must be convex.");
        }

        return Layer(std::move(v));
    }

    void reflect(const Line & l) {
        for (size_t i = 0; i < vertices.size(); i++) {
            const double dist = vertices[i].dot(l.n) - l.d;
            vertices[i] = vertices[i] - 2 * dist * l.n;
        }

        std::reverse(vertices.begin(), vertices.end());
    }

    bool contains_point(const Vec2 & p) const {
        const size_t N = vertices.size();

        for (size_t i = 0; i < N; i++) {
            const Vec2 edge = vertices[(i+1)%N] - vertices[i];
            const Vec2 edge_to_point = p - vertices[i];

            if (edge.right_orth().dot(edge_to_point) > Origami::EPSILON) {
                return false;
            }
        }

        return true;
    }

    bool is_on_boundary(const Vec2 & p) const {
        const size_t N = vertices.size();

        for (size_t i = 0; i < N; i++) {
            const Vec2 v1 = vertices[i];
            const Vec2 v2 = vertices[(i+1)%N];
            const Vec2 edge = (v2 - v1).normalized();
            const Vec2 edge_to_point = (p - v1).normalized();

            const double component = std::abs(edge.dot(edge_to_point));

            if ((component < 1-Origami::EPSILON) || (component > 1+Origami::EPSILON)) {
                continue;
            }

            const double min_x = std::min(v1.x, v2.x);
            const double max_x = std::max(v1.x, v2.x);
            const double min_y = std::min(v1.y, v2.y);
            const double max_y = std::max(v1.y, v2.y);
            if ((p.x < min_x - Origami::EPSILON) || (p.x > max_x + Origami::EPSILON)) {
                continue;
            } else if ((p.y < min_y - Origami::EPSILON) || (p.y > max_y + Origami::EPSILON)) {
                continue;
            }

            return true;
        }
        return false;
    }

    std::vector<Segment> subtract(const Segment & s) const {
        double t_enter = 0;
        double t_exit = 1;

        Vec2 dir = s.p2 - s.p1;
        if (dir.norm() < Origami::EPSILON) return {};

        for (size_t i = 0; i < vertices.size(); i++) {
            size_t j = (i + 1) % vertices.size();

            // Normal linalg
            const Vec2 edge_vec = vertices[j] - vertices[i];
            const Vec2 normal = edge_vec.right_orth(); 

            // Turns out you can just project everything and it works
            double num = (vertices[i] - s.p1).dot(normal);
            double denom = dir.dot(normal);

            // Parallel case
            if (std::abs(denom) < Origami::EPSILON) {
                if (num < Origami::EPSILON) { 
                    return {s}; 
                }
                continue; 
            } 
            
            // Normal case
            double t = num / denom;
            if (denom < 0) {
                t_enter = std::max(t_enter, t);
            } else {
                t_exit = std::min(t_exit, t);
            }
            if (t_enter > t_exit) return {s};
        }

        std::vector<Segment> result;
        result.reserve(2);
        if (t_enter > Origami::EPSILON) {
            result.push_back({s.p1, s.p1 + dir * t_enter});
        }
        if (t_exit < 1.0 - Origami::EPSILON) {
            result.push_back({s.p1 + dir * t_exit, s.p2});
        }
        return result;
    }

private:
    static Vec2 centroid(const std::vector<Vec2> & polygon) {
        Vec2 ans = {0, 0};

        for (const Vec2 & v : polygon) {
            ans.x += v.x;
            ans.y += v.y;
        }

        ans.x /= polygon.size();
        ans.y /= polygon.size();

        return ans;
    }

    static bool is_convex(const std::vector<Vec2> & polygon) {
        const size_t N = polygon.size();
        if (N < 3) return false;

        bool has_pos = false;
        bool has_neg = false;

        for (size_t i = 0; i < N; i++) {
            const Vec2 edge1 = polygon[(i + 1) % N] - polygon[i];
            const Vec2 edge2 = polygon[(i + 2) % N] - polygon[(i + 1) % N];

            const double cross = edge1.cross(edge2);
            if (cross > Origami::EPSILON) { has_pos = true; }
            if (cross < -Origami::EPSILON) { has_neg = true; }

            if (has_pos && has_neg) { return false; }
        }
        return true;
    }

    static double signed_area(const std::vector<Vec2> & polygon) {
        double area = 0;
        const size_t N = polygon.size();

        for (size_t i = 0; i < N; i++) {
            area += (polygon[i].x * polygon[(i + 1) % N].y);
            area -= (polygon[(i + 1) % N].x * polygon[i].y);
        }

        return area / 2;
    }

    static double area(const std::vector<Vec2> & polygon) {
        return std::abs(signed_area(polygon));
    }
};

Layer new_unit_square() {
    return Layer({Vec2{0, 0}, Vec2{1, 0}, Vec2{1, 1}, Vec2{0, 1}});
}
