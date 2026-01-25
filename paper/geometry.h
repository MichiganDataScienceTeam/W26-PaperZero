#pragma once
#include <cmath>
#include <stdexcept>

#include "constants.h"

/**
 * 2d vector with basic vector arithmetic and utilities
 */
struct Vec2 {
    double x, y;

    Vec2 operator+(Vec2 other) const {
        return {x + other.x, y + other.y};
    }

    Vec2 operator-(Vec2 other) const {
        return {x - other.x, y - other.y};
    }

    Vec2 operator*(double c) const {
        return {x * c, y * c};
    }

    Vec2 operator/(double c) const {
        return {x / c, y / c};
    }

    double dot(Vec2 other) const {
        return (x * other.x) + (y * other.y);
    }

    double cross(Vec2 other) const {
        return (x * other.y) - (y * other.x);
    }

    double norm() const {
        return std::sqrt(x * x + y * y);
    }

    Vec2 normalized() const {
        double n = norm();
        return {x / n, y / n};
    }

    Vec2 right_orth() const {
        return {y, -x};
    }
};

Vec2 operator*(double c, Vec2 other) {
    return other * c;
}

// Forward declare Line
struct Line;

/**
 * Line segment defined by endpoints
 */
struct Segment {
    Vec2 p1, p2;

    static Segment create(Vec2 p1, Vec2 p2) {
        if ((p1 - p2).norm() < Origami::EPSILON) {
            throw std::invalid_argument("Segment endpoints must be distinct.");
        }
        return Segment{p1, p2};
    }

    bool intersect(const Line & other, Vec2 & result) const;
    bool intersect(const Segment & other, Vec2 & result) const;
};


/**
 * Line defined by dot(n, <x, y>) = d
 * n: unit normal vector of the line
 * d: distance from (0, 0) to the line
 */
struct Line {
    Vec2 n;
    double d;

    Line(Vec2 n_in, double d_in) : n(n_in), d(d_in) {}

    Line(Vec2 p1, Vec2 p2) {
        Vec2 direction = p1 - p2;
        n = direction.right_orth().normalized();
        d = n.dot(p1);
    }

    Line(const Segment & s) : Line(s.p1, s.p2) {}

    bool intersect(const Line & other, Vec2 & result) const {
        if (std::abs(n.cross(other.n)) < Origami::EPSILON) {
            return false; // Exclude all parallel lines (including identical)
        }

        Vec2 p = d * n;
        Vec2 r = n.right_orth();

        result = p + (other.d - p.dot(other.n)) / r.dot(other.n) * r;
        return true;
    }

    bool intersect(const Segment & other, Vec2 & result) const {
        return intersect(Line(other), result) &&
               (result - other.p1).dot(result - other.p2) < Origami::EPSILON;
    }
};

bool Segment::intersect(const Line & other, Vec2 & result) const {
    return other.intersect(*this, result);
}

bool Segment::intersect(const Segment & other, Vec2 & result) const {
    return intersect(Line(other), result) &&
            (result - other.p1).dot(result - other.p2) < Origami::EPSILON;
}

// For bounding
struct Rect {
    double min_x, max_x, min_y, max_y;
};
