// cppimport
/*
<%
cfg["compiler_args"] = ["-O3", "-Wall", "-march=native", "-std=c++17"]
cfg["dependencies"] = ["constants.h", "geometry.h", "layer.h", "paper.h", "rasterizer.h"]
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <string>
#include <cstring>

#include "geometry.h"
#include "layer.h"
#include "paper.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ backend for 2D Origami Simulator";

    py::class_<Vec2>(m, "Vec2")
        .def(py::init<double, double>())
        .def_readwrite("x", &Vec2::x)
        .def_readwrite("y", &Vec2::y)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def(py::self / double())
        .def("dot", &Vec2::dot)
        .def("cross", &Vec2::cross)
        .def("norm", &Vec2::norm)
        .def("normalized", &Vec2::normalized)
        .def("__repr__", [](const Vec2 & v) {
            return "<Vec2 (x, y) = (" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")>";
        });

    py::class_<Segment>(m, "Segment")
        .def(py::init([](Vec2 p1, Vec2 p2) {
            return Segment::create(p1, p2);
        }))
        .def_readwrite("p1", &Segment::p1)
        .def_readwrite("p2", &Segment::p2)
        .def("__repr__", [](const Segment & seg) {
            return "<Segment (" + std::to_string(seg.p1.x) + ", " + std::to_string(seg.p1.y) + "), (" +
                   std::to_string(seg.p2.x) + ", " + std::to_string(seg.p2.y) + ")>";
        });

    py::class_<Layer>(m, "Layer")
        .def(py::init([](std::vector<Vec2> vertices) {
            return Layer::create(std::move(vertices));
        }))
        .def_readonly("vertices", &Layer::vertices)
        .def("__repr__", [](const Layer & l) {
            return "<Layer with " + std::to_string(l.vertices.size()) + " vertices>";
        });

    py::class_<Paper>(m, "Paper")
        .def(py::init<>())
        .def("copy", &Paper::copy)
        .def_readonly("layers", &Paper::layers)
        .def("fold", &Paper::global_fold, "Fold the paper along a segment")
        .def("compute_bounds", &Paper::compute_bounds, "Computes the global bounds of the Paper")
        .def("rasterize", [](Paper & self, int rows, int cols, double theta) {
            std::vector<uint8_t> grid;
            {
                py::gil_scoped_release release;
                grid = self.rasterize(rows, cols, theta);
            }

            py::array_t<bool> result({rows, cols});
            auto buffer_info = result.request();
            std::memcpy(buffer_info.ptr, grid.data(), grid.size() * sizeof(uint8_t));

            return result;
        }, py::arg("rows"), py::arg("cols"), py::arg("theta") = 0)
        .def("compute_boundary_points", [](Paper & self, double max_dist) {
            auto data = [&]() {
                py::gil_scoped_release release;
                return self.compute_boundary_points(max_dist);
            }();

            std::vector<Vec2> & points = data.first;
            std::vector<size_t> & indices = data.second;

            py::array_t<double> result_points({points.size(), static_cast<size_t>(2)});
            
            py::array_t<size_t> result_indices(indices.size());

            std::memcpy(result_points.mutable_data(), points.data(), points.size() * sizeof(Vec2));
            std::memcpy(result_indices.mutable_data(), indices.data(), indices.size() * sizeof(size_t));

            return py::make_tuple(result_points, result_indices);
        })
        .def("__repr__", [](const Paper & self) {
            return "<Paper object with " + std::to_string(self.layers.size()) + " layers>";
        });
}
