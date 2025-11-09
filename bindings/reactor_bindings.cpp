/**
 * Python bindings for Reactor Core (MLForge C++ backend)
 *
 * This module provides Python access to MLForge's high-performance C++ ML primitives
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// MLForge includes (will be available once MLForge is built)
// #include "ml/core/matrix.h"
// #include "ml/algorithms/linear_regression.h"
// #include "ml/algorithms/neural_net.h"

namespace py = pybind11;

// Placeholder bindings until MLForge headers are integrated
class MatrixWrapper {
public:
    MatrixWrapper(int rows, int cols) : rows_(rows), cols_(cols) {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }

private:
    int rows_;
    int cols_;
};

PYBIND11_MODULE(reactor_core_native, m) {
    m.doc() = "Reactor Core native bindings to MLForge C++ engine";

    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("mlforge_version") = "1.0.0";

    // Placeholder Matrix class
    py::class_<MatrixWrapper>(m, "Matrix")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
        .def("rows", &MatrixWrapper::rows)
        .def("cols", &MatrixWrapper::cols)
        .def("__repr__", [](const MatrixWrapper &mat) {
            return "<Matrix " + std::to_string(mat.rows()) + "x" +
                   std::to_string(mat.cols()) + ">";
        });

    // TODO: Add bindings for MLForge components:
    // - ml::core::Matrix
    // - ml::algorithms::LinearRegression
    // - ml::algorithms::LogisticRegression
    // - ml::algorithms::NeuralNet
    // - ml::algorithms::DecisionTree
    // - ml::serialization::Serializer
    // - ml::deployment::ModelServer

    m.def("info", []() {
        return "Reactor Core v1.0.0 - MLForge C++ backend integration";
    });
}
