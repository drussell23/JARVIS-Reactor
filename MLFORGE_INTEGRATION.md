# MLForge C++ Integration Guide

## Overview

Reactor Core integrates **MLForge** as a git submodule to provide high-performance C++ ML primitives accessible from Python via pybind11 bindings.

## Architecture

```
reactor-core/
├── mlforge/                    # Git submodule → MLForge C++ core
│   ├── include/ml/
│   │   ├── core/              # Matrix, utils, data structures
│   │   ├── algorithms/         # LinearRegression, NeuralNet, etc.
│   │   ├── ai/                # Transformers, RL, Quantum ML
│   │   ├── serialization/     # Model serialization
│   │   └── deployment/        # Model server, API
│   ├── src/
│   ├── tests/
│   └── CMakeLists.txt
│
├── bindings/
│   └── reactor_bindings.cpp   # Python bindings (pybind11)
│
├── CMakeLists.txt              # Build configuration
├── setup.py                    # Python build script
└── reactor_core/               # Python package
```

## Files Copied from MLForge

### 1. **Git Submodule** (Entire MLForge repo)
- **Location:** `mlforge/`
- **Source:** https://github.com/drussell23/MLForge
- **Contents:**
  - C++ headers in `mlforge/include/ml/`
  - C++ implementations in `mlforge/src/`
  - Tests in `mlforge/tests/`
  - CMake build files

### 2. **Build Configuration** (Adapted)
- **File:** `CMakeLists.txt`
- **Based on:** MLForge's `CMakeLists.txt`
- **Modifications:**
  - Added pybind11 integration
  - Configured Python module build
  - Set up linking against MLForgeLib

### 3. **C++ Components Available**

From MLForge's `include/ml/`:

#### Core Components
- `core/matrix.h` - Matrix operations
- `core/utils.h` - Utility functions
- `core/data_structures/kd_tree.h` - KD-tree
- `core/data_structures/graph_structures.h` - Graph structures
- `core/data_structures/trie.h` - Trie

#### Algorithms
- `algorithms/linear_regression.h` - Linear regression
- `algorithms/logistic_regression.h` - Logistic regression
- `algorithms/neural_net.h` - Neural networks
- `algorithms/decision_tree.h` - Decision trees

#### AI Components
- `ai/nlp_transformer.h` - NLP transformers
- `ai/reinforcement_learning.h` - Reinforcement learning
- `ai/quantum_ml.h` - Quantum ML

#### Serialization & Deployment
- `serialization/serializer.h` - Model serialization
- `deployment/model_server.h` - Model server
- `deployment/api.h` - API utilities

## Building from Source

### Prerequisites
```bash
# macOS
brew install cmake pybind11

# Ubuntu/Debian
sudo apt-get install cmake python3-pybind11
```

### Clone with Submodules
```bash
git clone --recursive https://github.com/drussell23/reactor-core.git
cd reactor-core
```

### Build C++ Bindings
```bash
# Install Python dependencies
pip install pybind11

# Build and install
pip install -e .
```

### Verify Installation
```python
import reactor_core
from reactor_core import reactor_core_native

# Check version
print(reactor_core_native.__version__)
print(reactor_core_native.info())

# Test Matrix binding (placeholder)
mat = reactor_core_native.Matrix(3, 3)
print(mat)  # <Matrix 3x3>
```

## Python Bindings Status

### ✅ Implemented
- [x] Module structure
- [x] Version info
- [x] Placeholder Matrix class

### 🚧 In Progress
- [ ] MLForge Matrix bindings
- [ ] Linear regression bindings
- [ ] Neural network bindings
- [ ] Serialization bindings

### 📋 Planned
- [ ] Full algorithm suite
- [ ] NLP transformer bindings
- [ ] Model server integration
- [ ] Quantum ML bindings

## Usage Examples

### Using MLForge Matrix (Planned)
```python
from reactor_core.reactor_core_native import Matrix
import numpy as np

# Create matrix from NumPy array
arr = np.array([[1, 2], [3, 4]])
mat = Matrix.from_numpy(arr)

# Perform operations (C++ accelerated)
result = mat.multiply(mat)

# Convert back to NumPy
result_np = result.to_numpy()
```

### Training with C++ Acceleration (Planned)
```python
from reactor_core import Trainer, TrainingConfig
from reactor_core.reactor_core_native import LinearRegression

# Use C++ backend for linear regression
config = TrainingConfig(
    model_name="custom-regression",
    use_cpp_backend=True,  # Use MLForge C++
)

trainer = Trainer(config)
trainer.train("./data/regression.csv")
```

## Development

### Adding New Bindings

1. **Edit `bindings/reactor_bindings.cpp`**
```cpp
#include "ml/algorithms/linear_regression.h"

PYBIND11_MODULE(reactor_core_native, m) {
    // Add new binding
    py::class_<ml::LinearRegression>(m, "LinearRegression")
        .def(py::init<>())
        .def("fit", &ml::LinearRegression::fit)
        .def("predict", &ml::LinearRegression::predict);
}
```

2. **Rebuild**
```bash
pip install -e . --force-reinstall
```

3. **Test**
```python
from reactor_core.reactor_core_native import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Running MLForge Tests
```bash
cd mlforge
mkdir build && cd build
cmake ..
make
ctest  # Run C++ tests
```

## Performance Benefits

| Operation | Pure Python | MLForge C++ | Speedup |
|-----------|-------------|-------------|---------|
| Matrix multiply (1000x1000) | 250ms | 15ms | **16.7x** |
| Linear regression fit | 180ms | 12ms | **15x** |
| Neural net forward pass | 320ms | 25ms | **12.8x** |

*(Benchmarks are estimates - actual results TBD)*

## Troubleshooting

### Submodule not initialized
```bash
git submodule update --init --recursive
```

### CMake can't find pybind11
```bash
pip install pybind11[global]
```

### Build fails on macOS
```bash
# Set macOS SDK explicitly
export CMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
pip install -e .
```

## Contributing

To contribute MLForge bindings:

1. Fork both `reactor-core` and `MLForge`
2. Make changes to MLForge C++ code
3. Update bindings in `reactor-core/bindings/`
4. Submit PRs to both repos

## References

- **MLForge Repository:** https://github.com/drussell23/MLForge
- **pybind11 Documentation:** https://pybind11.readthedocs.io/
- **Reactor Core:** https://github.com/drussell23/reactor-core

---

**Status:** 🚧 Work in Progress

MLForge integration is functional but bindings are minimal. Full C++ acceleration coming soon!
