# Testing Guide for JARVIS Reactor

## Overview

JARVIS Reactor includes tests at multiple levels:
1. **MLForge C++ Tests** - Native C++ algorithm tests
2. **Python Binding Tests** - pybind11 interface tests
3. **Integration Tests** - End-to-end Python API tests

## MLForge C++ Tests

The MLForge submodule includes comprehensive C++ tests for all algorithms.

### Running MLForge Tests

```bash
# Navigate to MLForge
cd mlforge

# Build MLForge
mkdir -p build && cd build
cmake ..
make

# Run all tests
ctest --verbose

# Or run specific tests
./test_logistic_regression
./test_linear_regression
./test_algorithms
```

### Available C++ Tests

Located in `mlforge/tests/`:

- **test_logistic_regression.cpp** - Comprehensive logistic regression tests
  - OR function test
  - AND function test
  - Ridge regularization test
  - XOR (non-linearly separable) test
  - Coefficient getter/setter test
  - Edge cases: empty dataset, mismatched dimensions, single sample
  - Stress tests: large feature values, constant features
  - Unusual regularization types

- **test_linear_regression.cpp** - Linear regression tests
- **test_algorithms.cpp** - General algorithm tests

### Test Example from MLForge

Here's an example from `test_logistic_regression.cpp`:

```cpp
// Test OR Function
void testOrFunction() {
    std::vector<double> data = { 0.0, 0.0,
                                 0.0, 1.0,
                                 1.0, 0.0,
                                 1.0, 1.0 };
    ml::core::Matrix2D X(4, 2, data);
    std::vector<double> y = {0, 1, 1, 1};

    ml::algorithms::LogisticRegression model;
    model.fit(X, y);
    std::vector<int> predictions = model.predict(X);

    assert(predictions[0] == 0);
    assert(predictions[1] == 1);
    assert(predictions[2] == 1);
    assert(predictions[3] == 1);
}
```

## Python Binding Tests

Test the pybind11 interface to ensure C++ classes are properly exposed to Python.

### Running Python Tests

```bash
# Install in development mode
pip install -e .

# Run Python tests
pytest tests/test_bindings.py -v
```

### Example Python Binding Test

Create `tests/test_bindings.py`:

```python
import pytest
from reactor_core import reactor_core_native


def test_module_import():
    """Test that the native module can be imported"""
    assert reactor_core_native.__version__ == "1.0.0"


def test_matrix_creation():
    """Test Matrix class creation"""
    mat = reactor_core_native.Matrix(3, 3)
    assert mat.rows() == 3
    assert mat.cols() == 3


def test_info_function():
    """Test info function"""
    info = reactor_core_native.info()
    assert "Reactor Core" in info
    assert "MLForge" in info
```

## Integration Tests

Test the full Python API including Reactor Core training workflows.

### Running Integration Tests

```bash
pytest tests/test_integration.py -v
```

### Example Integration Test

Create `tests/test_integration.py`:

```python
import pytest
from reactor_core import Trainer, TrainingConfig


def test_trainer_initialization():
    """Test Trainer can be initialized"""
    config = TrainingConfig(
        model_name="test-model",
        num_epochs=1,
    )
    trainer = Trainer(config)
    assert trainer.config.model_name == "test-model"


def test_environment_detection():
    """Test environment detection"""
    from reactor_core.utils import detect_environment

    env_info = detect_environment()
    assert env_info.env_type is not None
    assert env_info.cpu_arch is not None
    assert env_info.total_ram_gb > 0
```

## Test Coverage

### Current Coverage

- **MLForge C++ Core:** ✅ Comprehensive (see `mlforge/tests/`)
- **Python Bindings:** 🚧 Minimal (placeholder tests)
- **Python API:** ✅ Basic tests available
- **Integration:** 🚧 In progress

### Running Coverage Reports

```bash
# Python coverage
pytest --cov=reactor_core --cov-report=html tests/

# View report
open htmlcov/index.html
```

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test-cpp:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          submodules: recursive

      - name: Install CMake
        run: sudo apt-get install cmake

      - name: Build MLForge
        run: |
          cd mlforge
          mkdir build && cd build
          cmake ..
          make

      - name: Run C++ Tests
        run: |
          cd mlforge/build
          ctest --verbose

  test-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          submodules: recursive

      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install pybind11 cmake pytest pytest-cov
          pip install -e .

      - name: Run Python tests
        run: pytest tests/ -v --cov=reactor_core
```

## Manual Testing

### Test C++ Compilation

```bash
# Test that MLForge builds
cd mlforge
mkdir -p build && cd build
cmake ..
make
```

### Test Python Bindings

```bash
# Build and install
pip install -e .

# Quick test
python -c "from reactor_core import reactor_core_native; print(reactor_core_native.info())"
```

### Test Environment Detection

```bash
python -c "from reactor_core.utils import print_environment_info; print_environment_info()"
```

## Performance Benchmarks

Compare Python vs C++ performance:

```python
import time
import numpy as np
from reactor_core.reactor_core_native import Matrix

# Benchmark matrix operations
def benchmark_matrix_multiply():
    size = 1000
    arr = np.random.rand(size, size)

    # Python (NumPy)
    start = time.time()
    result_np = arr @ arr
    numpy_time = time.time() - start

    # C++ (MLForge) - when implemented
    # mat = Matrix.from_numpy(arr)
    # start = time.time()
    # result_cpp = mat.multiply(mat)
    # cpp_time = time.time() - start

    print(f"NumPy time: {numpy_time:.4f}s")
    # print(f"MLForge time: {cpp_time:.4f}s")
    # print(f"Speedup: {numpy_time/cpp_time:.2f}x")
```

## Troubleshooting

### Tests Fail to Run

```bash
# Reinitialize submodule
git submodule update --init --recursive

# Rebuild everything
pip install -e . --force-reinstall
```

### C++ Compilation Errors

```bash
# Check CMake version
cmake --version  # Should be >= 3.15

# Clean build
rm -rf build/
mkdir build && cd build
cmake ..
make
```

### Python Import Errors

```bash
# Verify installation
pip show jarvis-reactor

# Check import path
python -c "import reactor_core; print(reactor_core.__file__)"
```

## Contributing Tests

When adding new features, please include:

1. **C++ tests** in `mlforge/tests/`
2. **Python binding tests** in `tests/test_bindings.py`
3. **Integration tests** in `tests/test_integration.py`

Follow the pattern in `mlforge/tests/test_logistic_regression.cpp` for comprehensive test coverage.

---

**Reference:** See `mlforge/tests/test_logistic_regression.cpp` for an example of comprehensive C++ testing.
