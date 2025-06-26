# Dtype Fallback Tests

## Overview

The dtype fallback tests verify the automatic dtype compatibility handling implemented in `train.py` to resolve BFloat16 support issues across different hardware configurations.

## Tests Added

### `test_dtype_fallback_behavior()`
**Location**: `tests/test_train.py`

**Purpose**: Tests the core dtype fallback logic across different hardware scenarios.

**Test Cases**:
1. **CUDA with BFloat16 support** → Keep `bfloat16`
2. **CUDA without BFloat16 support** → Fallback to `float16`
3. **MPS device** → Fallback to `float32`
4. **CPU device** → Fallback to `float32`
5. **Non-bfloat16 dtypes** → Preserve unchanged
6. **Gradient scaler configuration** → Correct `enabled` flag based on actual dtype

### `test_dtype_fallback_with_console_output()`
**Location**: `tests/test_train.py`

**Purpose**: Tests that appropriate console warning messages are generated during dtype fallback.

**Verifies**:
- Warning messages printed when fallback occurs
- No warnings when fallback is not needed
- Correct fallback messages for each device type
- Proper `Dtype fallback: X → Y` logging

## Running the Tests

```bash
# Run both dtype fallback tests
python -m pytest tests/test_train.py::test_dtype_fallback_behavior tests/test_train.py::test_dtype_fallback_with_console_output -v

# Run individual tests
python -m pytest tests/test_train.py::test_dtype_fallback_behavior -v
python -m pytest tests/test_train.py::test_dtype_fallback_with_console_output -v
```

## Implementation Details

The tests use `unittest.mock.patch` to simulate different hardware configurations:
- Mock `torch.cuda.is_available()` for CUDA availability
- Mock `torch.cuda.is_bf16_supported()` for BFloat16 support
- Mock `torch.backends.mps.is_available()` for MPS availability

The console output test uses pytest's `capsys` fixture to capture and verify printed warning messages.

## Related Code

- **Implementation**: `train.py` lines 355-380 (dtype fallback logic)
- **Original Issue**: "Got unsupported ScalarType BFloat16" error
- **Solution**: Automatic dtype detection and fallback with user notification 