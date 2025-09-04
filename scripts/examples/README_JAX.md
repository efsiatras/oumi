# JAX Examples and Demos

This directory contains user-runnable example scripts for the JAX integration.

## Available Scripts

### 🧪 Comprehensive Model Testing (RECOMMENDED)
```bash
python comprehensive_jax_test.py
```
**Complete test suite for all JAX models**. Tests:
- All 6 JAX models (Llama3, Llama4, DeepSeek R1, GPT OSS, Kimi K2, Qwen3)
- CPU/GPU inference pipelines
- Performance benchmarking
- Memory usage analysis
- JAX 0.7.0+ compatibility validation

### ⚡ Performance Benchmarking
```bash
python jax_performance_benchmark.py
```
**Detailed performance analysis**. Measures:
- Inference latency (prefill vs decode)
- Throughput (tokens per second)
- Scaling characteristics with model size
- Attention mechanism comparisons
- Memory efficiency patterns

### 🎮 Interactive Model Testing
```bash
python interactive_jax_test.py
```
**User-friendly interactive testing**. Features:
- Choose specific models to test
- Configure model sizes (nano/tiny/small/medium)
- Real-time inference results
- Step-by-step guidance

Example usage:
```bash
python interactive_jax_test.py --model llama3 --size small
```

### 🤖 Interactive Demo
```bash
python chat_with_jax.py
```
Interactive chat interface with a tiny JAX model. Demonstrates:
- Character-level tokenization
- Real-time inference responses
- Autoregressive text generation

### 🧪 Local Model Testing (Legacy)
```bash
python jax_local_demo.py
```
Tests JAX models locally on CPU. Shows:
- Llama3 and DeepSeek R1 model initialization
- Actual inference with token generation
- Model architecture verification

### 🔧 System Integration Check
```bash
python jax_system_check.py
```
Verifies JAX integration is working correctly. Tests:
- JAX dependencies and imports
- Oumi builder system integration
- Configuration file validation

### 🚀 Engine Integration Demo
```bash
python jax_engine_demo.py
```
Demonstrates JAX engine integration with Oumi. Shows:
- JAX engine registration
- Conversation handling
- Mock inference pipeline

### ✅ Official Test Runner
```bash
python run_jax_tests.py
```
**Runs the official pytest test suite**. Executes:
- All 67 JAX integration tests
- JAX inference engine unit tests
- Test result analysis and reporting
- Environment validation

## Prerequisites

Install JAX dependencies:
```bash
pip install "oumi[jax]"
```

## Quick Start

1. **First time setup**: Verify installation
   ```bash
   python run_jax_tests.py           # Run official test suite
   python jax_system_check.py        # Check system integration
   ```

2. **Test all models**: Run comprehensive test
   ```bash
   python comprehensive_jax_test.py  # Test all 6 JAX models
   ```

3. **Performance analysis**: Run benchmarks
   ```bash
   python jax_performance_benchmark.py  # Detailed performance analysis
   ```

4. **Interactive exploration**: Test specific models
   ```bash
   python interactive_jax_test.py --model llama3 --size small
   ```

## Test Results Summary

✅ **All JAX integration tests pass** (67 passed, 12 skipped, 0 failed)

Supported models:
- **Llama3**: Standard transformer with efficient attention
- **Llama4**: MoE variants (Scout/Maverick) with expert routing
- **DeepSeek R1**: MLA (Multi-Head Latent Attention) with MoE
- **GPT OSS**: Sliding attention with configurable windows
- **Kimi K2**: Long context support with LoRA attention
- **Qwen3**: RoPE attention with MoE capabilities

## Notes

- These are **demo scripts** for users to run manually
- Use small models for CPU testing
- **Proper pytest tests** are in `tests/` directory
- For production usage, see the main JAX documentation
- All scripts support JAX 0.7.0+ and Python 3.11+
