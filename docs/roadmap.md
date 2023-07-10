# Roadmap

- [x] call PyTorch functions on JAX data without input data copy
- [x] call PyTorch functions on JAX data without input data copy under jit
- [x] support both GPU and CPU
- [x] (feature) support partial CPU building on systems without CUDA
- [x] (user-friendly) support functions with a single output (return a single output, not a tuple)
- [x] (user-friendly) support arbitrary argument input and output structure (use pytrees on the 
      Python side)
- [x] (feature) support batching (e.g., support for `jax.vmap`)
- [x] (feature) support integer input/output types
- [x] (feature) support mixed-precision arguments in inputs/outputs
- [x] (feature) support defining VJP for the wrapped function (import the experimental functionality 
      from [jit-JAXFriendlyInterface](https://github.com/rdyro/jfi-JAXFriendlyInterface))
- [ ] (tests) test how well device mapping works on multiple GPUs
- [ ] (tests) setup automatic tests for multiple versions of Python, PyTorch and JAX
- [ ] (feature) look into supporting in-place functions (support for output without copy)
- [ ] (feature) support TPU
