# Sources and attribution

- Research problem: https://github.com/triton-lang/triton/issues/8908, reported by GitHub user oulgen.
- Original diagnostic reproducer: https://gist.github.com/oulgen/c0083856049f6f822926020739ccce6b . Its source is excluded from this repository; the original gist's license has not been verified.
- The implementation was developed with AI assistance during interactive remote A800 experiments. The owner should review and understand the code before publishing or presenting it.
- PyTorch native LayerNorm backward served as the primary runtime/reference comparator; Triton provides JIT compilation and GPU programming APIs. These dependencies are not vendored.

The rewritten algorithms implement standard LayerNorm gradient formulas. Excluding the original gist does not itself establish legal independence or confer a license. Before public release, review similarities/provenance and select an appropriate license. No ownership, upstream acceptance, or compiler-fix claim is made here.

GitHub issue metadata was most recently read as Open during preparation. The three retrieved comments discuss the large block size and ongoing investigation. No claim is made that the problem is globally unsolved or that nobody else has implemented a solution.
