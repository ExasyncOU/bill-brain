# Bill - B-Intelligent

**A spatial neural network with 10,000 neurons and 3D brain visualization.**

Built by [Exasync](https://exasync.ai) - AI company based in Estonia, EU.

## Live Demo

**[https://exasyncou.github.io/bill-brain/](https://exasyncou.github.io/bill-brain/)**

## What makes Bill unique?

| Feature | Bill | Standard Transformer |
|---------|------|---------------------|
| Neuron positions | Learnable 3D coordinates | None (just layers) |
| Learning | Gradient + Hebbian | Gradient only |
| Inference | Cascade (ring by ring) | Layer by layer |
| Early exit | Yes (confidence-based) | No (always all layers) |
| Complexity | O(n*k) sparse | O(n^2) attention |
| Visualization | Real-time 3D | Not possible |
| Explainability | High (spatial clusters) | Low (black box) |

## Architecture

- **10,000 neurons** with learnable 3D positions
- **Sparse attention**: only k=32 nearest neighbors per neuron
- **Ring-based cascade**: activation propagates outward from center
- **Hebbian learning**: neurons that fire together move together in 3D space
- **Early exit**: stops computing when confidence is high enough
- **Brain-area anchoring**: prevents gravitational collapse

## Tech Stack

- **Backend**: Python 3.11 + PyTorch 2.6 + FastAPI
- **Frontend**: Three.js + WebGL + Bloom post-processing
- **3D Model**: Brain mesh with 50K vertices
- **Training**: CUDA (RTX 4080 Super, 16GB VRAM)

## Milestones

- M1: Neural Foundation (MNIST 97.3%)
- M2: Language Understanding (BPC 2.515)
- M3: Scale & Coherence (512 neurons, full Shakespeare)
- M4: Live Neural Visualization (WebSocket streaming)
- M5: BPE Tokenizer + Checkpoints + Spatial Recurrence
- M6: Chat System (multi-turn dialogue)
- M7: REST API + Dashboard + Benchmarks
- M8: Hippocampus (RAG Memory System)
- M9: Knowledge Distillation (learning from Claude)
- M10: Efficiency Benchmark (Bill vs MLP)

## License

Copyright 2026 Exasync OU. All rights reserved.
