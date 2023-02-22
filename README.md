# Compact Effective Basis Generation

Example repository for [ArXiv:2302.10660](https://arxiv.org/abs/2302.10660)

- [h4.py](h4.py): fast example of ring and linear H4 from paper  
- [h4-slow.py](h4-slow.py): slow example (does not need extra structures)  
- [utils.py](utils.py): providing extra structures for fast example  

## In a nutshell

- small quantum circuits are generated via graph-based construction (see [ArXiv:2207.12421](https://arxiv.org/abs/2207.12421))  
- the total wavefunction is prepared a linear combination of the graph circuits  
- parameters are optimized via a global optimization procedure  

