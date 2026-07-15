"""CPU-side emergent physics layer.

Stage 6 MVP:
- Field: energy grid that diffuses and receives audio injections.
- Entities: simple organisms with DNA genes.
- Forces: translate world + audio into entity motion.
- Ecosystem: runs birth/growth/competition/extinction (MVP rules).

The renderer consumes the resulting WorldState summaries and exposes them to GLSL.
"""

