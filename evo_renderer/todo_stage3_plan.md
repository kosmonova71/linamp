Stage 3 (Genetic Shader Modules) - implemented

- modules.py
  - add ShaderModule + MODULES primitive snippets: noise/warp/fractal/particles

- genome.py
  - replace Stage 1/2 parameter-only genome with Stage 3 structural genome
  - fields: modules (module DNA), params (warp_strength, feedback, colour genes, mutation_rate)
  - mutate(): add/remove/replace/tune actions

- shader_factory.py
  - assemble fragment shader by concatenating module code snippets
  - keep EvoRenderer uniform + feedback loop contract
  - inject warp_strength/feedback/colour genes as literals

- memory.py
  - add EvolutionMemory (champions memory)

Note: Fitness scoring + saving into EvolutionMemory is not yet wired into evo_renderer/EvolutionEngine because the current evo_renderer/evolution.py does not measure fitness. Next step would add scoring and memory.save(genome, score).

