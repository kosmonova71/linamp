#!/usr/bin/env python3

"""evo_visualizer.py

Minimum living renderer scaffold.

Milestone 1:
  - Single organism that can mutate live.
  - Visual genome parameters -> generated GLSL fragment shader.
  - Render via OpenGL feedback ping-pong FBO.

This file intentionally avoids MilkDrop preset loading, transitions, VM,
expression parsing, and fallback visuals.

GTK dependency: Gtk.GLArea.

Features:
  - Real-time genome evolution with adaptive mutation strength
  - Audio-reactive rendering with feedback effects
  - Human feedback steering (like/dislike)
  - Preset/checkpoint system
  - Keyboard controls for interaction
  - Comprehensive logging and error handling
"""

from __future__ import annotations

import gi
import array
import copy
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple, Union, List, Optional, Any

import OpenGL.GL as GL

from evo.modular_genome import ModularGenome
from evo.modular_shader_factory import ModularShaderFactory
from evo.genome_base import GenomeBase, MUTATION_WEIGHTS, ShaderCache
from evo_renderer.evolution_memory import EvolutionMemory, MusicSignature, SpeciesRecord

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

# ============================================================================
# Constants
# ============================================================================

TWO_PI = 6.28318
GLSL_FLOAT_PRECISION = 6

# Mutation history is kept bounded so genomes do not grow without limit across
# generations (every clone deep-copies the list). Unbounded growth was one of
# the Python-side memory creep causes in the OOM post-mortem.
MAX_MUTATION_HISTORY = 64

# Geometry constants
FULLSCREEN_QUAD_VERTICES = [
    -1.0, -1.0, 0.0, 0.0,
     1.0, -1.0, 1.0, 0.0,
     1.0,  1.0, 1.0, 1.0,
    -1.0, -1.0, 0.0, 0.0,
     1.0,  1.0, 1.0, 1.0,
    -1.0,  1.0, 0.0, 1.0,
]
QUAD_VERTEX_COUNT = 6
QUAD_STRIDE_BYTES = 16  # 4 floats * 4 bytes

# Audio constants
AUDIO_KEYS = ("bass", "mid", "treb", "beat")
AUDIO_CLAMP_MAX = 1.5
AUDIO_CLAMP_MIN = 0.0

# Palette function constants
PALETTE_A = (0.15, 0.20, 0.35)
PALETTE_B = (0.65, 0.25, 0.75)
PALETTE_C = (1.00, 0.95, 0.85)
PALETTE_D = (0.20, 0.40, 0.60)

# Noise/FBM constants
NOISE_FREQ_X = 123.34
NOISE_FREQ_Y = 456.21
NOISE_OFFSET = 45.32
FBM_ITERATIONS = 4
FBM_SCALE_X = 1.6
FBM_SCALE_Y = 1.2
FBM_SCALE_NEG_Y = -1.2

# ============================================================================


@dataclass
class EvoConfig:
    """Configuration for the evolutionary renderer.
    
    Attributes:
        feedback_alpha: Blending factor for feedback buffer (0.0-1.0)
        feedback_res: Resolution of feedback buffer textures (square)
        mutation_interval_secs: Time between mutations in seconds
        mutation_strength: Initial mutation step size
        mutation_strength_min: Minimum mutation strength bound
        mutation_strength_max: Maximum mutation strength bound
        debug: Enable debug logging and visualization
        score_weights: Weighted scoring for genome parameters
        enable_logging: Enable detailed logging output
        preset_dir: Directory for saving/loading presets
        brightness_base: Base brightness multiplier
        brightness_audio_scale: Audio influence on brightness
        population_size: Number of organisms in the population
        fitness_image_res: Resolution for fitness readback (square)
        enable_visual_fitness: Use image-based fitness instead of parameter score
        enable_lineage: Enable memory DNA lineage tracking
        enable_feedback_operators: Enable advanced feedback modes
        crossover_rate: Probability of crossover vs mutation in population
    """
    feedback_alpha: float = 0.85
    feedback_res: int = 512
    mutation_interval_secs: float = 10.0
    mutation_strength: float = 0.1
    mutation_strength_min: float = 0.01
    mutation_strength_max: float = 0.5
    debug: bool = False
    score_weights: Dict[str, float] = field(default_factory=dict)
    enable_logging: bool = True
    preset_dir: Optional[Path] = None
    brightness_base: float = 0.85
    brightness_audio_scale: float = 0.35
    population_size: int = 4
    fitness_image_res: int = 64
    enable_visual_fitness: bool = True
    enable_lineage: bool = True
    enable_feedback_operators: bool = True
    crossover_rate: float = 0.3
    enable_ecosystem_memory: bool = False
    enable_dream_mode: bool = False
    dream_threshold: float = 0.15
    enable_world_evolution: bool = False
    world_evolution_interval_secs: float = 60.0
    music_match_threshold: float = 0.75
    memory_db_path: Optional[str] = None
    memory_watchdog_mb: float = 4096.0

    # ------------------------------------------------------------------
    # Spatial Ecosystem (specs 46-60)
    # ------------------------------------------------------------------
    enable_spatial_ecosystem: bool = False
    spatial_population_size: int = 18
    spatial_world_w: int = 64          # resource field grid resolution
    spatial_world_h: int = 64
    ecosystem_step_hz: float = 30.0
    enable_senses: bool = True         # 53 evolution of senses
    enable_organs: bool = True         # 54 internal organs
    enable_brain: bool = True          # 55 brain network
    enable_signals: bool = True        # 50 communication
    enable_flocking: bool = True       # 51 collective behaviour
    enable_predator_prey: bool = True  # 49 food chains
    enable_climate: bool = True        # 52 seasonal music climate
    enable_complexity: bool = True     # 57 evolution of complexity
    enable_migration: bool = True     # 58 migration
    enable_extinction: bool = True     # 59 extinction & recovery
    enable_energy_economy: bool = True # 56 energy economy
    show_territories: bool = False     # debug overlay

    def __post_init__(self):
        """Initialize default score weights if not provided."""
        if not self.score_weights:
            self.score_weights = {
                "warp": 0.45,
                "noise_scale": 0.20,
                "feedback": 0.20,
                "symmetry": 0.15,
            }
        
        # Ensure preset_dir exists
        if self.preset_dir:
            self.preset_dir = Path(self.preset_dir)
            self.preset_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate extended config
        if self.population_size < 1:
            self.population_size = 1
        if self.crossover_rate < 0.0:
            self.crossover_rate = 0.0
        elif self.crossover_rate > 1.0:
            self.crossover_rate = 1.0
        self.dream_threshold = max(0.0, min(1.0, float(self.dream_threshold)))
        self.world_evolution_interval_secs = max(5.0, float(self.world_evolution_interval_secs))
        self.music_match_threshold = max(0.0, min(1.0, float(self.music_match_threshold)))

        # Validate spatial ecosystem config
        self.spatial_population_size = max(2, int(self.spatial_population_size))
        self.spatial_world_w = max(8, int(self.spatial_world_w))
        self.spatial_world_h = max(8, int(self.spatial_world_h))
        self.ecosystem_step_hz = max(1.0, float(self.ecosystem_step_hz))


class VisualGenome(GenomeBase):
    """Represents a visual organism's genetic parameters.
    
    Tracks mutation acceptance history to adaptively adjust mutation strength,
    implementing an implicit quality-seeking algorithm.
    """
    __slots__ = ("params", "_recent_accepts", "_window_size", "_strength", 
                 "lineage", "generation", "fitness", "parent_id", "mutations",
                 "age", "lifespan")

    PARAM_BOUNDS = {
        "warp": (0.0, 1.0),
        "feedback": (0.0, 1.0),
        "rotation": (-TWO_PI, TWO_PI),
        "noise_scale": (0.1, 20.0),
        "symmetry": (1, 12),
        "colour_shift": (-TWO_PI, TWO_PI),
        "field_type": (0, 8),
        "flow_speed": (0.0, 3.0),
        "curl_strength": (0.0, 2.0),
        "domain_warp": (0.0, 2.0),
        "fractal_layers": (1, 12),
        "turbulence": (0.0, 3.0),
        "vortex": (-5.0, 5.0),
        "mirror": (0, 1),
        "particle_density": (0.0, 5.0),
        "glow": (0.0, 2.0),
        "contrast": (0.0, 3.0),
        "colour_palette": (0.0, 20.0),
        "feedback_mode": (0, 6),
        # --- 3D / fake-raymarch genes (the "alive organism" physics) ---
        "dimension": (0, 3),
        "camera_speed": (0.0, 5.0),
        "camera_orbit": (-6.28, 6.28),
        "camera_depth": (1.0, 10.0),
        "camera_roll": (-6.28, 6.28),
        "depth_scale": (0.0, 10.0),
        "ray_steps": (8, 64),
        "fog_density": (0.0, 5.0),
    }

    INT_KEYS = {"symmetry", "field_type", "fractal_layers", "mirror",
                "feedback_mode", "dimension", "ray_steps"}

    def __init__(self, strength: float = 0.1, lineage: Optional[str] = None, 
                 generation: int = 0, parent_id: Optional[str] = None):
        """Initialize a genome with default parameters.
        
        Args:
            strength: Initial mutation step size
            lineage: Lineage identifier for memory DNA
            generation: Generation number
            parent_id: Parent genome identifier
        """
        self.params: Dict[str, Union[float, int]] = {
            "warp": 0.5,
            "noise_scale": 3.0,
            "feedback": 0.85,
            "rotation": 0.0,
            "symmetry": 4,
            "colour_shift": 0.0,
            "field_type": 0,
            "flow_speed": 1.0,
            "curl_strength": 0.5,
            "domain_warp": 0.5,
            "fractal_layers": 4,
            "turbulence": 1.0,
            "vortex": 0.0,
            "mirror": 0,
            "particle_density": 1.0,
            "glow": 0.5,
            "contrast": 1.0,
            "colour_palette": 0.0,
            "feedback_mode": 0,
            "dimension": 0,
            "camera_speed": 1.0,
            "camera_orbit": 1.0,
            "camera_depth": 3.0,
            "camera_roll": 0.0,
            "depth_scale": 1.0,
            "ray_steps": 32,
            "fog_density": 1.0,
        }
        self._recent_accepts: List[int] = []
        self._window_size = 8
        self._strength = float(strength)
        self.lineage = lineage or self._generate_id()
        self.generation = generation
        self.fitness = 0.0
        self.parent_id = parent_id
        self.mutations: List[str] = []
        self.age = 0
        self.lifespan = 500

    def _generate_id(self) -> str:
        """Generate a unique genome identifier."""
        import uuid
        return str(uuid.uuid4())[:8]

    def mutate(self, strength: Optional[float] = None, large: bool = False) -> None:
        """Mutate a random parameter by the current strength.
        
        Args:
            strength: Optional override for mutation strength
            large: When True, apply a stronger / multi-step mutation used when
                an organism outlives its lifespan to force succession.
        """
        if strength is not None:
            self._strength = float(strength)
        if large:
            self._strength = min(self._strength * 2.0 + 0.1, 1.0)

        self._mutate_once()
        if large:
            self._mutate_once()

    def _mutate_once(self) -> None:
        """Single mutation pass using weighted gene selection."""
        keys = list(self.params.keys())
        weights = [MUTATION_WEIGHTS.get(k, 1) for k in keys]
        key = random.choices(keys, weights=weights)[0]
        old_val = self.params[key]
        delta = 0.0

        if key in self.INT_KEYS:
            lo, hi = self.PARAM_BOUNDS[key]
            step = random.choice([-1, 1])
            new_val = int(old_val) + step
            if new_val < lo:
                new_val = int(lo)
            elif new_val > hi:
                new_val = int(hi)
            self.params[key] = new_val
            delta = float(new_val - old_val)
        else:
            lo, hi = self.PARAM_BOUNDS[key]
            delta = random.uniform(-self._strength, self._strength)
            if key == "noise_scale":
                delta = delta * 2.0
            new_val = float(old_val) + delta
            new_val = max(lo, min(hi, new_val))
            self.params[key] = new_val

        mutation_desc = f"{key}{delta:+.3f}" if isinstance(old_val, (int, float)) else key
        self.mutations.append(mutation_desc)
        if len(self.mutations) > MAX_MUTATION_HISTORY:
            del self.mutations[:-MAX_MUTATION_HISTORY]
        self.clamp()

    def record_accept(self, accepted: bool) -> None:
        """Record whether a mutation was accepted.
        
        Args:
            accepted: Whether the mutation was accepted
        """
        self._recent_accepts.append(1 if accepted else 0)
        if len(self._recent_accepts) > self._window_size:
            self._recent_accepts.pop(0)

    def adapt_strength(self, lo: float, hi: float) -> None:
        """Adaptively adjust mutation strength based on acceptance rate.
        
        Decreases strength if acceptance rate is low (exploration needed),
        increases if high (good area found).
        
        Args:
            lo: Minimum mutation strength bound
            hi: Maximum mutation strength bound
        """
        if not self._recent_accepts:
            return
        rate = sum(self._recent_accepts) / len(self._recent_accepts)
        if rate < 0.2:
            self._strength = max(lo, self._strength * 0.7)
        elif rate > 0.6:
            self._strength = min(hi, self._strength * 1.3)

    def clamp(self) -> None:
        """Clamp all parameters to valid ranges."""
        for key, (lo, hi) in self.PARAM_BOUNDS.items():
            if key not in self.params:
                continue
            
            val = self.params[key]
            
            if isinstance(val, int):
                self.params[key] = int(max(lo, min(hi, val)))
            else:
                val_float = float(val)
                
                # Special handling for rotation (wrap into [-pi, +pi]).
                if key == "rotation":
                    self.params[key] = float(((val_float + math.pi) % TWO_PI) - math.pi)
                else:
                    self.params[key] = max(lo, min(hi, val_float))

    def crossover(self, other: "VisualGenome") -> "VisualGenome":
        """Create a child genome by crossover with another genome.
        
        Args:
            other: Parent genome to crossover with
            
        Returns:
            New child genome
        """
        child = VisualGenome(
            strength=(self._strength + other._strength) / 2.0,
            lineage=f"{self.lineage}x{other.lineage}",
            generation=max(self.generation, other.generation) + 1,
            parent_id=self.lineage,
        )
        for key in self.params:
            if random.random() < 0.5:
                child.params[key] = self.params[key]
            else:
                child.params[key] = other.params[key]
        child.clamp()
        return child

    def clone(self) -> "VisualGenome":
        """Create a deep copy of this genome.
        
        Returns:
            Copy of this genome
        """
        g = VisualGenome(
            strength=self._strength,
            lineage=self.lineage,
            generation=self.generation,
            parent_id=self.parent_id,
        )
        g.params = copy.deepcopy(self.params)
        g._recent_accepts = list(self._recent_accepts)
        g.fitness = self.fitness
        g.mutations = list(self.mutations)
        return g

    def shader_signature(self) -> Dict[str, Union[float, int]]:
        """Return only the genome fields that affect the generated shader.

        Used as the :class:`ShaderCache` key so an unchanged phenotype reuses
        its compiled GL program. Must NOT include lineage, generation, fitness
        or mutation history -- those churn every generation and would defeat
        the cache (forcing recompiles and eviction leaks).
        """
        return dict(self.params)

    def to_dict(self) -> Dict[str, Union[float, int]]:
        """Export genome as a dictionary."""
        data = copy.deepcopy(self.params)
        data["_lineage"] = self.lineage
        data["_generation"] = self.generation
        data["_fitness"] = self.fitness
        data["_parent_id"] = self.parent_id
        data["_mutations"] = list(self.mutations)
        return data
    
    def from_dict(self, data: Dict[str, Union[float, int]]) -> None:
        """Load genome from a dictionary."""
        for key in self.params:
            if key in data:
                self.params[key] = data[key]
        self.lineage = data.get("_lineage", self._generate_id())
        self.generation = int(data.get("_generation", 0))
        self.fitness = float(data.get("_fitness", 0.0))
        self.parent_id = data.get("_parent_id")
        self.mutations = list(data.get("_mutations", []))
        self.clamp()

    def distance(self, other: "VisualGenome") -> float:
        """Normalized 0..1 phenotypic distance to another genome.

        Each parameter is compared relative to its allowed range so that genes
        with wide bounds do not dominate the metric.
        """
        total = 0.0
        count = 0
        for k, (lo, hi) in self.PARAM_BOUNDS.items():
            if k not in other.params:
                continue
            rng = max(abs(hi - lo), 1e-6)
            total += abs(self.params[k] - other.params[k]) / rng
            count += 1
        return total / max(count, 1)

    def environmental_affinity(self, audio: Dict[str, float]) -> float:
        """Bonus for how well this genome suits the current audio.

        Bass-heavy music favours vortex motion; treble favours particle
        density. Mirrors the selection pressure in the modular genome.
        """
        bass = float(audio.get("bass", 0.0))
        treb = float(audio.get("treb", 0.0))
        score = 0.0
        if bass > 0.7:
            score += float(self.params.get("vortex", 0.0)) * 0.2
        if treb > 0.7:
            score += float(self.params.get("particle_density", 0.0)) * 0.1
        return score


class ShaderFactory:
    """Generates GLSL shaders from genome parameters."""

    @staticmethod
    def _glsl_float(value: float, lo: Optional[float] = None, hi: Optional[float] = None) -> str:
        """Convert a float to a valid GLSL literal.
        
        Guard against non-finite or out-of-range values reaching the shader
        source, which would produce an invalid GLSL literal and fail to compile.
        GLSL float literals must contain a '.' or exponent.
        
        Args:
            value: Float value to convert
            lo: Optional lower bound
            hi: Optional upper bound
            
        Returns:
            Valid GLSL float literal string
        """
        v = float(value)
        if not math.isfinite(v):
            v = 0.0
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        s = repr(v)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s

    def _build_field_functions(self, p: Dict) -> str:
        """Build field function GLSL code based on field_type."""
        fractal_layers = int(p.get("fractal_layers", 4))
        curl_strength = self._glsl_float(p.get("curl_strength", 0.5), lo=0.0, hi=2.0)
        
        return f"""
float hash21(vec2 p) {{
    p = fract(p*vec2({NOISE_FREQ_X},{NOISE_FREQ_Y}));
    p += dot(p, p+{NOISE_OFFSET});
    return fract(p.x*p.y);
}}

float noise(vec2 p) {{
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash21(i);
    float b = hash21(i+vec2(1.0,0.0));
    float c = hash21(i+vec2(0.0,1.0));
    float d = hash21(i+vec2(1.0,1.0));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(a,b,u.x) + (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
}}

float fbm(vec2 p) {{
    float v = 0.0;
    float a = 0.5;
    mat2 m = mat2({FBM_SCALE_X},{FBM_SCALE_Y},{FBM_SCALE_NEG_Y},{FBM_SCALE_X});
    for(int i=0;i<{fractal_layers};i++){{
        v += a*noise(p);
        p = m*p;
        a *= 0.5;
    }}
    return v;
}}

float turbulence(vec2 p) {{
    float v = 0.0;
    float a = 0.5;
    mat2 m = mat2({FBM_SCALE_X},{FBM_SCALE_Y},{FBM_SCALE_NEG_Y},{FBM_SCALE_X});
    for(int i=0;i<{fractal_layers};i++){{
        v += a*abs(noise(p)*2.0-1.0);
        p = m*p;
        a *= 0.5;
    }}
    return v;
}}

vec2 curl(vec2 p) {{
    float e = 0.01;
    float n1 = fbm(p+vec2(0.0,e));
    float n2 = fbm(p-vec2(0.0,e));
    float n3 = fbm(p+vec2(e,0.0));
    float n4 = fbm(p-vec2(e,0.0));
    return vec2(n1-n2, n4-n3) * {curl_strength};
}}

float field_fbm(vec2 p) {{
    vec2 wp = p + vec2(fbm(p + k_domain_warp), fbm(p + k_domain_warp + 3.14)) * k_domain_warp;
    return fbm(wp);
}}

float field_trig(vec2 p) {{
    return sin(p.x)*cos(p.y) + 0.5*sin(p.x*0.5+p.y*0.3);
}}

float field_radial(vec2 p) {{
    return length(p);
}}

float field_curl(vec2 p) {{
    vec2 c = curl(p);
    return fbm(p + c);
}}

float field_turb(vec2 p) {{
    return turbulence(p) + 0.5*fbm(p);
}}

float field_voronoi(vec2 p) {{
    vec2 i = floor(p);
    vec2 f = fract(p);
    float d = 1.0;
    for(int y=-1;y<=1;y++){{
        for(int x=-1;x<=1;x++){{
            vec2 neighbor = vec2(float(x),float(y));
            vec2 point = hash21(i+neighbor)*vec2(1.0);
            vec2 diff = neighbor + point - f;
            d = min(d, length(diff));
        }}
    }}
    return d;
}}

float field(vec2 p) {{
    if(k_field_type == 0) return field_fbm(p);
    if(k_field_type == 1) return field_trig(p);
    if(k_field_type == 2) return field_radial(p);
    if(k_field_type == 3) return field_curl(p);
    if(k_field_type == 4) return field_voronoi(p);
    if(k_field_type == 5) return fbm(p) + k_turbulence*0.5;
    if(k_field_type == 6) return field_turb(p);
    if(k_field_type == 7) return fbm(p + sin(p*3.0)*k_domain_warp*0.5);
    return fbm(p + curl(p)*k_curl_strength);
 }}

  float scene(vec3 p) {{
      float n = 0.0;
      n += noise(p.xy);
      n += noise(p.yz) * 0.5;
      n += noise(p.xz) * 0.25;
      return n;
  }}

 float hash31(vec3 p) {{
     return fract(sin(dot(p, vec3(12.3,45.6,78.9))) * 43758.5);
 }}

 mat3 rotateX(float a) {{
     float c = cos(a);
     float s = sin(a);
     return mat3(
         1.0, 0.0, 0.0,
         0.0, c,   -s,
         0.0, s,   c
     );
 }}

 mat3 rotateY(float a) {{
     float c = cos(a);
     float s = sin(a);
     return mat3(
         c,   0.0, s,
         0.0, 1.0, 0.0,
         -s,  0.0, c
     );
 }}

 vec3 feedback_transform3(vec3 p) {{
     float r = length(p);
     p.xy *= mat2(
         cos(r), -sin(r),
         sin(r),  cos(r)
     );
     return p;
 }}
 """

    def _build_feedback_function(self, p: Dict) -> str:
        """Build feedback UV transformation function."""
        return """
vec2 feedback_transform(vec2 uv, float t, vec2 q) {
    int mode = k_feedback_mode;
    vec2 result = uv;
    if(mode == 0) {
        result = uv;
    } else if(mode == 1) {
        float angle = atan(uv.y-0.5, uv.x-0.5) + 0.1*sin(t*0.3);
        float dist = length(uv-0.5);
        result = 0.5 + vec2(cos(angle),sin(angle))*dist;
    } else if(mode == 2) {
        float dist = length(uv-0.5);
        result = 0.5 + (uv-0.5)*(1.0 + 0.02*sin(t*0.5));
    } else if(mode == 3) {
        result = abs(fract(uv*2.0 + 0.5*sin(t*0.2)) - 0.5);
    } else if(mode == 4) {
        result = fract(uv + 0.02*sin(t*0.4));
    } else if(mode == 5) {
        result = uv * (1.0 - 0.005*sin(t*0.3));
    } else if(mode == 6) {
        // Trick 1: kaleidoscope feedback recursion - fold through k_symmetry
        // mirrors so fractal mirrors / crystal structures evolve mandala-like.
        vec2 c = uv - 0.5;
        float a2 = atan(c.y, c.x);
        float rad = length(c);
        float seg = 6.28318 / float(k_symmetry);
        a2 = mod(a2, seg);
        a2 = abs(a2 - seg*0.5);
        a2 += 0.02*sin(t*0.3);
        result = vec2(cos(a2), sin(a2))*rad + 0.5;
    }
    return result;
}
"""

    def _build_palette(self, palette_id: float, colour_shift: float) -> str:
        """Build colour palette function."""
        palettes = [
            (
                f"{PALETTE_A[0]},{PALETTE_A[1]},{PALETTE_A[2]}",
                f"{PALETTE_B[0]},{PALETTE_B[1]},{PALETTE_B[2]}",
                f"{PALETTE_C[0]},{PALETTE_C[1]},{PALETTE_C[2]}",
                f"{PALETTE_D[0]},{PALETTE_D[1]},{PALETTE_D[2]}",
            ),
            ("0.05,0.10,0.20", "0.90,0.40,0.20", "2.00,1.80,1.60", "0.00,0.33,0.67"),
            ("0.50,0.50,0.50", "0.50,0.50,0.50", "1.00,1.00,1.00", "0.00,0.33,0.67"),
            ("0.80,0.20,0.20", "0.20,0.60,0.80", "2.00,1.00,1.50", "0.00,0.10,0.20"),
            ("0.10,0.30,0.10", "0.70,0.90,0.30", "1.50,2.00,1.00", "0.00,0.50,0.00"),
            ("0.20,0.00,0.30", "0.80,0.60,0.90", "1.00,1.50,2.00", "0.30,0.00,0.50"),
        ]
        
        pid = int(palette_id) % len(palettes)
        a, b, c, d = palettes[pid]
        
        return f"""
vec3 palette(float t) {{
    vec3 pa = vec3({a});
    vec3 pb = vec3({b});
    vec3 pc = vec3({c});
    vec3 pd = vec3({d});
    return pa + pb*cos({TWO_PI}*(pc*t + pd) + vec3(0.0,1.0,2.0)*{self._glsl_float(colour_shift)});
}}
"""

    def generate(self, genome: VisualGenome) -> str:
        """Generate a GLSL fragment shader from genome parameters.
        
        Args:
            genome: VisualGenome instance with parameters
            
        Returns:
            Complete GLSL fragment shader source code
        """
        p = genome.params
        sym = int(p["symmetry"])
        field_fns = self._build_field_functions(p)
        fb_fns = self._build_feedback_function(p)
        pal = self._build_palette(p.get("colour_palette", 0.0), p.get("colour_shift", 0.0))
        
        flow_speed = self._glsl_float(p.get("flow_speed", 1.0), lo=0.0, hi=3.0)
        domain_warp = self._glsl_float(p.get("domain_warp", 0.5), lo=0.0, hi=2.0)
        glow = self._glsl_float(p.get("glow", 0.5), lo=0.0, hi=2.0)
        contrast = self._glsl_float(p.get("contrast", 1.0), lo=0.0, hi=3.0)
        vortex = self._glsl_float(p.get("vortex", 0.0), lo=-5.0, hi=5.0)
        particle_density = self._glsl_float(p.get("particle_density", 1.0), lo=0.0, hi=5.0)
        feedback_mode = int(p.get("feedback_mode", 0))
        mirror = int(p.get("mirror", 0))

        return f"""
#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform vec4 u_audio;

uniform sampler2D u_prev_frame;

uniform float u_feedback_alpha;
uniform float u_brightness;

const float k_warp = {self._glsl_float(p['warp'])};
const float k_noise_scale = {self._glsl_float(p['noise_scale'], lo=0.1)};
const float k_feedback = {self._glsl_float(p['feedback'])};
const float k_rotation = {self._glsl_float(p['rotation'])};
const int   k_symmetry = {sym};
const float k_colour_shift = {self._glsl_float(p['colour_shift'])};
const float k_flow_speed = {flow_speed};
const float k_glow = {glow};
const float k_contrast = {contrast};
const float k_vortex = {vortex};
const float k_particle_density = {particle_density};
const float k_domain_warp = {domain_warp};
const int   k_field_type = {int(p.get('field_type', 0))};
const int   k_feedback_mode = {feedback_mode};
const int   k_mirror = {mirror};
 const float k_curl_strength = {self._glsl_float(p.get('curl_strength', 0.5), lo=0.0, hi=2.0)};
 const float k_turbulence = {self._glsl_float(p.get('turbulence', 1.0), lo=0.0, hi=3.0)};
 const int   k_fractal_layers = {int(p.get('fractal_layers', 4))};

 // --- 3D / fake-raymarch genome (the "alive organism" physics) ---
 const float k_dimension = {self._glsl_float(p.get('dimension', 0.0), lo=0.0, hi=3.0)};
 const float k_camera_speed = {self._glsl_float(p.get('camera_speed', 1.0), lo=0.0, hi=5.0)};
 const float k_camera_orbit = {self._glsl_float(p.get('camera_orbit', 1.0), lo=-6.28, hi=6.28)};
 const float k_camera_depth = {self._glsl_float(p.get('camera_depth', 3.0), lo=1.0, hi=10.0)};
 const float k_camera_roll = {self._glsl_float(p.get('camera_roll', 0.0), lo=-6.28, hi=6.28)};
 const float k_depth_scale = {self._glsl_float(p.get('depth_scale', 1.0), lo=0.0, hi=10.0)};
 const int   k_ray_steps = {int(p.get('ray_steps', 32))};
 const float k_fog_density = {self._glsl_float(p.get('fog_density', 1.0), lo=0.0, hi=5.0)};

 {field_fns}

 {fb_fns}

 {pal}

 bool bad(vec3 c)
 {{
     return any(isnan(c)) || any(isinf(c));
 }}

 void main() {{
    vec2 uv = v_uv;
    uv = (uv - 0.5) * vec2(u_resolution.x / u_resolution.y, 1.0);

    float t = u_time * k_flow_speed;

    // Rotation
    float angle = k_rotation + t * (0.15 + u_audio.y*0.3);
    float cs = cos(angle);
    float sn = sin(angle);
    mat2 R = mat2(cs, -sn, sn, cs);
    vec2 p = R * uv;

    float r = length(p);
    float th = atan(p.y, p.x);

    // Vortex
    float vortex_angle = k_vortex * r;
    cs = cos(vortex_angle);
    sn = sin(vortex_angle);
    mat2 V = mat2(cs, -sn, sn, cs);
    p = V * p;

    // Symmetry
    float seg = {TWO_PI} / float(k_symmetry);
    th = abs(mod(th, seg) - seg*0.5);
    if(k_mirror == 1) {{
        th = abs(th);
    }}

    vec2 q = vec2(cos(th), sin(th)) * r;

    // --- Shader operators (discovered by evolution via existing genes) ---

    // Trick 2: polar domain distortion (gated by vortex gene) - liquid
    // galaxies / spiral organisms / vortex flowers.
    if(k_vortex != 0.0) {{
        float pr = length(q);
        float pth = atan(q.y, q.x);
        pth += sin(pr*8.0 - t*0.5) * 0.3 * abs(k_vortex);
        pth += sin(pth*6.0 + t*0.3) * 0.15 * abs(k_vortex);
        q = vec2(cos(pth), sin(pth)) * pr;
    }}

    // Trick 7: audio-driven topology mutation (gated by vortex gene) - the
    // music reshapes the organism's "body", not just its colour.
    if(k_vortex != 0.0) {{
        q *= 1.0 + u_audio.x * 0.3 * abs(k_vortex);
        q.x += u_audio.z * 0.1 * sin(q.y*20.0 + t);
        q.y += u_audio.w * 0.1 * cos(q.x*16.0 - t);
    }}

    // Warp
    float warp = (k_warp + 1.0) * (1.0 + u_audio.x * 0.9 + u_audio.w * 0.25);
    q *= warp;

    // Domain warp
    if(k_domain_warp > 0.01) {{
        q += vec2(fbm(q*0.5 + t*0.1), fbm(q*0.5 + t*0.1 + 5.0)) * k_domain_warp * 0.3;
    }}

    // --- Fake 3D: ray-like depth field (the "travelling through clouds" loop) ---
    // 1. Moving camera genome: the organism is not rotating, the viewer is
    //    travelling through it.
    vec3 camera = vec3(
        sin(u_time * k_camera_speed),
        cos(u_time * k_camera_speed),
        -k_camera_depth
    );
    camera.xy *= k_camera_orbit;          // orbital radius
    // 6. Audio-controlled flight: bass pushes the camera forward, treble steers.
    camera.z -= u_audio.x * 2.0;
    camera.xy += sin(u_time) * u_audio.z;

    // 2./7. 3D rotation: real spatial motion by rotating the sample point.
    mat3 rot = rotateY(u_time * 0.2) * rotateX(u_time * 0.13);

    // 4. 3D feedback warp applied per sample -> twisting tunnels / wormholes /
    //    galaxy spirals.
    vec3 ro = camera;
    vec3 rd = normalize(vec3(q * k_noise_scale, 1.5));

    float depth = 0.0;
    float total = 0.0;

    int steps = min(k_ray_steps, 32);
    for(int i=0;i<steps;i++) {{
        if(total > 4.0) break;

        vec3 pos = ro + rd * depth;

        // 3D rotation of the sample point -> the world tumbles around you.
        vec3 rp = rot * pos;

        // 4. 3D feedback warp (spiral / wormhole twist) on the volume.
        rp = feedback_transform3(rp);

        float d = scene(rp);
        total += exp(-d * 4.0 * k_fog_density);

        // 6. beat advances the march through the volume.
        depth += k_depth_scale * 0.05 + u_audio.w * 0.5;
    }}

    // Legacy 2D field (kept so non-3D genomes still render the organism).
    float n = field(q * k_noise_scale + vec2(t*0.18, -t*0.12));
    n += 0.25*sin(q.x*3.0 + t*0.9 + u_audio.z*2.0);

    // 8. Blend 2D organism with 3D volume by the dimension gene (0..3).
    float dim = clamp(k_dimension / 3.0, 0.0, 1.0);
    float n3 = total / float(k_ray_steps);
    n = mix(n, n3, dim);

    // Contrast
    n = (n - 0.5) * k_contrast + 0.5;
    n = clamp(n, 0.0, 1.0);

    // Colour
    float hueT = n + 0.10*t + 0.25*u_audio.x + 0.15*u_audio.y + 0.20*u_audio.z + 0.05*u_audio.w;
    vec3 col = palette(hueT);

    // Glow
    if(k_glow > 0.01) {{
        vec3 glow_col = palette(hueT + 0.5);
        col += glow_col * k_glow * 0.15 * (1.0 + u_audio.x);
    }}

    // Trick 3: reaction-diffusion illusion (gated by domain_warp gene) -
    // bacteria / coral / alien skin laid over the organism.
    if(k_domain_warp > 0.01) {{
        float ca2 = fbm(q*3.0 + t*0.05);
        float cb2 = fbm(q*6.0 - t*0.03);
        float cells = smoothstep(0.45, 0.55, ca2 - cb2);
        col = mix(col, vec3(cells), 0.35 * clamp(k_domain_warp, 0.0, 1.0));
    }}

    // Trick 5: cheap fake bloom (gated by glow gene) - energy fields.
    if(k_glow > 0.01) {{
        vec3 bloom = vec3(0.0);
        for(int i=1;i<5;i++){{
            float f = float(i)*0.002;
            bloom += texture(u_prev_frame, v_uv + vec2(f, 0.0)).rgb;
        }}
        col += bloom * 0.15 * k_glow;
    }}

    // 5. Particles with depth: now they fly through 3D space.
    if(k_particle_density > 0.01) {{
        vec3 particlePos = vec3(q * k_particle_density * 10.0, sin(u_time));
        float particle = hash31(floor(particlePos * 20.0));
        col += vec3(1.0) * smoothstep(0.97, 1.0, particle) * 0.5;
    }}

    // Feedback with operator
    vec2 fb_uv = feedback_transform(v_uv, t, q);

    // Trick 4: chromatic aberration (shift scaled by bass u_audio.x).
    float caShift = 0.003 * u_audio.x;
    vec3 prev;
    prev.r = texture(u_prev_frame, fb_uv + vec2(caShift, 0.0)).r;
    prev.g = texture(u_prev_frame, fb_uv).g;
    prev.b = texture(u_prev_frame, fb_uv - vec2(caShift, 0.0)).b;

    // Trick 6: nonlinear trails (gain + gamma ghost memory).
    prev *= (1.0 + k_feedback*0.3);
    prev = pow(prev, vec3(0.95));

    float a = clamp(u_feedback_alpha * k_feedback, 0.0, 0.999);
    vec3 outCol = mix(col, prev, a);

    // Trick 9: neon edge extraction (gated by curl_strength gene) -
    // glowing veins / electric networks / biological structures.
    if(k_curl_strength > 0.01) {{
        float e = 0.0035;
        vec3 ea = texture(u_prev_frame, fb_uv + vec2(e, 0.0)).rgb;
        vec3 eb = texture(u_prev_frame, fb_uv - vec2(e, 0.0)).rgb;
        float edge = length(ea - eb);
        outCol += vec3(edge) * 2.0 * clamp(k_curl_strength, 0.0, 2.0);
    }}

    outCol *= u_brightness;

    if(bad(outCol)){{
        outCol = vec3(0.0);
    }}

    outCol = clamp(outCol, 0.0, 1.0);
    fragColor = vec4(outCol, 1.0);
}}
"""


class FeedbackBuffer:
    """Ping-pong framebuffer for feedback effects.
    
    Maintains two textures and framebuffers for iterative feedback rendering.
    Resolution matches the viewport (not a fixed square) so the feedback
    effect stays sharp and the fitness readback is consistent with what is
    displayed.
    """
    
    def __init__(self, width: int, height: Optional[int] = None):
        """Initialize feedback buffer.
        
        Args:
            width: Texture width in pixels.
            height: Texture height in pixels. Defaults to ``width`` (square).
        """
        if height is None:
            height = width
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.res = self.width  # legacy square alias
        self.fbo: List[Optional[int]] = [None, None]
        self.tex: List[Optional[int]] = [None, None]
        self.ping = 0

    def _create_textures(self) -> None:
        """(Re)create the two ping-pong textures/framebuffers at current size."""
        for i in (0, 1):
            if self.tex[i] is None:
                self.fbo[i] = GL.glGenFramebuffers(1)
                self.tex[i] = GL.glGenTextures(1)

            GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex[i])
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA16F,
                self.width,
                self.height,
                0,
                GL.GL_RGBA,
                GL.GL_HALF_FLOAT,
                None,
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo[i])
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0,
                GL.GL_TEXTURE_2D,
                self.tex[i],
                0,
            )

            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"FeedbackBuffer FBO {i} incomplete: {hex(status)}")

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # Clear both buffers
        for i in (0, 1):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo[i])
            GL.glViewport(0, 0, self.width, self.height)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self.ping = 0

    def init_gl(self) -> None:
        """Initialize OpenGL resources.
        
        Raises:
            RuntimeError: If framebuffer is incomplete
        """
        try:
            self._create_textures()
            logger.info(f"FeedbackBuffer initialized: {self.width}x{self.height}")
        except Exception as e:
            logger.error(f"FeedbackBuffer initialization failed: {e}")
            raise

    def resize(self, width: int, height: Optional[int] = None) -> None:
        """Resize the feedback buffer to match the viewport.

        No-op if the size is unchanged. Deletes and recreates the textures so
        the feedback effect always matches the current GLArea dimensions.
        """
        if height is None:
            height = width
        width = max(1, int(width))
        height = max(1, int(height))
        if width == self.width and height == self.height and self.tex[0] is not None:
            return
        self.cleanup()
        self.width = width
        self.height = height
        self.res = self.width
        self._create_textures()

    def src_tex(self) -> int:
        """Get source texture ID for current frame."""
        return int(self.tex[1 - self.ping] or 0)

    def dst_fbo(self) -> int:
        """Get destination framebuffer ID for current frame."""
        return int(self.fbo[self.ping] or 0)

    def swap(self) -> None:
        """Swap ping-pong buffers for next frame."""
        self.ping = 1 - self.ping

    def cleanup(self) -> None:
        """Release OpenGL resources."""
        try:
            for i in (0, 1):
                if self.tex[i]:
                    GL.glDeleteTextures(1, [self.tex[i]])
                    self.tex[i] = None
                if self.fbo[i]:
                    GL.glDeleteFramebuffers(1, [self.fbo[i]])
                    self.fbo[i] = None
            logger.debug("FeedbackBuffer cleanup complete")
        except Exception as e:
            logger.error(f"FeedbackBuffer cleanup error: {e}")


class AudioState:
    """Tracks audio levels for reactive rendering.
    
    Maintains bass, mid, treble, and beat frequencies in normalized range.
    """
    __slots__ = ("audio",)

    def __init__(self):
        """Initialize audio state with zero levels."""
        self.audio: Dict[str, float] = {
            "bass": 0.0,
            "mid": 0.0,
            "treb": 0.0,
            "beat": 0.0,
        }

    def update(self, audio_levels: Dict[str, float]) -> None:
        """Update audio levels with clamping.
        
        Args:
            audio_levels: Dictionary with 'bass', 'mid', 'treb', 'beat' keys
        """
        for key in AUDIO_KEYS:
            val = float(audio_levels.get(key, 0.0))
            self.audio[key] = max(AUDIO_CLAMP_MIN, min(AUDIO_CLAMP_MAX, val))

    def uvec(self) -> Tuple[float, float, float, float]:
        """Get audio levels as tuple for shader uniforms.
        
        Returns:
            (bass, mid, treble, beat)
        """
        return (
            self.audio["bass"],
            self.audio["mid"],
            self.audio["treb"],
            self.audio["beat"],
        )


class EvolutionTimer:
    """Timer for triggering genome mutations at regular intervals."""
    __slots__ = ("interval_secs", "_next")

    def __init__(self, interval_secs: float):
        """Initialize timer.
        
        Args:
            interval_secs: Seconds between mutation triggers
        """
        self.interval_secs = float(interval_secs)
        self._next = time.time() + self.interval_secs

    def due(self) -> bool:
        """Check if mutation is due.
        
        Returns:
            True if elapsed time exceeds interval
        """
        return time.time() >= self._next

    def reset(self) -> None:
        """Reset timer for next interval."""
        self._next = time.time() + self.interval_secs


class FitnessEvaluator:
    """Evaluates genome fitness using visual image analysis."""
    
    def __init__(self, res: int = 64):
        self.res = res
        self._previous_image: Optional[bytes] = None
    
    def evaluate(self, genome: Any, image_data: bytes, 
                 previous_data: Optional[bytes] = None) -> float:
        """Evaluate fitness based on image analysis.
        
        Args:
            genome: Genome to evaluate
            image_data: Raw RGBA image bytes from readPixels
            previous_data: Previous frame image bytes for motion detection
            
        Returns:
            Fitness score (higher = better)
        """
        if np is None or len(image_data) < self.res * self.res * 4:
            return 0.5
        
        fitness = 0.0
        
        try:
            arr = np.frombuffer(image_data, dtype=np.uint8)
            arr = arr.reshape(self.res, self.res, 4)
            rgb = arr[:, :, :3].astype(np.float32) / 255.0
            gray = rgb.mean(axis=2)
            
            # Entropy (complexity)
            hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0))
            hist = hist / max(hist.sum(), 1)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            fitness += min(entropy / 4.0, 1.0) * 0.3
            
            # Edge amount (structural interest)
            dx = np.abs(gray[:, 2:] - gray[:, :-2])
            dy = np.abs(gray[2:, :] - gray[:-2, :])
            edges = (dx.mean() + dy.mean()) / 2.0
            fitness += min(edges * 10.0, 1.0) * 0.25
            
            # Colour variance
            colour_var = rgb.var(axis=(0, 1)).mean()
            fitness += min(colour_var * 20.0, 1.0) * 0.25
            
            # Motion (if previous frame available)
            if previous_data is not None and len(previous_data) >= self.res * self.res * 4:
                prev_arr = np.frombuffer(previous_data, dtype=np.uint8)
                prev_arr = prev_arr.reshape(self.res, self.res, 4)
                prev_gray = prev_arr[:, :, :3].astype(np.float32).mean(axis=2)
                motion = np.abs(gray - prev_gray).mean()
                fitness += min(motion * 15.0, 1.0) * 0.2
            
            # Symmetry bonus (only meaningful for genomes exposing a numeric
            # symmetry parameter; modular genomes fold symmetry in-shader).
            params = getattr(genome, "params", None)
            sym = int(params.get("symmetry", 0)) if isinstance(params, dict) else 0
            if sym > 1:
                half = gray[:, :self.res // 2]
                mirror_half = np.fliplr(half)
                symmetry_score = 1.0 - np.abs(half - mirror_half).mean()
                fitness += symmetry_score * 0.1

            # Penalize excessive brightness to prevent evolution of white flashing noise.
            brightness = float(rgb.mean())
            fitness -= max(brightness - 0.8, 0.0) * 0.3
            
        except Exception as e:
            logger.debug(f"Fitness evaluation failed: {e}")
            return 0.0
        
        return min(fitness, 1.0)
    
    def score_from_genome(self, genome: Any, weights: Optional[Dict[str, float]] = None) -> float:
        """Fallback score when image fitness is disabled / unavailable."""
        if weights is None:
            weights = {
                "warp": 0.45,
                "noise_scale": 0.20,
                "feedback": 0.20,
                "symmetry": 0.15,
            }
        # Legacy VisualGenome
        if hasattr(genome, "params") and isinstance(getattr(genome, "params"), dict):
            p = genome.params
            score = 0.0
            score += p.get("warp", 0.0) * weights.get("warp", 0.8)
            score += p.get("noise_scale", 0.0) * weights.get("noise_scale", 0.033)
            score += p.get("feedback", 0.0) * weights.get("feedback", 0.1)
            score += p.get("symmetry", 0) * weights.get("symmetry", 0.05)
            score += p.get("field_type", 0) * weights.get("field_type", 0.02)
            score += p.get("glow", 0.0) * weights.get("glow", 0.1)
            score += p.get("contrast", 1.0) * weights.get("contrast", 0.1)
            score += p.get("turbulence", 0.0) * weights.get("turbulence", 0.05)
            return float(score)

        # ModularGenome heuristic
        score = 0.0
        if hasattr(genome, "field"):
            # encourage more complex field modules
            field = str(getattr(genome, "field"))
            score += {"fbm": 0.35, "voronoi": 0.25, "reaction_diffusion": 0.45, "mandelbrot": 0.5, "sdf": 0.3}.get(field, 0.25)
        if hasattr(genome, "warp"):
            warp = str(getattr(genome, "warp"))
            score += {"curl": 0.2, "vortex": 0.15, "twist": 0.1, "none": 0.05}.get(warp, 0.1)
        if hasattr(genome, "palette"):
            score += 0.15
        if hasattr(genome, "post") and isinstance(getattr(genome, "post"), list):
            post = set(getattr(genome, "post"))
            score += 0.05 * len(post)
            score += 0.08 if "glow" in post else 0.0
            score += 0.08 if "bloom" in post else 0.0
            score += 0.06 if "chromatic_aberration" in post else 0.0
            # Reward the new shader operators so evolution discovers combinations
            # (kaleidoscope is rewarded via feedback_mode, not post).
            score += 0.05 if "polar_galaxy" in post else 0.0
            score += 0.05 if "cells" in post else 0.0
            score += 0.05 if "nonlinear_trails" in post else 0.0
            score += 0.05 if "audio_topology" in post else 0.0
            score += 0.05 if "depth_cloud" in post else 0.0
            score += 0.05 if "neon_edge" in post else 0.0
            fb_mode = str(getattr(genome, "feedback_mode", "spiral"))
            score += 0.05 if fb_mode in ("kaleido", "mirror", "zoom") else 0.0
        return float(max(0.0, min(1.0, score)))


class Population:
    """Manages a population of evolving genomes.
    
    Genome-type agnostic: it only relies on the :class:`GenomeBase` contract
    (``fitness``, ``lineage``, ``mutate``, ``clone``, ``crossover``), so the
    same population works for both ``VisualGenome`` and ``ModularGenome``.
    """
    
    def __init__(self, size: int = 4, crossover_rate: float = 0.3, genome_factory=None):
        self.size = max(1, int(size))
        self.crossover_rate = float(crossover_rate)
        self.organisms: List[Any] = []
        self._generation = 0
        self._best_fitness = 0.0
        self._lineage: Dict[str, int] = {}
        self._genome_factory = genome_factory
        
        def _default_factory():
            return VisualGenome(strength=0.1, generation=0)

        if genome_factory is None:
            genome_factory = _default_factory
        
        for _ in range(self.size):
            g = genome_factory()
            self.organisms.append(g)
            self._record_lineage(g.lineage)

    
    def evolve(self, fitness_fn) -> None:
        """Evolve the population using fitness-based selection.
        
        Args:
            fitness_fn: Callable that takes a genome and returns a fitness score
        """
        # Age every organism one generation, forcing a large mutation once an
        # organism outlives its lifespan (natural succession).
        for g in self.organisms:
            if hasattr(g, "age_one_generation"):
                g.age_one_generation()
            else:
                g.age = getattr(g, "age", 0) + 1

        scores = [(fitness_fn(g), g) for g in self.organisms]
        scores.sort(key=lambda x: x[0], reverse=True)
        
        self._best_fitness = max(self._best_fitness, scores[0][0])
        
        # Keep top 50%, replace bottom 50% with mutations/crossover
        survivors = scores[:max(1, self.size // 2)]
        new_organisms: List[Any] = [g for _, g in survivors]
        
        while len(new_organisms) < self.size:
            if random.random() < self.crossover_rate and len(survivors) >= 2:
                parent_a = random.choice([g for _, g in survivors])
                parent_b = random.choice([g for _, g in survivors])
                child = parent_a.crossover(parent_b)
                child.mutate()
            else:
                parent = random.choice([g for _, g in survivors])
                child = parent.clone()
                child.mutate()
                if hasattr(child, "parent_id"):
                    child.parent_id = parent.lineage
            
            child.generation = self._generation + 1
            new_organisms.append(child)
            self._record_lineage(child.lineage)

        self.organisms = new_organisms[:self.size]
        self._generation += 1

    def _record_lineage(self, lineage: str) -> None:
        """Increment the lineage counter, keeping the dict bounded.

        Without a cap the set of unique lineage strings grows forever over long
        sessions; we keep the most recent entries only.
        """
        self._lineage[lineage] = self._lineage.get(lineage, 0) + 1
        if len(self._lineage) > 1024:
            # Drop the oldest recorded lineage (dicts preserve insertion order).
            self._lineage.pop(next(iter(self._lineage)))
    
    def best(self) -> Any:
        """Return the organism with highest recorded fitness."""
        return max(self.organisms, key=lambda g: g.fitness)
    
    def stats(self) -> Dict:
        """Return population statistics."""
        return {
            "size": len(self.organisms),
            "generation": self._generation,
            "best_fitness": self._best_fitness,
            "unique_lineages": len(self._lineage),
        }


class MemoryDNA:
    """Tracks lineage and mutation history for hereditary presets."""
    
    def __init__(self, preset_dir: Optional[Path] = None):
        self.preset_dir = preset_dir or Path.home() / ".evo_renderer" / "lineage"
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        self._lineage_file = self.preset_dir / "lineage.json"
        self._records: Dict[str, Dict] = {}
        self._last_save = 0.0
        self._save_interval = 30.0
        self._load()
    
    def _load(self) -> None:
        """Load lineage records from disk."""
        try:
            if self._lineage_file.exists():
                with open(self._lineage_file, "r") as f:
                    self._records = json.load(f)
        except Exception:
            self._records = {}
    
    def _save(self) -> None:
        """Save lineage records to disk."""
        try:
            with open(self._lineage_file, "w") as f:
                json.dump(self._records, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save lineage: {e}")
    
    def record(self, genome: Any) -> None:
        """Record a genome in the lineage database.
        
        Args:
            genome: Genome to record
        """
        record = {
            "parent": genome.parent_id,
            "generation": genome.generation,
            "fitness": genome.fitness,
            "mutations": list(genome.mutations[-10:]),
            "timestamp": time.time(),
            "params": genome.to_dict(),
        }
        self._records[genome.lineage] = record
        # Bound the in-memory lineage log so it cannot grow without limit over
        # long sessions (one entry per unique lineage). Disk is the durable copy.
        if len(self._records) > 2000:
            oldest = min(
                self._records,
                key=lambda k: self._records[k].get("timestamp", 0.0),
            )
            del self._records[oldest]
        now = time.time()
        if now - self._last_save >= self._save_interval:
            self._last_save = now
            self._save()
    
    def get_lineage(self, lineage_id: str) -> Optional[Dict]:
        """Get lineage history for a genome.
        
        Args:
            lineage_id: Lineage identifier
            
        Returns:
            Lineage record or None
        """
        return self._records.get(lineage_id)
    
    def export_preset(self, genome: Any, path: Optional[Path] = None) -> bool:
        """Export genome as hereditary preset.
        
        Args:
            genome: Genome to export
            path: Export path
            
        Returns:
            True if successful
        """
        try:
            if path is None:
                path = self.preset_dir / f"lineage_{genome.lineage}.json"
            
            preset = {
                "genome": genome.to_dict(),
                "parent": genome.parent_id,
                "generation": genome.generation,
                "fitness": genome.fitness,
                "mutations": list(genome.mutations),
                "timestamp": time.time(),
            }
            
            with open(path, "w") as f:
                json.dump(preset, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.warning(f"Failed to export preset: {e}")
            return False


# ============================================================================
# Spatial Ecosystem (specs 46 - 60)
# ----------------------------------------------------------------------------
# CPU supervision (spec 60):
#   - Music analysis        -> AudioState / music DNA
#   - Long-term memory      -> EvolutionMemory (existing) + ExtinctionArchive
#   - Species database      -> SpatialEcosystem.species
#   - Evolution decisions   -> SpatialEcosystem.step()
#   - User interaction      -> keyboard toggles on EvoRenderer
# GPU simulation world (spec 60):
#   - Organism positions    -> per-organism viewport/scissor
#   - Energy / resource     -> ResourceField texture (u_resource_tex)
#   - Physics / flocking    -> SpatialEcosystem.step() (CPU) -> GPU draw
#   - Shader graph exec     -> per-organism compiled ModularShaderFactory program
#   - Particle systems      -> organism genome post operators
#   - Rendering             -> territorial passes (feedback + screen)
# ============================================================================

def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    return (v, p, q)


# Complexity tiers (spec 57): evolution acquires complexity over time.
COMPLEXITY_TIERS = [
    ("fbm", "none"),                 # 0 simple noise
    ("fbm", "curl"),                 # 1 fbm
    ("fbm", "vortex"),               # 2 curl
    ("voronoi", "none"),             # 3 voronoi
    ("reaction_diffusion", "curl"),  # 4 reaction diffusion
    ("mandelbrot", "twist"),         # 5 volumetric-ish
    ("sdf", "vortex"),               # 6 raymarch
    ("sdf", "curl"),                 # 7 hybrid
]

SIGNAL_TYPES = ("danger", "food", "mate", "music")


@dataclass
class Senses:
    """Evolved sensory apparatus (spec 53).

    Different organisms perceive different parts of the music and the world, so
    they do not all react to the same stimuli.
    """

    hearing_range: float = 0.35
    bass_sensitivity: float = 1.0
    treble_sensitivity: float = 1.0
    harmony_sensitivity: float = 1.0
    motion_sensitivity: float = 1.0
    beat_prediction: float = 0.0
    colour_sensitivity: float = 1.0

    def clone(self) -> "Senses":
        return Senses(
            hearing_range=self.hearing_range,
            bass_sensitivity=self.bass_sensitivity,
            treble_sensitivity=self.treble_sensitivity,
            harmony_sensitivity=self.harmony_sensitivity,
            motion_sensitivity=self.motion_sensitivity,
            beat_prediction=self.beat_prediction,
            colour_sensitivity=self.colour_sensitivity,
        )

    def mutate(self, strength: float = 0.15) -> None:
        def rnd(attr, lo, hi):
            v = getattr(self, attr) + random.uniform(-strength, strength) * (hi - lo)
            return max(lo, min(hi, v))
        self.hearing_range = rnd("hearing_range", 0.1, 0.8)
        self.bass_sensitivity = rnd("bass_sensitivity", 0.0, 2.0)
        self.treble_sensitivity = rnd("treble_sensitivity", 0.0, 2.0)
        self.harmony_sensitivity = rnd("harmony_sensitivity", 0.0, 2.0)
        self.motion_sensitivity = rnd("motion_sensitivity", 0.0, 2.0)
        self.beat_prediction = rnd("beat_prediction", 0.0, 1.0)
        self.colour_sensitivity = rnd("colour_sensitivity", 0.2, 2.0)


@dataclass
class Organs:
    """Internal functional modules (spec 54).

    Movement, skeleton, skin, glow, particles and memory evolve separately and
    each draws from the shared energy pool.
    """

    movement: float = 0.5
    skeleton: float = 0.5
    skin: float = 0.5
    glow: float = 0.3
    particles: float = 0.3
    memory: float = 0.5

    def clone(self) -> "Organs":
        return Organs(self.movement, self.skeleton, self.skin,
                      self.glow, self.particles, self.memory)

    def mutate(self, strength: float = 0.15) -> None:
        for a in ("movement", "skeleton", "skin", "glow", "particles", "memory"):
            lo, hi = 0.0, 1.0
            v = getattr(self, a) + random.uniform(-strength, strength) * (hi - lo)
            setattr(self, a, max(lo, min(hi, v)))

    def energy_cost(self) -> float:
        return (0.4 * self.glow + 0.4 * self.particles
                + 0.15 * self.skeleton + 0.1 * self.skin)


class OrganismBrain:
    """Tiny neural controller (spec 55).

    Inputs:  bass, treble, energy, nearby pressure, age, fitness.
    Outputs: move, grow, reproduce, mutate, change_shader.
    """

    IN = 6
    HID = 5
    OUT = 5
    OUT_LABELS = ("move", "grow", "reproduce", "mutate", "change_shader")

    def __init__(self, seed: Optional[int] = None):
        rnd = random.Random(seed)
        self.w1 = [rnd.uniform(-1.0, 1.0) for _ in range(self.IN * self.HID)]
        self.w2 = [rnd.uniform(-1.0, 1.0) for _ in range(self.HID * self.OUT)]
        self.b1 = [0.0] * self.HID
        self.b2 = [0.0] * self.OUT
        self._last_h: Optional[List[float]] = None
        self._last_in: Optional[List[float]] = None
        self.fit = 0.0

    def clone(self) -> "OrganismBrain":
        b = OrganismBrain()
        b.w1 = list(self.w1)
        b.w2 = list(self.w2)
        b.b1 = list(self.b1)
        b.b2 = list(self.b2)
        return b

    def mutate(self, strength: float = 0.2) -> None:
        for k in range(len(self.w1)):
            self.w1[k] += random.uniform(-strength, strength)
        for k in range(len(self.w2)):
            self.w2[k] += random.uniform(-strength, strength)

    def process(self, inputs: List[float]) -> List[float]:
        h = []
        for j in range(self.HID):
            s = self.b1[j]
            base = j * self.IN
            for i in range(self.IN):
                s += self.w1[base + i] * inputs[i]
            h.append(math.tanh(s))
        out = []
        for k in range(self.OUT):
            s = self.b2[k]
            base = k * self.HID
            for j in range(self.HID):
                s += self.w2[base + j] * h[j]
            out.append(math.tanh(s))
        self._last_h = h
        self._last_in = list(inputs)
        return out

    def reinforce(self, reward: float, lr: float = 0.04) -> None:
        if self._last_h is None or self._last_in is None:
            return
        for k in range(self.OUT):
            base = k * self.HID
            for j in range(self.HID):
                self.w2[base + j] += lr * reward * self._last_h[j]
        for j in range(self.HID):
            base = j * self.IN
            for i in range(self.IN):
                self.w1[base + i] += lr * reward * self._last_in[i] * self._last_h[j]

    def fitness(self) -> float:
        return float(self.fit)


@dataclass
class SpeciesDef:
    """A species: colour, diet (predator/prey, spec 49) and biome affinity."""

    sid: int
    color: Tuple[float, float, float]
    diet: set = field(default_factory=set)
    biome_affinity: float = 0.5
    born_at: float = 0.0
    avg_complexity: float = 0.0


@dataclass
class Signal:
    """A broadcast message (spec 50): danger / food / mate / music."""

    x: float
    y: float
    stype: str
    species: int
    strength: float
    ttl: float


class ResourceField:
    """Invisible resource fields (spec 48) represented as textures (spec 60).

    Channels: bass, treble, harmony, motion. Deposited from audio at moving
    sources, diffused, decayed, and consumed by organisms.
    """

    def __init__(self, w: int, h: int):
        self.w = max(4, int(w))
        self.h = max(4, int(h))
        n = self.w * self.h
        self.bass = [0.0] * n
        self.treb = [0.0] * n
        self.harm = [0.0] * n
        self.mot = [0.0] * n
        self._t = 0.0

    def _idx(self, x: int, y: int) -> int:
        x = max(0, min(self.w - 1, x))
        y = max(0, min(self.h - 1, y))
        return y * self.w + x

    def deposit(self, ch: str, nx: float, ny: float, amount: float) -> None:
        arr = getattr(self, ch)
        cx = nx * (self.w - 1)
        cy = ny * (self.h - 1)
        R = 4
        sigma = 2.0
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                x = int(round(cx)) + dx
                y = int(round(cy)) + dy
                if 0 <= x < self.w and 0 <= y < self.h:
                    d2 = dx * dx + dy * dy
                    arr[y * self.w + x] = min(1.5, arr[y * self.w + x]
                                             + amount * math.exp(-d2 / (2.0 * sigma * sigma)))

    def sample(self, ch: str, nx: float, ny: float) -> float:
        arr = getattr(self, ch)
        x = max(0, min(self.w - 1, int(nx * (self.w - 1))))
        y = max(0, min(self.h - 1, int(ny * (self.h - 1))))
        return float(arr[y * self.w + x])

    def consume(self, nx: float, ny: float, frac: float = 0.25) -> Dict[str, float]:
        out = {}
        for ch in ("bass", "treb", "harm", "mot"):
            arr = getattr(self, ch)
            x = max(0, min(self.w - 1, int(nx * (self.w - 1))))
            y = max(0, min(self.h - 1, int(ny * (self.h - 1))))
            i = y * self.w + x
            amt = arr[i] * frac
            arr[i] -= amt
            out[ch] = amt
        return out

    def gradient(self, ch: str, nx: float, ny: float, eps: float = 0.02) -> Tuple[float, float]:
        gx = self.sample(ch, nx + eps, ny) - self.sample(ch, nx - eps, ny)
        gy = self.sample(ch, nx, ny + eps) - self.sample(ch, nx, ny - eps)
        return (gx, gy)

    def update(self, audio: Dict[str, float], dt: float) -> None:
        self._t += dt
        bass = float(audio.get("bass", 0.0))
        treb = float(audio.get("treb", 0.0))
        mid = float(audio.get("mid", 0.0))
        beat = float(audio.get("beat", 0.0))
        # Bass drifts left when loud, treble right (spec 58 migration drivers).
        bx = 0.5 - bass * 0.45 + 0.08 * math.sin(self._t * 0.25)
        tx = 0.5 + treb * 0.45 + 0.08 * math.cos(self._t * 0.21)
        hx = 0.5 + 0.12 * math.sin(self._t * 0.17 + mid)
        my = 0.5 + beat * 0.4
        self.deposit("bass", bx, 0.5, bass * 0.9 * dt * 12.0)
        self.deposit("treb", tx, 0.5, treb * 0.9 * dt * 12.0)
        self.deposit("harm", hx, 0.5, mid * 0.7 * dt * 12.0)
        self.deposit("mot", 0.5, my, beat * 0.8 * dt * 12.0)
        self._diffuse_decay()

    def _diffuse_decay(self) -> None:
        for arr in (self.bass, self.treb, self.harm, self.mot):
            nw = list(arr)
            w, h = self.w, self.h
            for y in range(h):
                for x in range(w):
                    i = y * w + x
                    s = arr[i] * 4.0
                    c = 0
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            xx = x + dx
                            yy = y + dy
                            if 0 <= xx < w and 0 <= yy < h:
                                s += arr[yy * w + xx]
                                c += 1
                    nw[i] = (s / (c + 4.0)) * 0.992
            arr[:] = nw

    def rgba_array(self) -> List[float]:
        out = []
        for i in range(self.w * self.h):
            out.append(min(1.0, self.bass[i]))
            out.append(min(1.0, self.treb[i]))
            out.append(min(1.0, self.harm[i]))
            out.append(min(1.0, self.mot[i]))
        return out


class Climate:
    """Seasonal music climate (spec 52).

    Slowly drifting temperature shaped by the music; warm favours fluid
    organisms, cold favours crystal organisms.
    """

    def __init__(self):
        self.temperature = 0.5
        self.season = 0.0

    def update(self, audio: Dict[str, float], dt: float) -> None:
        energy = (float(audio.get("bass", 0.0)) + float(audio.get("mid", 0.0))
                  + float(audio.get("treb", 0.0)) + float(audio.get("beat", 0.0))) / 4.0
        target = max(0.0, min(1.0, energy * 1.4))
        tau = 45.0
        k = 1.0 - math.exp(-dt / tau)
        self.temperature += (target - self.temperature) * k
        self.season += dt / 600.0

    def biome_weight(self, field_type: str) -> float:
        warm = {"fbm": 1.1, "reaction_diffusion": 1.2, "voronoi": 0.9,
                "mandelbrot": 0.85, "sdf": 0.95}.get(field_type, 1.0)
        cold = {"fbm": 0.95, "reaction_diffusion": 0.9, "voronoi": 1.2,
                "mandelbrot": 1.15, "sdf": 1.1}.get(field_type, 1.0)
        return warm * self.temperature + cold * (1.0 - self.temperature)

    def complexity_bonus(self) -> float:
        return 0.3 + 0.7 * self.temperature


class ExtinctionArchive:
    """Dormant store of extinct species (spec 59)."""

    def __init__(self):
        self.dormant: Dict[int, Dict] = {}
        self.revival_cooldown = 0.0

    def archive(self, species: SpeciesDef, genome_dict: Dict) -> None:
        self.dormant[species.sid] = {
            "color": species.color,
            "diet": list(species.diet),
            "biome_affinity": species.biome_affinity,
            "genome": genome_dict,
            "avg_complexity": species.avg_complexity,
        }

    def maybe_revive(self, dt: float) -> Optional[int]:
        self.revival_cooldown -= dt
        if self.revival_cooldown > 0.0 or not self.dormant:
            return None
        self.revival_cooldown = 20.0
        if random.random() < 0.8:
            return random.choice(list(self.dormant.keys()))
        return None


class SpatialOrganism:
    """A living organism occupying its own territory (specs 46, 47, 53-58)."""

    def __init__(self, genome, species: int, *, energy: float = 1.0,
                 pos: Optional[Tuple[float, float]] = None, generation: int = 0):
        self.genome = genome
        self.species = species
        self.pos = list(pos) if pos else [random.random(), random.random()]
        self.vel = [random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)]
        self.radius = random.uniform(0.06, 0.12)
        self.energy = energy
        self.age = 0.0
        self.life_span = random.uniform(60.0, 240.0)
        self.senses = Senses()
        self.organs = Organs()
        self.brain = OrganismBrain(seed=random.randint(0, 1 << 30))
        self.complexity_level = 0
        self.fitness = 0.0
        self.alive = True
        self.generation = generation
        self.preferred = random.choice(("bass", "treb", "harm", "mot"))
        self.signal_cooldown = 0.0
        self._prev_energy = energy

    def clone_child(self) -> "SpatialOrganism":
        g = self.genome.clone()
        g.mutate(strength=0.12)
        child = SpatialOrganism(
            g, self.species, energy=self.energy * 0.4,
            pos=(self.pos[0] + random.uniform(-0.05, 0.05),
                 self.pos[1] + random.uniform(-0.05, 0.05)),
            generation=self.generation + 1,
        )
        child.senses = self.senses.clone()
        child.senses.mutate(0.1)
        child.organs = self.organs.clone()
        child.organs.mutate(0.1)
        child.brain = self.brain.clone()
        child.brain.mutate(0.15)
        child.complexity_level = self.complexity_level
        return child

    def perceive(self, audio: Dict[str, float], field: ResourceField,
                 signals: List[Signal], neighbours: int) -> List[float]:
        s = self.senses
        vals = {
            "bass": float(audio.get("bass", 0.0)) * s.bass_sensitivity,
            "treb": float(audio.get("treb", 0.0)) * s.treble_sensitivity,
            "harm": float(audio.get("mid", 0.0)) * s.harmony_sensitivity,
            "mot": float(audio.get("beat", 0.0)) * s.motion_sensitivity,
        }
        return [
            vals["bass"], vals["treb"],
            max(0.0, min(1.0, self.energy / 3.0)),
            min(1.0, neighbours / 8.0),
            min(1.0, self.age / self.life_span),
            max(0.0, min(1.0, self.fitness)),
        ]

    def step(self, dt: float, eco: "SpatialEcosystem", audio: Dict[str, float],
             neighbours: List["SpatialOrganism"]) -> None:
        if not self.alive:
            return
        self.age += dt
        self.signal_cooldown = max(0.0, self.signal_cooldown - dt)
        n_in = len(neighbours)
        inputs = self.perceive(audio, eco.field, eco.signals, n_in)
        acts = [0.0, 0.0, 0.0, 0.0, 0.0]
        if eco.config.enable_brain:
            acts = self.brain.process(inputs)
        move_a, grow_a, repro_a, mutate_a, shader_a = acts

        self._move(dt, eco, audio, neighbours, move_a)
        intake = self._consume(eco.field, dt)
        self._pay_costs(dt, eco)
        self._act(dt, eco, audio, grow_a, repro_a, mutate_a, shader_a, intake)
        self._emit_signal(eco, audio, intake)

        # Carrying capacity: energy cannot grow without bound (spec 56).
        self.energy = min(eco.max_energy * 1.15, self.energy)

        # rolling fitness
        inst = (max(0.0, min(1.0, self.energy / 3.0)) * 0.5
                + min(1.0, intake) * 0.3 + eco.climate.biome_weight(
                    getattr(self.genome, "field_type", "fbm")) * 0.2)
        self.fitness = self.fitness * 0.9 + inst * 0.1
        self.brain.fit = self.brain.fit * 0.95 + (inst - 0.5) * 0.05

        if self.energy <= 0.0 or self.age >= self.life_span:
            self.alive = False

    def _move(self, dt, eco, audio, neighbours, move_a) -> None:
        ax, ay = 0.0, 0.0
        mv = self.organs.movement if eco.config.enable_organs else 0.5
        rng = self.senses.hearing_range if eco.config.enable_senses else 0.35

        # Flocking (spec 51): alignment, cohesion, separation.
        if eco.config.enable_flocking and neighbours:
            avx = avy = 0.0
            cx = cy = 0.0
            sx = sy = 0.0
            for o in neighbours:
                avx += o.vel[0]
                avy += o.vel[1]
                cx += o.pos[0]
                cy += o.pos[1]
                d = math.hypot(o.pos[0] - self.pos[0], o.pos[1] - self.pos[1]) or 1e-3
                if d < rng * 0.5:
                    sx += (self.pos[0] - o.pos[0]) / d
                    sy += (self.pos[1] - o.pos[1]) / d
            n = len(neighbours)
            ax += (avx / n) * 0.4 + sx * 0.6
            ay += (avy / n) * 0.4 + sy * 0.6
            ax += (cx / n - self.pos[0]) * 0.3
            ay += (cy / n - self.pos[1]) * 0.3

        # Migration (spec 58): climb the preferred resource gradient.
        if eco.config.enable_migration:
            gx, gy = eco.field.gradient(self.preferred, self.pos[0], self.pos[1])
            ax += gx * 0.8 * (self.senses.motion_sensitivity if eco.config.enable_senses else 1.0)
            ay += gy * 0.8
            # Lateral music current.
            ax += (float(audio.get("treb", 0.0)) - float(audio.get("bass", 0.0))) * 0.05

        # Signal reactions (spec 50).
        if eco.config.enable_signals:
            for sig in eco.signals:
                d = math.hypot(sig.x - self.pos[0], sig.y - self.pos[1])
                if d > rng:
                    continue
                w = (1.0 - d / rng)
                if sig.stype == "danger":
                    ax += (self.pos[0] - sig.x) * w * 1.2
                    ay += (self.pos[1] - sig.y) * w * 1.2
                elif sig.stype in ("food", "music"):
                    ax += (sig.x - self.pos[0]) * w * 0.8
                    ay += (sig.y - self.pos[1]) * w * 0.8
                elif sig.stype == "mate" and sig.species == self.species:
                    ax += (sig.x - self.pos[0]) * w * 1.0
                    ay += (sig.y - self.pos[1]) * w * 1.0

        self.vel[0] = (self.vel[0] + ax * dt * 0.6) * 0.96
        self.vel[1] = (self.vel[1] + ay * dt * 0.6) * 0.96
        sp = math.hypot(self.vel[0], self.vel[1])
        maxsp = 0.15 * (0.4 + mv)
        if sp > maxsp:
            self.vel[0] *= maxsp / sp
            self.vel[1] *= maxsp / sp
        self.pos[0] += self.vel[0] * dt * (0.5 + 0.5 * abs(move_a))
        self.pos[1] += self.vel[1] * dt * (0.5 + 0.5 * abs(move_a))
        # Bounce off world edges.
        if self.pos[0] < 0.02:
            self.pos[0] = 0.02
            self.vel[0] = abs(self.vel[0])
        if self.pos[0] > 0.98:
            self.pos[0] = 0.98
            self.vel[0] = -abs(self.vel[0])
        if self.pos[1] < 0.02:
            self.pos[1] = 0.02
            self.vel[1] = abs(self.vel[1])
        if self.pos[1] > 0.98:
            self.pos[1] = 0.98
            self.vel[1] = -abs(self.vel[1])

    def _consume(self, field: ResourceField, dt: float) -> float:
        sens = self.senses
        eaten = field.consume(self.pos[0], self.pos[1], 0.2)
        gain = (eaten["bass"] * sens.bass_sensitivity
                + eaten["treb"] * sens.treble_sensitivity
                + eaten["harm"] * sens.harmony_sensitivity
                + eaten["mot"] * sens.motion_sensitivity)
        gain = min(1.5, gain * 1.5 * dt * 8.0)
        self.energy += gain
        return gain

    def _visual_cost(self, eco) -> float:
        g = self.genome
        ft = getattr(g, "field_type", "fbm")
        base = {"fbm": 0.2, "voronoi": 0.3, "reaction_diffusion": 0.5,
                "mandelbrot": 0.6, "sdf": 1.0}.get(ft, 0.2)
        posts = getattr(g, "post", None) or []
        base += 0.08 * len(posts)
        if eco.config.enable_organs:
            base += 0.5 * self.organs.energy_cost()
        base += 0.12 * self.complexity_level
        return base

    def _pay_costs(self, dt: float, eco) -> None:
        if not eco.config.enable_energy_economy:
            return
        cost = (0.012 + self._visual_cost(eco)) * dt
        # Luxury metabolism: high energy is expensive to maintain (spec 56),
        # which bounds the population's energy and keeps GPU-heavy organisms in check.
        cost += 0.08 * max(0.0, self.energy - 1.0) * dt
        self.energy -= cost

    def _act(self, dt, eco, audio, grow_a, repro_a, mutate_a, shader_a, intake) -> None:
        # Grow (spec 47): radius tracks energy.
        en = max(0.0, min(1.0, self.energy / 3.0))
        target_r = 0.05 + 0.10 * en + 0.03 * self.complexity_level
        self.radius += (target_r - self.radius) * 0.1

        # Mutation (spec 57 via brain, plus time-based drift).
        if eco.config.enable_brain and shader_a > 0.4 and random.random() < 0.05:
            self.genome.mutate(strength=0.1)
        elif random.random() < 0.02:
            self.genome.mutate(strength=0.08)

        # Evolution of complexity (spec 57): acquire higher tiers when energy
        # is plentiful and the climate allows it.
        if (eco.config.enable_complexity and self.energy > 1.3
                and self.complexity_level < 7
                and random.random() < 0.02 * (0.5 + eco.climate.complexity_bonus())):
            self.complexity_level += 1
            ft, wp = COMPLEXITY_TIERS[self.complexity_level]
            try:
                self.genome.field_type = ft
                self.genome.warp = wp
            except Exception:
                pass
            self.energy -= 0.3

        # Reproduction (spec 46): spawn a child when thriving.
        if (self.energy > 1.3
                and (not eco.config.enable_brain or repro_a > 0.2)
                and random.random() < 0.10):
            child = self.clone_child()
            eco.spawn_child(child)
            self.energy -= 0.4

    def _emit_signal(self, eco, audio, intake) -> None:
        if not eco.config.enable_signals or self.signal_cooldown > 0.0:
            return
        stype = None
        if self.energy < 0.3:
            stype = "danger"
        elif intake < 0.01 and self.energy > 1.0:
            stype = "food"
        elif self.energy > 1.4 and self.age > self.life_span * 0.4:
            stype = "mate"
        elif float(audio.get("beat", 0.0)) > 0.7:
            stype = "music"
        if stype:
            eco.signals.append(Signal(self.pos[0], self.pos[1], stype,
                                      self.species, 1.0, 2.5))
            self.signal_cooldown = 3.0


class SpatialEcosystem:
    """CPU-supervised spatial ecosystem (specs 46-60)."""

    def __init__(self, config: "EvoConfig"):
        self.config = config
        self.max_energy = 3.0
        self.t = 0.0
        self.step_count = 0
        self.field = ResourceField(config.spatial_world_w, config.spatial_world_h)
        self.climate = Climate()
        self.signals: List[Signal] = []
        self.archive = ExtinctionArchive()
        self.species: Dict[int, SpeciesDef] = {}
        self.organisms: List[SpatialOrganism] = []
        self._next_sid = 0
        self._spawned_this_step: List[SpatialOrganism] = []

        n_species = 4
        for i in range(n_species):
            self._new_species(diet=self._random_diet(n_species, i))

        for _ in range(config.spatial_population_size):
            sid = random.randrange(n_species)
            org = SpatialOrganism(ModularGenome(), sid, energy=random.uniform(1.0, 1.6))
            self.organisms.append(org)

    def _new_species(self, diet: Optional[set] = None) -> int:
        sid = self._next_sid
        self._next_sid += 1
        hue = (sid * 0.137) % 1.0
        color = _hsv_to_rgb(hue, 0.7, 1.0)
        self.species[sid] = SpeciesDef(
            sid=sid, color=color, diet=set(diet or set()),
            biome_affinity=random.random(), born_at=self.t)
        return sid

    def _random_diet(self, n: int, i: int) -> set:
        if n <= 1:
            return set()
        prey = (i + 1) % n
        return {prey} if random.random() < 0.35 else set()

    def spawn_child(self, child: SpatialOrganism) -> None:
        self._spawned_this_step.append(child)

    def _neighbours(self, org: SpatialOrganism) -> List[SpatialOrganism]:
        rng = getattr(org.senses, "hearing_range", 0.35)
        out = []
        for o in self.organisms:
            if o is org or not o.alive:
                continue
            if math.hypot(o.pos[0] - org.pos[0], o.pos[1] - org.pos[1]) < rng:
                out.append(o)
        return out

    def step(self, dt: float, audio: Dict[str, float],
             music_dna: Any = None) -> None:
        dt = min(0.1, max(1e-3, dt))
        self.t += dt
        self.step_count += 1

        self.field.update(audio, dt)
        if self.config.enable_climate:
            self.climate.update(audio, dt)

        for sig in self.signals:
            sig.ttl -= dt
            sig.strength = max(0.0, sig.strength - dt * 0.3)
        self.signals = [s for s in self.signals if s.ttl > 0.0 and s.strength > 0.01]

        for org in self.organisms:
            if org.alive:
                org.step(dt, self, audio, self._neighbours(org))

        if self.config.enable_predator_prey:
            self._predation()

        # Integrate births.
        if self._spawned_this_step:
            for c in self._spawned_this_step:
                if len(self.organisms) < self.config.spatial_population_size:
                    self.organisms.append(c)
                else:
                    weakest = min(self.organisms, key=lambda o: o.fitness)
                    self.organisms.remove(weakest)
                    self.organisms.append(c)
            self._spawned_this_step = []

        self._extinction_check()
        if self.config.enable_extinction:
            alive_count = sum(1 for o in self.organisms if o.alive)
            self._maybe_revive(dt, force=alive_count < 4)
        if random.random() < 0.004:
            self._maybe_speciate()
        self._mutate_diets()

        self.organisms = [o for o in self.organisms if o.alive]
        self._update_species_stats()

    def _predation(self) -> None:
        for pred in self.organisms:
            if not pred.alive:
                continue
            diet = self.species.get(pred.species)
            if not diet or not diet.diet:
                continue
            for prey in self.organisms:
                if not prey.alive or prey is pred:
                    continue
                if prey.species not in diet.diet:
                    continue
                d = math.hypot(prey.pos[0] - pred.pos[0], prey.pos[1] - pred.pos[1])
                if d < pred.radius + prey.radius + 0.02 and pred.energy < 2.2:
                    eat = min(0.06, prey.energy * 0.25)
                    pred.energy = min(self.max_energy, pred.energy + eat * 0.8)
                    prey.energy -= eat
                    if prey.energy <= 0.0:
                        prey.alive = False

    def _extinction_check(self) -> None:
        present = {o.species for o in self.organisms if o.alive}
        for sid in list(self.species.keys()):
            if sid not in present:
                best = max((o for o in self.organisms if o.species == sid),
                           key=lambda o: o.fitness, default=None)
                if best is not None:
                    self.archive.archive(self.species[sid], best.genome.to_dict())
                del self.species[sid]

    def _maybe_revive(self, dt: float, force: bool = False) -> None:
        if force:
            # World is too empty: revive a dormant species immediately (spec 59 recovery).
            if self.archive.dormant:
                sid = random.choice(list(self.archive.dormant.keys()))
                self.archive.revival_cooldown = 20.0
            else:
                sid = None
        else:
            sid = self.archive.maybe_revive(dt)
        if sid is None or sid in self.species:
            return
        rec = self.archive.dormant[sid]
        new_sid = self._new_species(diet=set(rec["diet"]))
        self.species[new_sid].color = (
            min(1.0, rec["color"][0] + random.uniform(-0.2, 0.2)),
            min(1.0, rec["color"][1] + random.uniform(-0.2, 0.2)),
            min(1.0, rec["color"][2] + random.uniform(-0.2, 0.2)),
        )
        g = ModularGenome()
        try:
            g.from_dict(rec["genome"])
        except Exception:
            pass
        for _ in range(2):
            org = SpatialOrganism(g.clone(), new_sid,
                                  energy=random.uniform(0.9, 1.3))
            lvl = int(min(7, rec.get("avg_complexity", 0.0) * 7 + random.uniform(-1, 1)))
            org.complexity_level = max(0, lvl)
            self.organisms.append(org)

    def _maybe_speciate(self) -> None:
        if len(self.species) >= 8:
            return
        sid = self._new_species(diet=self._random_diet(len(self.species) + 1, 0))
        org = SpatialOrganism(ModularGenome(), sid, energy=random.uniform(1.0, 1.4))
        self.organisms.append(org)

    def _mutate_diets(self) -> None:
        if len(self.species) < 2:
            return
        for sp in self.species.values():
            if random.random() < 0.01:
                others = [s for s in self.species if s != sp.sid]
                if others:
                    tgt = random.choice(others)
                    if tgt in sp.diet:
                        sp.diet.discard(tgt)
                    else:
                        sp.diet.add(tgt)

    def _update_species_stats(self) -> None:
        for sid, sp in self.species.items():
            members = [o for o in self.organisms if o.alive and o.species == sid]
            if members:
                sp.avg_complexity = sum(o.complexity_level for o in members) / len(members)

    def alive_organisms(self) -> List[SpatialOrganism]:
        return [o for o in self.organisms if o.alive]


class EvoRenderer(Gtk.GLArea):
    """Real-time evolutionary visual renderer using OpenGL and GTK.
    
    Features:
      - Real-time shader generation from evolving genome parameters
      - Feedback-based rendering with ping-pong buffers
      - Audio-reactive effects
      - Human feedback steering (like/dislike)
      - Keyboard controls for mutation and presets
    """
    
    VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;
out vec2 v_uv;
void main(){
    v_uv = texcoord;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

    def __init__(self, config: EvoConfig):
        """Initialize renderer.
        
        Args:
            config: EvoConfig instance with renderer parameters
        """
        super().__init__()

        self._config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_halign(Gtk.Align.FILL)
        self.set_valign(Gtk.Align.FILL)
        self.set_required_version(3, 3)
        self.set_has_depth_buffer(False)
        self.set_has_stencil_buffer(False)
        self.set_auto_render(True)
        self.set_visible(True)

        self.genome = ModularGenome()
        self.shader_factory = ModularShaderFactory()
        # on_evict releases the GL program so evicted entries do not leak on the
        # GPU (the root cause of the OOM). The cache never evicts the program
        # currently bound because _set_program calls shader_cache.protect().
        self.shader_cache = ShaderCache(on_evict=self._delete_program)

        self.audio = AudioState()
        self._preference_bias = 0.0

        self._population_view = False

        self.evolve_timer = EvolutionTimer(self._config.mutation_interval_secs)

        self._vao: Optional[int] = None
        self._vbo: Optional[int] = None
        self._program: Optional[int] = None
        self.pending_delete: List[int] = []
        self.gl_ready = False
        self._tick_id: Optional[int] = None

        self._loc_time = -1
        self._loc_res = -1
        self._loc_audio = -1  # legacy (unused in modular)
        self._loc_prev = -1   # legacy (unused in modular)
        self._loc_fb_alpha = -1  # legacy (unused in modular)
        self._loc_brightness = -1  # legacy (unused in modular)

        self._loc_bass = -1
        self._loc_mid = -1
        self._loc_treble = -1
        self._loc_previous_frame = -1

        # Music DNA uniforms
        self._loc_md_bpm = -1
        self._loc_md_energy = -1
        self._loc_md_danceability = -1
        self._loc_md_beat_strength = -1
        self._loc_md_spectral_centroid = -1
        self._loc_md_harmonic_complexity = -1
        self._loc_md_dynamic_range = -1
        self._loc_md_key = -1
        self._loc_md_mood = -1

        self.feedback = FeedbackBuffer(self._config.feedback_res)
        self._start_time = time.time()
        self._generation = 0  # renderer-level evolution step counter (logging only)

        # Population system
        self.population = Population(
            size=self._config.population_size,
            crossover_rate=self._config.crossover_rate,
            genome_factory=lambda: ModularGenome(generation=0),
        )
        # Keep Population[0] in sync with the renderer's current genome.
        self.population.organisms[0] = self.genome


        # Fitness and lineage
        self.fitness_evaluator = FitnessEvaluator(res=self._config.fitness_image_res)
        self.memory_dna = MemoryDNA(
            preset_dir=self._config.preset_dir / "lineage" if self._config.preset_dir else None
        )
        self._previous_image: Optional[bytes] = None
        self._fitness_pbo: Optional[int] = None
        self._fitness_fbo: Optional[int] = None
        self._fitness_tex: Optional[int] = None

        # Ecosystem memory
        self.ecosystem_memory: Optional[EvolutionMemory] = None
        self._music_pattern_interval_secs = 15.0
        self._last_music_pattern_ts = time.time()
        self._world_evolution_timer = time.time()
        self._dream_active = False
        self._current_music_sig: Optional[MusicSignature] = None
        self._current_music_dna: Optional[Any] = None
        self._last_beat_evolution_ts: float = 0.0
        self._beat_evolution_cooldown: float = 2.0

        if self._config.enable_ecosystem_memory:
            self.ecosystem_memory = EvolutionMemory(db_path=self._config.memory_db_path)
            self.ecosystem_memory.load_world_dna()
            self._logger.info("Ecosystem memory enabled")

        # Connect GTK signals
        self.connect("realize", self.on_realize)
        self.connect("unrealize", self.on_unrealize)
        self.connect("render", self.on_render)
        self.connect("resize", self.on_resize)

        # Population visualization: 'p' toggles the grid, clicking a tile
        # promotes that organism to the champion.
        try:
            key_controller = Gtk.EventControllerKey()
            key_controller.connect("key-pressed", self._on_key_pressed)
            self.add_controller(key_controller)

            click_controller = Gtk.GestureClick()
            click_controller.connect("pressed", self._on_population_click)
            self.add_controller(click_controller)
        except Exception as e:
            self._logger.debug(f"Could not attach population-view controllers: {e}")


        # Spatial ecosystem (specs 46-60)
        self.ecosystem: Optional[SpatialEcosystem] = None
        self._spatial_active = bool(self._config.enable_spatial_ecosystem)
        if self._config.enable_spatial_ecosystem:
            try:
                self.ecosystem = SpatialEcosystem(self._config)
                self._logger.info(
                    f"Spatial ecosystem enabled: {len(self.ecosystem.organisms)} organisms, "
                    f"{len(self.ecosystem.species)} species"
                )
            except Exception as e:
                self._logger.error(f"Spatial ecosystem init failed: {e}")
                self.ecosystem = None
                self._spatial_active = False
        self._eco_programs: Dict[str, int] = {}
        self._eco_loc: Dict[int, Dict[str, int]] = {}
        self._eco_used: Dict[str, int] = {}
        self._resource_tex: Optional[int] = None
        self._resource_tex_size: Tuple[int, int] = (0, 0)
        self._last_frame_ts = time.time()
        self._cur_eco_prog: Optional[int] = None

        self._logger.info(f"EvoRenderer initialized with config: {asdict(config)}")

    def update_audio(self, audio_levels: Dict[str, float]) -> None:
        """Update audio levels for reactive effects.
        
        Args:
            audio_levels: Dict with 'bass', 'mid', 'treb', 'beat' keys
        """
        self.audio.update(audio_levels)

    def _init_fitness_fbo(self) -> None:
        """Initialize a small FBO for fitness readback."""
        res = self._config.fitness_image_res
        try:
            self._fitness_tex = GL.glGenTextures(1)
            self._fitness_fbo = GL.glGenFramebuffers(1)
            
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._fitness_tex)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F,
                res, res, 0, GL.GL_RGBA, GL.GL_HALF_FLOAT, None
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fitness_fbo)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                GL.GL_TEXTURE_2D, self._fitness_tex, 0
            )
            
            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"Fitness FBO incomplete: {hex(status)}")
            
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            self._logger.debug(f"Fitness FBO initialized: {res}x{res}")
        except Exception as e:
            self._logger.warning(f"Fitness FBO init failed: {e}")

    def _capture_and_score(self, genome=None) -> float:
        """Capture current frame and score fitness.
        
        Args:
            genome: Genome to score. Defaults to self.genome.
            
        Returns:
            Fitness score
        """
        if genome is None:
            genome = self.genome
        if self._fitness_fbo is None or self._fitness_tex is None:
            genome.fitness = self.fitness_evaluator.score_from_genome(genome)
            return genome.fitness
        
        res = self._config.fitness_image_res
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fitness_fbo)
            GL.glViewport(0, 0, res, res)
            GL.glUseProgram(self._program)
            GL.glBindVertexArray(self._vao)
            
            bass, mid, treb, beat = self.audio.uvec()
            if self._loc_time >= 0:
                GL.glUniform1f(self._loc_time, float(time.time() - self._start_time))
            if self._loc_res >= 0:
                GL.glUniform2f(self._loc_res, float(res), float(res))

            # Modular uniforms
            if self._loc_bass >= 0:
                GL.glUniform1f(self._loc_bass, bass)
            if self._loc_mid >= 0:
                GL.glUniform1f(self._loc_mid, mid)
            if self._loc_treble >= 0:
                GL.glUniform1f(self._loc_treble, treb)

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.feedback.src_tex())
            if self._loc_previous_frame >= 0:
                GL.glUniform1i(self._loc_previous_frame, 0)
            
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)
            GL.glFinish()
            
            pixels = GL.glReadPixels(0, 0, res, res, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            image_data = bytes(bytearray(pixels)) if not isinstance(pixels, bytes) else pixels
            
            current_fitness = self.fitness_evaluator.evaluate(
                genome, image_data, self._previous_image
            )
            
            self._previous_image = image_data
            genome.fitness = current_fitness
            
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            
            return current_fitness
        except Exception as e:
            self._logger.debug(f"Fitness capture failed: {e}")
            genome.fitness = self.fitness_evaluator.score_from_genome(genome)
            return genome.fitness

    def score(self, candidate: Any) -> float:
        """Score a genome based on visual fitness or configured weights."""
        base = float(candidate.fitness)
        if self._config.enable_visual_fitness:
            base = float(candidate.fitness)
        else:
            base = float(self.fitness_evaluator.score_from_genome(candidate, self._config.score_weights))
        if self._config.enable_ecosystem_memory:
            return self._compute_long_term_fitness(candidate)
        return base

    def evolve(self) -> None:
        """Perform one evolution step: mutate and potentially accept.
        
        Uses population-based evolution when population_size > 1,
        otherwise falls back to single-organism hill climbing.

        Genetic mutation events (trick 10): in single-organism mode there is
        a small chance of a dramatic ``metamorphosis`` (species jump) instead
        of a single local mutation, so the organism can escape local optima.
        """
        self._generation += 1
        self._logger.info(f"Evolution triggered (gen {self._generation}, pop={self._config.population_size})")

        # Rare dramatic metamorphosis (genetic mutation event).
        if self._config.population_size <= 1 and random.random() < 0.02:
            self.genome.metamorphosis()
            frag = self.shader_factory.generate(self.genome)
            self._recompile_fragment_shader(frag, self.genome)
            self._logger.info(f"Metamorphosis event: {self.genome.mutations[-1] if self.genome.mutations else 'unknown'}")

        if self._config.population_size > 1:
            self._evolve_population()
        else:
            self._evolve_single()

    def _score_genome(self, genome) -> float:
        """Render a genome to the fitness FBO and return its fitness score.

        Uses the shader cache so candidates sharing a genome state reuse a
        compiled program instead of recompiling every time. The active program
        is swapped to the candidate and restored to the current champion
        afterwards.
        """
        current_genome = self.genome
        program = self._get_program(genome)
        self._set_program(program)
        score = self._capture_and_score(genome)
        self._set_program(self._get_program(current_genome))
        self.genome = current_genome
        return score

    def _evolve_single(self) -> None:
        """Single-organism hill climbing evolution."""
        old = self.genome
        candidate = copy.deepcopy(old)
        candidate.mutate()
        
        self._logger.info(f"Mutation: {candidate.mutations[-1] if candidate.mutations else 'unknown'}")

        old_fitness = self.score(old)
        candidate_fitness = self._score_genome(candidate)

        if candidate_fitness > old_fitness:
            self._set_program(self._get_program(candidate))
            self.genome = candidate
            old.record_accept(True)
            self._logger.info(f"Mutation accepted (fitness {old_fitness:.3f} -> {candidate_fitness:.3f})")
            if self._config.enable_ecosystem_memory and self.ecosystem_memory:
                self.ecosystem_memory.record_species_died(old.lineage, cause="replaced")
                rec = SpeciesRecord(
                    genome_json=json.dumps(self.genome.to_dict()),
                    lineage=self.genome.lineage,
                    generation=self.genome.generation,
                    fitness=float(self.genome.fitness),
                    historical_fitness=float(self.genome.fitness),
                    adaptability=float(getattr(self.genome, "_strength", 0.1)),
                )
                self.ecosystem_memory.record_species_born(rec)
        else:
            old.record_accept(False)
            self._logger.info(f"Mutation rejected (fitness {old_fitness:.3f} vs {candidate_fitness:.3f})")

        self.genome.adapt_strength(
            self._config.mutation_strength_min,
            self._config.mutation_strength_max,
        )
        
        self._logger.info(f"Generation: {self._generation}, strength: {self.genome._strength:.4f}")

    def diversity_bonus(self, genome, population) -> float:
        """Return a 0..1 diversity pressure for ``genome`` within ``population``.

        Stranger organisms score higher, keeping the ecosystem from collapsing
        into a single converged phenotype.
        """
        if len(population) <= 1:
            return 0.0
        distance = 0.0
        for other in population:
            if other is genome:
                continue
            try:
                distance += genome.distance(other)
            except Exception:
                pass
        return min(distance / max(len(population), 1), 1.0)

    def _evolve_population(self) -> None:
        """Population-based evolution with fitness evaluation.

        Fitness combines the visual score with phenotype diversity pressure and
        audio-environmental affinity so strange organisms survive and the
        population adapts to the music.
        """
        self._logger.info(f"Evaluating population of {len(self.population.organisms)} organisms")

        organisms = self.population.organisms
        old_lineages = {getattr(org, "lineage", str(i)) for i, org in enumerate(organisms)}

        for i, org in enumerate(organisms):
            base = self._score_genome(org)
            div = self.diversity_bonus(org, organisms)
            env = org.environmental_affinity(self.audio.audio)
            org.fitness = base + self._preference_bias * 0.1 + div * 0.15 + env * 0.1
            self._logger.debug(
                f"Organism {i}: base={base:.3f} div={div:.3f} env={env:.3f} "
                f"fitness={org.fitness:.3f}, field={getattr(org, 'field_type', None)}, "
                f"warp={getattr(org, 'warp', None)}"
            )

        # Evolve population
        self.population.evolve(lambda g: g.fitness)

        new_lineages = {getattr(org, "lineage", str(i)) for i, org in enumerate(self.population.organisms)}

        # Select best as current render target
        best = self.population.best()
        stats = self.population.stats()

        self._logger.info(f"Generation advanced: {self._generation}, best_fitness: {stats['best_fitness']:.3f}")

        if best != self.genome:
            self._set_program(self._get_program(best))
            self.genome = best
            if self._config.enable_lineage:
                self.memory_dna.record(self.genome)
            self._logger.info(f"New champion: {best.lineage}, mutations: {best.mutations[-3:] if best.mutations else 'none'}")

            if self._config.enable_ecosystem_memory and self.ecosystem_memory:
                for lineage in old_lineages - new_lineages:
                    self.ecosystem_memory.record_species_died(lineage, cause="replaced")
                rec = SpeciesRecord(
                    genome_json=json.dumps(self.genome.to_dict()),
                    lineage=self.genome.lineage,
                    generation=self.genome.generation,
                    fitness=float(self.genome.fitness),
                    historical_fitness=float(self.genome.fitness),
                    adaptability=float(getattr(self.genome, "_strength", 0.1)),
                )
                self.ecosystem_memory.record_species_born(rec)

    def human_feedback(self, sentiment: float) -> None:
        """Record human preference to steer evolution.
        
        Args:
            sentiment: Positive (like) or negative (dislike) feedback
        """
        self._preference_bias += sentiment * 0.1
        self._preference_bias = max(-1.0, min(1.0, self._preference_bias))
        
        if self._config.enable_lineage:
            self.memory_dna.record(self.genome)
        
        self._logger.info(f"Human feedback: {sentiment:+.2f}, bias: {self._preference_bias:+.2f}")

    # ------------------------------------------------------------------
    # Ecosystem memory integration
    # ------------------------------------------------------------------

    def _update_music_signature(self) -> None:
        if not self._config.enable_ecosystem_memory or not self.ecosystem_memory:
            return
        self._current_music_sig = MusicSignature.from_audio(self.audio)
        try:
            from evo_renderer.audio_analyzer import AudioAnalyzer
            analyzer = AudioAnalyzer()
            if hasattr(self.audio, "raw") and self.audio.raw is not None:
                self._current_music_dna = analyzer.analyze(self.audio.raw, self.audio.sample_rate)
        except Exception:
            pass

    def _check_dream_mode(self) -> bool:
        if not self._config.enable_dream_mode or not self.ecosystem_memory:
            return False
        energy = float(getattr(self.audio, "energy", 0.0))
        if energy < self._config.dream_threshold and not self._dream_active:
            self._dream_active = True
            self._logger.info(f"Dream mode activated (energy={energy:.3f})")
            return True
        elif energy >= self._config.dream_threshold and self._dream_active:
            self._dream_active = False
            self._logger.info("Dream mode deactivated")
        return self._dream_active

    def _maybe_store_music_pattern(self) -> None:
        if not self._config.enable_ecosystem_memory or not self.ecosystem_memory or not self._current_music_sig:
            return
        now = time.time()
        if now - self._last_music_pattern_ts >= self._music_pattern_interval_secs:
            self._last_music_pattern_ts = now
            self.ecosystem_memory.store_music_pattern(self._current_music_sig, self.genome, self.genome.fitness)

    def _maybe_evolve_world(self) -> None:
        if not self._config.enable_world_evolution or not self.ecosystem_memory:
            return
        now = time.time()
        if now - self._world_evolution_timer >= self._config.world_evolution_interval_secs:
            self._world_evolution_timer = now
            self.ecosystem_memory.evolve_world()
            self._logger.debug(f"World DNA evolved: {self.ecosystem_memory.world_dna.to_dict()}")

    def _maybe_revive_species(self) -> None:
        if not self._config.enable_ecosystem_memory or not self.ecosystem_memory or not self._current_music_sig:
            return
        match = self.ecosystem_memory.revive_species(self._current_music_sig, threshold=self._config.music_match_threshold)
        if match:
            self._logger.info(f"Species revived from music match (similarity={match['similarity']:.3f}, section={match['section']})")

    def _compute_long_term_fitness(self, genome) -> float:
        if not self._config.enable_ecosystem_memory or not self.ecosystem_memory:
            return float(genome.fitness)
        lineage_id = getattr(genome, "lineage", "")
        adaptability = float(getattr(genome, "_strength", 0.1))
        return self.ecosystem_memory.long_term_fitness(lineage_id, float(genome.fitness), adaptability)

    def dream_once(self) -> None:
        """Trigger a single dream-mode mutation."""
        if not self._config.enable_ecosystem_memory or not self.ecosystem_memory:
            return
        dreams = self.ecosystem_memory.dream(lambda: ModularGenome(generation=self.genome.generation + 1), count=1)
        if dreams:
            candidate = dreams[0]
            frag = self.shader_factory.generate(candidate)
            self._recompile_fragment_shader(frag, candidate)
            self.genome = candidate

    def save_ecosystem(self, path: Optional[Union[str, Path]] = None) -> bool:
        if not self.ecosystem_memory:
            return False
        if path is None and self._config.preset_dir:
            path = self._config.preset_dir / "ecosystem_memory.json"
        if path is None:
            return False
        return self.ecosystem_memory.save(path)

    def load_ecosystem(self, path: Union[str, Path]) -> bool:
        if not self.ecosystem_memory:
            return False
        return self.ecosystem_memory.load(path)

    def trigger_transition(self) -> None:
        """Trigger a large mutation.

        Compiles the new shader before swapping the genome so the rendered
        program and the genome stay in sync. A compile failure propagates
        rather than silently reverting to the previous genome.
        """
        candidate = copy.deepcopy(self.genome)
        candidate.mutate(strength=self._config.mutation_strength_max)
        frag = self.shader_factory.generate(candidate)
        self._recompile_fragment_shader(frag, candidate)
        self.genome = candidate
        if self._config.enable_lineage:
            self.memory_dna.record(self.genome)

    def save_preset(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Save current genome as a preset file.
        
        Args:
            path: Save path. If None, uses config preset_dir with auto-generated name.
            
        Returns:
            True if successful
        """
        try:
            if path is None:
                if not self._config.preset_dir:
                    self._logger.error("No preset directory configured")
                    return False
                path = self._config.preset_dir / f"preset_{int(time.time())}.json"
            else:
                path = Path(path)
            
            path.parent.mkdir(parents=True, exist_ok=True)
            
            preset_data = {
                "timestamp": time.time(),
                "genome": self.genome.to_dict(),
                "preference_bias": self._preference_bias,
                "lineage": self.genome.lineage,
                "generation": self.genome.generation,
                "parent_id": self.genome.parent_id,
                "fitness": self.genome.fitness,
                "mutations": list(self.genome.mutations),
                "population_stats": self.population.stats() if self._config.population_size > 1 else None,
            }
            
            with open(path, "w") as f:
                json.dump(preset_data, f, indent=2, default=str)
            
            self._logger.info(f"Preset saved to {path}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to save preset: {e}")
            return False

    def load_preset(self, path: Union[str, Path]) -> bool:
        """Load genome from a preset file.
        
        Args:
            path: Preset file path
            
        Returns:
            True if successful
        """
        try:
            path = Path(path)
            with open(path, "r") as f:
                preset_data = json.load(f)
            
            self.genome.from_dict(preset_data.get("genome", {}))
            self._preference_bias = float(preset_data.get("preference_bias", 0.0))
            self.genome.lineage = preset_data.get("lineage", self.genome.lineage)
            self.genome.generation = int(preset_data.get("generation", self.genome.generation))
            self.genome.parent_id = preset_data.get("parent_id")
            self.genome.fitness = float(preset_data.get("fitness", 0.0))
            self.genome.mutations = list(preset_data.get("mutations", []))
            
            if self.gl_ready:
                frag = self.shader_factory.generate(self.genome)
                self._recompile_fragment_shader(frag, self.genome)
            
            self._logger.info(f"Preset loaded from {path}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to load preset: {e}")
            return False

    def on_resize(self, area: Gtk.GLArea, width: int, height: int) -> None:
        """Handle GLArea resize and request re-render.
        
        GTK reports sizes in logical pixels; shaders/GL viewport must use
        framebuffer pixels, so we store scaled dimensions.
        """
        try:
            scale = float(area.get_scale_factor())
            self._last_glarea_w = int(width * scale)
            self._last_glarea_h = int(height * scale)
            # Ensure we redraw with the updated viewport.
            self.queue_render()
        except Exception as e:
            self._logger.debug(f"on_resize failed: {e}")

    def on_realize(self, area: Gtk.GLArea) -> None:
        """Handle widget realization and GL initialization.
        
        Args:
            area: The GLArea widget
        """
        try:
            self.make_current()

            self._logger.info("OpenGL context made current")

            try:
                self._create_fullscreen_quad()
                self._logger.debug("Fullscreen quad created")
            except Exception as e:
                self._logger.error(f"Failed to build fullscreen quad: {e}")
                raise

            try:
                self.feedback.init_gl()
                self._logger.debug("Feedback buffer initialized")
            except Exception as e:
                self._logger.error(f"Failed to init feedback buffer: {e}")
                raise

            try:
                if self._config.enable_visual_fitness:
                    self._init_fitness_fbo()
            except Exception as e:
                self._logger.warning(f"Fitness FBO init skipped: {e}")

            try:
                frag = self.shader_factory.generate(self.genome)
                self._recompile_fragment_shader(frag, self.genome)
                self._logger.info("Initial shader compiled successfully")
            except Exception as e:
                self._logger.error(f"Failed to compile initial shader: {e}")
                raise

            try:
                if self.ecosystem is not None:
                    self._init_resource_texture()
            except Exception as e:
                self._logger.warning(f"Resource texture init skipped: {e}")

            self.gl_ready = True

            if self._tick_id is None:
                self._tick_id = self.add_tick_callback(self._on_tick)
                self._logger.debug("Tick callback registered")

            self.queue_render()
        
        except Exception as e:
            self._logger.critical(f"Realization failed: {e}")
            self.gl_ready = False

    def _on_tick(self, widget: Gtk.GLArea, frame_clock) -> bool:
        """Tick callback to request continuous redraws.
        
        Without this GTK only repaints on resize/realize, freezing the output.
        
        Args:
            widget: The GLArea widget
            frame_clock: Frame clock (unused)
            
        Returns:
            True to continue receiving ticks
        """
        self.queue_render()
        return True

    # ------------------------------------------------------------------
    # Spatial ecosystem rendering (specs 46-60)
    # ------------------------------------------------------------------

    def _init_resource_texture(self) -> None:
        """Create the GPU resource-field texture (specs 48, 60)."""
        if self.ecosystem is None:
            return
        w = self.ecosystem.field.w
        h = self.ecosystem.field.h
        self._resource_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._resource_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._resource_tex_size = (w, h)

    def _upload_resource_texture(self) -> None:
        if self._resource_tex is None or self.ecosystem is None:
            return
        data = array.array('f', self.ecosystem.field.rgba_array())
        w, h = self._resource_tex_size
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._resource_tex)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                           GL.GL_RGBA, GL.GL_FLOAT, data)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _spatialize_shader(self, src: str) -> str:
        """Inject territory + species uniforms into a generated shader.

        ``sample_uv`` is remapped to global world coordinates so feedback
        trails persist across the whole world, while the organism's own pattern
        still uses its local ``v_uv``.
        """
        if "u_territory_min" in src:
            return src
        decl = (
            "\nuniform vec2 u_territory_min;\n"
            "uniform vec2 u_territory_size;\n"
            "uniform float u_org_energy;\n"
            "uniform vec3 u_org_tint;\n"
            "uniform sampler2D u_resource_tex;\n"
        )
        src = src.replace("void main() {", decl + "void main() {", 1)
        src = src.replace(
            "vec2 sample_uv = v_uv;",
            "vec2 sample_uv = u_territory_min + v_uv * u_territory_size;",
            1,
        )
        src = src.replace(
            "fragColor = vec4(col, 1.0);",
            "vec4 rfield = texture(u_resource_tex, u_territory_min + v_uv * u_territory_size);\n"
            "col += rfield.rgb * 0.15;\n"
            "col *= mix(0.2, 1.0, clamp(u_org_energy, 0.0, 1.0));\n"
            "col = mix(col, col * (0.35 + 0.65 * u_org_tint), 0.55);\n"
            "fragColor = vec4(col, 1.0);",
            1,
        )
        return src

    def _get_eco_program(self, org: "SpatialOrganism") -> int:
        key = "eco:" + self.shader_cache.key(org.genome)
        prog = self._eco_programs.get(key)
        if prog is None:
            frag = self.shader_factory.generate(org.genome)
            frag = self._spatialize_shader(frag)
            prog = self._compile_program(self.VERTEX_SHADER, frag)
            self._eco_programs[key] = prog
        if self.ecosystem is not None:
            self._eco_used[key] = self.ecosystem.step_count
        return prog

    def _bind_eco_program(self, program: int) -> Dict[str, int]:
        if program == self._cur_eco_prog and program in self._eco_loc:
            return self._eco_loc[program]
        GL.glUseProgram(program)
        self._cur_eco_prog = program
        loc = {
            "time": GL.glGetUniformLocation(program, "u_time"),
            "res": GL.glGetUniformLocation(program, "u_resolution"),
            "bass": GL.glGetUniformLocation(program, "bass"),
            "mid": GL.glGetUniformLocation(program, "mid"),
            "treb": GL.glGetUniformLocation(program, "treble"),
            "prev": GL.glGetUniformLocation(program, "previous_frame"),
            "tmin": GL.glGetUniformLocation(program, "u_territory_min"),
            "tsize": GL.glGetUniformLocation(program, "u_territory_size"),
            "energy": GL.glGetUniformLocation(program, "u_org_energy"),
            "tint": GL.glGetUniformLocation(program, "u_org_tint"),
            "restex": GL.glGetUniformLocation(program, "u_resource_tex"),
        }
        self._eco_loc[program] = loc
        return loc

    def _draw_organism(self, org: "SpatialOrganism", program: int, prev_tex: int,
                       W: int, H: int, rect, t: float, bass: float, mid: float,
                       treb: float) -> None:
        loc = self._bind_eco_program(program)
        x0, y0, w, h = rect
        iw = max(1, int(w))
        ih = max(1, int(h))
        GL.glViewport(int(x0), int(y0), iw, ih)
        GL.glScissor(int(x0), int(y0), iw, ih)

        if loc["time"] >= 0:
            GL.glUniform1f(loc["time"], float(t))
        if loc["res"] >= 0:
            GL.glUniform2f(loc["res"], float(w), float(h))
        if loc["bass"] >= 0:
            GL.glUniform1f(loc["bass"], float(bass))
        if loc["mid"] >= 0:
            GL.glUniform1f(loc["mid"], float(mid))
        if loc["treb"] >= 0:
            GL.glUniform1f(loc["treb"], float(treb))

        gmin_x = x0 / W
        gmin_y = y0 / H
        gsize_x = w / W
        gsize_y = h / H
        if loc["tmin"] >= 0:
            GL.glUniform2f(loc["tmin"], gmin_x, gmin_y)
        if loc["tsize"] >= 0:
            GL.glUniform2f(loc["tsize"], gsize_x, gsize_y)

        en = max(0.0, min(1.0, org.energy / (self.ecosystem.max_energy if self.ecosystem else 3.0)))
        if loc["energy"] >= 0:
            GL.glUniform1f(loc["energy"], en)

        tint = (1.0, 1.0, 1.0)
        if self.ecosystem is not None and org.species in self.ecosystem.species:
            tint = self.ecosystem.species[org.species].color
        if loc["tint"] >= 0:
            GL.glUniform3f(loc["tint"], float(tint[0]), float(tint[1]), float(tint[2]))

        if loc["restex"] >= 0 and self._resource_tex is not None:
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._resource_tex)
            GL.glUniform1i(loc["restex"], 1)
        if loc["prev"] >= 0:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, prev_tex)
            GL.glUniform1i(loc["prev"], 0)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)

    def _eco_territories(self, orgs: "list", W: int, H: int) -> "list":
        """Compute pixel territory rectangles with competition (spec 47)."""
        min_dim = float(min(W, H))
        raw = []
        for o in orgs:
            en = max(0.0, min(1.0, o.energy / 3.0))
            cr = o.radius * (0.4 + 0.6 * en)
            cx = o.pos[0] * W
            cy = o.pos[1] * H
            half = cr * min_dim
            raw.append([cx, cy, cr, half, en])
        n = len(raw)
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(raw[i][0] - raw[j][0], raw[i][1] - raw[j][1])
                min_d = raw[i][2] + raw[j][2]
                if 1e-4 < d < min_d:
                    weak, strong = (i, j) if raw[i][4] <= raw[j][4] else (j, i)
                    new_r = max(0.005, min_d - raw[strong][2] - 0.002)
                    raw[weak][2] = new_r
                    raw[weak][3] = new_r * min_dim
        rects = []
        for cx, cy, _cr, half, _en in raw:
            x0 = max(0.0, min(float(W) - 1.0, cx - half))
            y0 = max(0.0, min(float(H) - 1.0, cy - half))
            w = max(1.0, min(float(W) - x0, 2.0 * half))
            h = max(1.0, min(float(H) - y0, 2.0 * half))
            rects.append((x0, y0, w, h))
        return rects

    def _prune_eco_programs(self) -> None:
        if self.ecosystem is None:
            return
        cur = self.ecosystem.step_count
        for key in list(self._eco_programs.keys()):
            if cur - self._eco_used.get(key, -9999) > 180:
                prog = self._eco_programs.pop(key)
                if prog not in self._eco_programs.values():
                    try:
                        GL.glDeleteProgram(prog)
                    except Exception:
                        pass
                self._eco_loc.pop(prog, None)
                self._eco_used.pop(key, None)

    def _render_spatial(self, dt: float, W: int, H: int) -> bool:
        """Render the spatial ecosystem as territorial per-organism passes."""
        eco = self.ecosystem
        if eco is None:
            return False

        eco.step(dt, self.audio.audio, self._current_music_dna)
        self._upload_resource_texture()

        bass, mid, treb, beat = self.audio.uvec()
        t = time.time() - self._start_time

        self.feedback.resize(W, H)

        orgs = eco.alive_organisms()
        rects = self._eco_territories(orgs, W, H)

        GL.glBindVertexArray(self._vao)

        # Feedback pass (accumulate world trails).
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.feedback.dst_fbo())
        GL.glEnable(GL.GL_SCISSOR_TEST)
        prev_tex = self.feedback.src_tex()
        for org, rect in zip(orgs, rects):
            prog = self._get_eco_program(org)
            self._draw_organism(org, prog, prev_tex, W, H, rect, t, bass, mid, treb)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        self.feedback.swap()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # Screen pass.
        default_fbo = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, default_fbo)
        GL.glViewport(0, 0, W, H)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        prev_tex = self.feedback.src_tex()
        for org, rect in zip(orgs, rects):
            prog = self._get_eco_program(org)
            self._draw_organism(org, prog, prev_tex, W, H, rect, t, bass, mid, treb)
        GL.glDisable(GL.GL_SCISSOR_TEST)

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        self._cur_eco_prog = None
        self._prune_eco_programs()
        return True

    def toggle_spatial_ecosystem(self) -> None:
        """Toggle the spatial ecosystem renderer on/off."""
        if self.ecosystem is None and not self._config.enable_spatial_ecosystem:
            try:
                self.ecosystem = SpatialEcosystem(self._config)
                self._config.enable_spatial_ecosystem = True
            except Exception as e:
                self._logger.error(f"Could not start spatial ecosystem: {e}")
                return
        if self._spatial_active is False and self.gl_ready and self._resource_tex is None:
            try:
                self._init_resource_texture()
            except Exception as e:
                self._logger.warning(f"Resource texture lazy-init skipped: {e}")
        self._spatial_active = not self._spatial_active
        self._logger.info(f"Spatial ecosystem {'on' if self._spatial_active else 'off'}")
        self.queue_render()

    def on_unrealize(self, area: Gtk.GLArea) -> None:
        """Handle widget unrealization and GL cleanup.
        
        Args:
            area: The GLArea widget
        """
        try:
            if self._tick_id is not None:
                self.remove_tick_callback(self._tick_id)
                self._tick_id = None
        except Exception as e:
            self._logger.warning(f"Error removing tick callback: {e}")

        # Clean up GL resources
        try:
            for attr in ("_program", "_vbo", "_vao", "_fitness_fbo", "_fitness_tex"):
                obj = getattr(self, attr, None)
                if obj:
                    try:
                        if attr == "_program":
                            GL.glDeleteProgram(obj)
                        elif attr == "_vbo":
                            GL.glDeleteBuffers(1, [obj])
                        elif attr == "_vao":
                            GL.glDeleteVertexArrays(1, [obj])
                        elif attr in ("_fitness_fbo",):
                            GL.glDeleteFramebuffers(1, [obj])
                        elif attr in ("_fitness_tex",):
                            GL.glDeleteTextures(1, [obj])
                    except Exception as e:
                        self._logger.warning(f"Error deleting {attr}: {e}")
                    finally:
                        setattr(self, attr, None)
            
            # Clean up feedback buffer
            if hasattr(self, "feedback") and self.feedback:
                self.feedback.cleanup()

            # Clean up spatial ecosystem GPU resources (specs 46-60).
            for prog in getattr(self, "_eco_programs", {}).values():
                try:
                    GL.glDeleteProgram(prog)
                except Exception:
                    pass
            self._eco_programs = {}
            self._eco_loc = {}
            self._eco_used = {}
            self._cur_eco_prog = None
            if getattr(self, "_resource_tex", None):
                try:
                    GL.glDeleteTextures(1, [self._resource_tex])
                except Exception:
                    pass
                self._resource_tex = None

            # Clean up cached compiled programs (clear() queues via on_evict).
            if hasattr(self, "shader_cache") and self.shader_cache:
                self.shader_cache.clear()
                self._program = None

            # Flush any queued deletions inside the current GL context.
            while self.pending_delete:
                prog = self.pending_delete.pop(0)
                try:
                    GL.glDeleteProgram(prog)
                except Exception as e:
                    self._logger.debug(f"Failed to delete queued GL program {prog}: {e}")

            self._logger.info("GL cleanup complete")
        
        except Exception as e:
            self._logger.error(f"Unrealize error: {e}")
        
        self.gl_ready = False

    def _create_fullscreen_quad(self) -> None:
        """Create a fullscreen quad for rendering.
        
        Raises:
            RuntimeError: If numpy is not available or GL operations fail
        """
        if np is None:
            raise RuntimeError("EvoRenderer requires numpy for the fullscreen quad")

        data = np.asarray(FULLSCREEN_QUAD_VERTICES, dtype=np.float32)

        try:
            self._vao = GL.glGenVertexArrays(1)
            self._vbo = GL.glGenBuffers(1)

            GL.glBindVertexArray(self._vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, len(data) * 4, data, GL.GL_STATIC_DRAW)

            # Vertex layout: position (2 floats) + texcoord (2 floats)
            stride = QUAD_STRIDE_BYTES
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(0))
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(8))

            GL.glBindVertexArray(0)
            self._logger.debug(f"Fullscreen quad created: VAO={self._vao}, VBO={self._vbo}")
        
        except Exception as e:
            self._logger.error(f"Failed to create fullscreen quad: {e}")
            raise

    def _compile_program(self, vs_source: str, fs_source: str) -> int:
        """Compile and link a complete GL program.
        
        Args:
            vs_source: Vertex shader source code
            fs_source: Fragment shader source code
            
        Returns:
            Program object ID
            
        Raises:
            RuntimeError: If compilation or linking fails
        """
        try:
            vs = self._compile_shader(GL.GL_VERTEX_SHADER, vs_source)
            fs = self._compile_shader(GL.GL_FRAGMENT_SHADER, fs_source)

            prog = GL.glCreateProgram()
            GL.glAttachShader(prog, vs)
            GL.glAttachShader(prog, fs)
            GL.glLinkProgram(prog)

            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)

            if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
                err = GL.glGetProgramInfoLog(prog)
                if isinstance(err, bytes):
                    err = err.decode("utf-8", errors="ignore")
                raise RuntimeError(f"Program link failed: {err}")

            return int(prog)
        
        except Exception as e:
            self._logger.error(f"Program compilation failed: {e}")
            raise

    def _compile_shader(self, shader_type: int, source: str) -> int:
        """Compile a single shader.
        
        Args:
            shader_type: GL.GL_VERTEX_SHADER or GL.GL_FRAGMENT_SHADER
            source: Shader source code
            
        Returns:
            Shader object ID
            
        Raises:
            RuntimeError: If compilation fails
        """
        try:
            sh = GL.glCreateShader(shader_type)
            GL.glShaderSource(sh, source)
            GL.glCompileShader(sh)
            
            if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
                err = GL.glGetShaderInfoLog(sh)
                if isinstance(err, bytes):
                    err = err.decode("utf-8", errors="ignore")
                raise RuntimeError(f"Shader compile failed: {err}")
            
            return int(sh)
        
        except Exception as e:
            self._logger.error(f"Shader compilation error: {e}")
            raise

    def _recompile_fragment_shader(self, new_frag_source: str, genome: Any = None) -> None:
        """Recompile fragment shader and refresh uniform locations.

        Args:
            new_frag_source: New fragment shader source code.
            genome: Optional genome to associate with the compiled program in
                the shader cache. When provided the program is cached so future
                scoring of an identical genome reuses it.

        Raises:
            RuntimeError: If compilation fails
        """
        try:
            new_prog = self._compile_program(self.VERTEX_SHADER, new_frag_source)

            # Never delete a program still referenced by the cache (it may be
            # bound by another organism's scored program).
            if self._program is not None and self._program not in self.shader_cache.programs():
                GL.glDeleteProgram(self._program)

            self._set_program(new_prog)

            if genome is not None:
                self.shader_cache.store(self.shader_cache.key(genome), new_prog)

            self._logger.debug(f"Fragment shader recompiled: program={self._program}")

        except Exception as e:
            self._logger.error(f"Fragment shader recompilation failed: {e}")
            raise

    def _get_program(self, genome: Any) -> Optional[int]:
        """Return a compiled program for ``genome``, compiling + caching on
        first use. Falls back to the current program if compilation fails.
        """
        key = self.shader_cache.key(genome)
        program = self.shader_cache.get(key)
        if program is not None:
            return program
        try:
            frag = self.shader_factory.generate(genome)
            program = self._compile_program(self.VERTEX_SHADER, frag)
        except Exception as e:
            self._logger.error(
                f"Shader compile failed for genome {getattr(genome, 'lineage', '?')}: {e}"
            )
            return self._program
        self.shader_cache.store(key, program)
        return program

    def _set_program(self, program: Optional[int]) -> None:
        """Bind ``program`` as the active program and refresh uniform locations.

        Programs are cached so the GL program object is reused; only the
        uniform-location lookups need refreshing when switching programs. The
        active program is "protected" so the cache never evicts/deletes it
        while it is bound.
        """
        if program is None or program == self._program:
            return
        self._program = int(program)
        self.shader_cache.protect(self._program)

        self._loc_time = GL.glGetUniformLocation(self._program, "u_time")
        self._loc_res = GL.glGetUniformLocation(self._program, "u_resolution")

        # ModularShaderFactory uniforms
        self._loc_bass = GL.glGetUniformLocation(self._program, "bass")
        self._loc_mid = GL.glGetUniformLocation(self._program, "mid")
        self._loc_treble = GL.glGetUniformLocation(self._program, "treble")
        self._loc_previous_frame = GL.glGetUniformLocation(self._program, "previous_frame")

        # Legacy uniforms (may not exist in modular shaders)
        self._loc_audio = GL.glGetUniformLocation(self._program, "u_audio")
        self._loc_prev = GL.glGetUniformLocation(self._program, "u_prev_frame")
        self._loc_fb_alpha = GL.glGetUniformLocation(self._program, "u_feedback_alpha")
        self._loc_brightness = GL.glGetUniformLocation(self._program, "u_brightness")

        # Music DNA uniforms (optional)
        self._loc_md_bpm = GL.glGetUniformLocation(self._program, "md_bpm")
        self._loc_md_energy = GL.glGetUniformLocation(self._program, "md_energy")
        self._loc_md_danceability = GL.glGetUniformLocation(self._program, "md_danceability")
        self._loc_md_beat_strength = GL.glGetUniformLocation(self._program, "md_beat_strength")
        self._loc_md_spectral_centroid = GL.glGetUniformLocation(self._program, "md_spectral_centroid")
        self._loc_md_harmonic_complexity = GL.glGetUniformLocation(self._program, "md_harmonic_complexity")
        self._loc_md_dynamic_range = GL.glGetUniformLocation(self._program, "md_dynamic_range")
        self._loc_md_key = GL.glGetUniformLocation(self._program, "md_key")
        self._loc_md_mood = GL.glGetUniformLocation(self._program, "md_mood")

    def _delete_program(self, program: Optional[int]) -> None:
        """Queue a compiled GL program for deletion during the next render.

        Used as the :class:`ShaderCache` ``on_evict`` callback. Deferring
        deletion avoids ``GLError: invalid operation`` when the cache evicts
        outside an active GTK GLArea context.
        """
        if not program:
            return
        self.pending_delete.append(int(program))

    def _apply_render_uniforms(self, *, bass: float, mid: float, treb: float,
                               t: float, device_w: int, device_h: int,
                               feedback_res: int, previous_frame_tex: int) -> None:
        """Apply the uniforms shared by the feedback and screen passes."""
        if self._program is None:
            return

        if self._loc_res >= 0:
            GL.glUniform2f(self._loc_res, float(device_w), float(device_h))

        if self._loc_time >= 0:
            GL.glUniform1f(self._loc_time, float(t))

        if self._loc_bass >= 0:
            GL.glUniform1f(self._loc_bass, float(bass))
        if self._loc_mid >= 0:
            GL.glUniform1f(self._loc_mid, float(mid))
        if self._loc_treble >= 0:
            GL.glUniform1f(self._loc_treble, float(treb))

        if self._loc_fb_alpha >= 0:
            GL.glUniform1f(self._loc_fb_alpha, float(self._config.feedback_alpha))
        if self._loc_brightness >= 0:
            GL.glUniform1f(self._loc_brightness, float(self._config.brightness_base))

        if self._loc_previous_frame >= 0:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, previous_frame_tex)
            GL.glUniform1i(self._loc_previous_frame, 0)

        # Music DNA uniforms
        self._apply_music_dna_uniforms()

    def _apply_music_dna_uniforms(self) -> None:
        """Apply optional MusicDNA uniforms to the current shader program."""
        if self._program is None or not hasattr(self, "_current_music_dna"):
            return
        dna = self._current_music_dna
        if self._loc_md_bpm >= 0:
            GL.glUniform1f(self._loc_md_bpm, float(dna.bpm))
        if self._loc_md_energy >= 0:
            GL.glUniform1f(self._loc_md_energy, float(dna.energy))
        if self._loc_md_danceability >= 0:
            GL.glUniform1f(self._loc_md_danceability, float(dna.danceability))
        if self._loc_md_beat_strength >= 0:
            GL.glUniform1f(self._loc_md_beat_strength, float(dna.beat_strength))
        if self._loc_md_spectral_centroid >= 0:
            GL.glUniform1f(self._loc_md_spectral_centroid, float(dna.spectral_centroid))
        if self._loc_md_harmonic_complexity >= 0:
            GL.glUniform1f(self._loc_md_harmonic_complexity, float(dna.harmonic_complexity))
        if self._loc_md_dynamic_range >= 0:
            GL.glUniform1f(self._loc_md_dynamic_range, float(dna.dynamic_range))
        if self._loc_md_key >= 0:
            GL.glUniform1f(self._loc_md_key, float(_key_to_float(dna.key)))
        if self._loc_md_mood >= 0:
            GL.glUniform1f(self._loc_md_mood, float(_mood_to_float(dna.mood)))

    def on_render(self, area: Gtk.GLArea, ctx) -> bool:
        """Main render callback."""
        if not self._program or not self.gl_ready:
            return False

        # Process queued GL program deletions inside the active context.
        while self.pending_delete:
            prog = self.pending_delete.pop(0)
            try:
                GL.glDeleteProgram(prog)
            except Exception as e:
                self._logger.debug(f"Failed to delete queued GL program {prog}: {e}")

        # Spatial ecosystem (specs 46-60): CPU-supervised simulation rendered
        # as territorial per-organism passes on the GPU.
        if self._spatial_active and self.ecosystem is not None and self._vao is not None:
            now = time.time()
            dt = min(0.1, max(1e-3, now - self._last_frame_ts))
            self._last_frame_ts = now
            scale = float(area.get_scale_factor())
            aw = int(area.get_allocated_width() * scale)
            ah = int(area.get_allocated_height() * scale)
            if aw > 0 and ah > 0:
                self._last_glarea_w = aw
                self._last_glarea_h = ah
            else:
                aw = getattr(self, "_last_glarea_w", int(1024 * scale))
                ah = getattr(self, "_last_glarea_h", int(768 * scale))
            try:
                if self._render_spatial(dt, aw, ah):
                    return True
            except Exception as e:
                self._logger.error(f"Spatial render error: {e}")
                self._spatial_active = False

        try:
            default_fbo = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)

            # Ecosystem memory: update audio context
            if self._config.enable_ecosystem_memory:
                self._update_music_signature()
                self._maybe_store_music_pattern()
                self._maybe_evolve_world()
                if self._check_dream_mode():
                    self.dream_once()
                self._maybe_revive_species()

            # Evolution step
            if self.evolve_timer.due():
                self.evolve()
                self.evolve_timer.reset()

            bass, mid, treb, beat = self.audio.uvec()

            # Beat-driven evolution events
            self._maybe_beat_evolution(bass, mid, treb, beat)

            t = time.time() - self._start_time

            # Match the feedback buffer to the actual viewport so the feedback
            # effect stays sharp and resolution-independent.
            scale = float(area.get_scale_factor())
            allocated_w = int(area.get_allocated_width() * scale)
            allocated_h = int(area.get_allocated_height() * scale)
            if allocated_w > 0 and allocated_h > 0:
                self._last_glarea_w = allocated_w
                self._last_glarea_h = allocated_h
            else:
                allocated_w = getattr(self, "_last_glarea_w", int(1024 * scale))
                allocated_h = getattr(self, "_last_glarea_h", int(768 * scale))
            self.feedback.resize(allocated_w, allocated_h)

            # Population visualization mode: draw a grid of every organism.
            if self._population_view and self._config.population_size > 1:
                return self._render_population_grid(
                    t, bass, mid, treb, allocated_w, allocated_h
                )

            # Always render the current champion, regardless of any program
            # left bound by the population grid view.
            self._set_program(self._get_program(self.genome))
            GL.glUseProgram(self._program)
            GL.glBindVertexArray(self._vao)

            # Feedback pass (matched to viewport resolution)
            self._apply_render_uniforms(
                bass=bass,
                mid=mid,
                treb=treb,
                t=t,
                device_w=self.feedback.width,
                device_h=self.feedback.height,
                feedback_res=self.feedback.res,
                previous_frame_tex=self.feedback.src_tex(),
            )

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.feedback.dst_fbo())
            GL.glViewport(0, 0, self.feedback.width, self.feedback.height)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)

            self.feedback.swap()

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, default_fbo)

            # Screen pass
            GL.glViewport(0, 0, allocated_w, allocated_h)
            w = allocated_w
            h = allocated_h

            self._apply_render_uniforms(
                bass=bass,
                mid=mid,
                treb=treb,
                t=t,
                device_w=w,
                device_h=h,
                feedback_res=self.feedback.res,
                previous_frame_tex=self.feedback.src_tex(),
            )

            GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)
            GL.glFlush()

            GL.glBindVertexArray(0)
            GL.glUseProgram(0)

            # Visual fitness readback
            if self._config.enable_visual_fitness and self._fitness_fbo is not None:
                try:
                    self._capture_and_score()
                except Exception:
                    pass

            self._memory_watchdog()

            return True

        except Exception as e:
            self._logger.error(f"Render error: {e}")
            return False

    def _render_population_grid(self, t: float, bass: float, mid: float, treb: float,
                                w: int, h: int) -> bool:
        """Draw every organism in the population into a tiled grid.

        Used by the population visualization mode so the whole ecosystem is
        visible at once. Click a tile (see ``_on_population_click``) to promote
        that organism to the champion.
        """
        organisms = self.population.organisms
        n = max(1, len(organisms))
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        tile_w = w / cols
        tile_h = h / rows

        GL.glUseProgram(self._program)
        GL.glBindVertexArray(self._vao)

        for idx, org in enumerate(organisms):
            r = idx // cols
            c = idx % cols
            x0 = c * tile_w
            y0 = h - (r + 1) * tile_h  # GL viewport origin is bottom-left
            GL.glViewport(int(x0), int(y0), int(tile_w), int(tile_h))
            self._set_program(self._get_program(org))
            self._apply_render_uniforms(
                bass=bass,
                mid=mid,
                treb=treb,
                t=t,
                device_w=int(tile_w),
                device_h=int(tile_h),
                feedback_res=self.feedback.res,
                previous_frame_tex=self.feedback.src_tex(),
            )
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)

        GL.glFlush()
        GL.glViewport(0, 0, w, h)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        return True

    def _memory_watchdog(self) -> None:
        """Log a warning when RSS exceeds the configured threshold.

        Throttled to once every few seconds so it costs almost nothing in the
        render loop. Tries ``psutil`` first, then falls back to ``/proc/self``
        and ``resource``, so no hard new dependency is required. This is the
        diagnostic the OOM post-mortem asked for: a tripwire that fires *before*
        the kernel OOM killer does.
        """
        now = time.time()
        if now - getattr(self, "_watchdog_ts", 0.0) < 5.0:
            return
        self._watchdog_ts = now

        try:
            rss_mb = self._current_rss_mb()
        except Exception:
            return

        threshold = float(self._config.memory_watchdog_mb)
        if rss_mb > threshold:
            self._logger.warning(
                "MEMORY WATCHDOG: RSS %.0f MB exceeds %.0f MB threshold "
                "(gen=%d, pop=%d, cached_programs=%d, lineage_records=%d)",
                rss_mb, threshold, self._generation,
                len(self.population.organisms), len(self.shader_cache.cache),
                len(self.memory_dna._records),
            )

    def _current_rss_mb(self) -> float:
        """Best-effort resident set size in MB, with progressive fallbacks."""
        try:
            import psutil  # optional, nicer interface  # type: ignore[reportMissingModuleSource]
            return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            pass
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # value is in kB
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
        try:
            import resource
            # ru_maxrss is peak RSS in kB on Linux
            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        except Exception:
            return 0.0

    def toggle_population_view(self) -> None:
        """Toggle the population visualization grid on/off."""
        self._population_view = not self._population_view
        self._logger.info(f"Population view {'on' if self._population_view else 'off'}")
        self.queue_render()

    def _on_population_click(self, gesture, n_press, x, y) -> None:
        """Promote the clicked organism tile to the champion in population mode."""
        if not self._population_view:
            return
        organisms = self.population.organisms
        if not organisms:
            return
        scale = float(self.get_scale_factor())
        px = x * scale
        py = y * scale
        w = getattr(self, "_last_glarea_w", int(px))
        h = getattr(self, "_last_glarea_h", int(py))
        n = len(organisms)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        tile_w = w / cols
        tile_h = h / rows
        gy = h - py  # GL y is bottom-up
        r = int(gy // tile_h) if tile_h > 0 else 0
        c = int(px // tile_w) if tile_w > 0 else 0
        idx = r * cols + c
        if 0 <= idx < n:
            chosen = organisms[idx]
            self._set_program(self._get_program(chosen))
            self.genome = chosen
            self._logger.info(f"Selected organism {idx} ({chosen.lineage}) as champion")
            if self._config.enable_lineage:
                self.memory_dna.record(self.genome)
            self.queue_render()

    def _on_key_pressed(self, controller, keyval, keycode, state) -> bool:
        """Keyboard shortcut: 'p' toggles the population grid; 'e' toggles the
        spatial ecosystem; 't' toggles territory debug overlay."""
        from gi.repository import Gdk

        if keyval == Gdk.KEY_p:
            self.toggle_population_view()
            return True
        if keyval == Gdk.KEY_e:
            self.toggle_spatial_ecosystem()
            return True
        if keyval == Gdk.KEY_t:
            self._config.show_territories = not self._config.show_territories
            self._logger.info(f"Territory overlay {'on' if self._config.show_territories else 'off'}")
            return True
        return False

    def _maybe_beat_evolution(self, bass: float, mid: float, treb: float, beat: float) -> None:
        """Trigger a lightweight mutation on strong beats."""
        if beat < 0.7:
            return
        now = time.time()
        if now - self._last_beat_evolution_ts < self._beat_evolution_cooldown:
            return
        self._last_beat_evolution_ts = now
        try:
            self.genome.mutate(strength=0.05)
            frag = self.shader_factory.generate(self.genome)
            self._recompile_fragment_shader(frag, self.genome)
            self._logger.debug("Beat-driven mutation applied")
        except Exception as e:
            self._logger.debug(f"Beat-driven mutation failed: {e}")


def _key_to_float(key: str) -> float:
    mapping = {"C": 0.0, "C#": 0.083, "D": 0.167, "D#": 0.25, "E": 0.333,
               "F": 0.417, "F#": 0.5, "G": 0.583, "G#": 0.667, "A": 0.75,
               "A#": 0.833, "B": 0.917}
    return mapping.get(key, 0.0)


def _mood_to_float(mood: str) -> float:
    mapping = {"calm": 0.0, "melancholic": 0.2, "dark": 0.4,
               "energetic": 0.6, "euphoric": 0.8, "tense": 1.0}
    return mapping.get(mood, 0.0)


if __name__ == "__main__":
    import gi

    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk

    class App(Gtk.Application):
        """Main application class."""
        
        def __init__(self):
            super().__init__(application_id="org.blackboxai.evo")
            self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        def do_activate(self):
            """Activate the application."""
            try:
                preset_dir = Path.home() / ".evo_renderer" / "presets"
                
                config = EvoConfig(
                    feedback_alpha=0.85,
                    feedback_res=512,
                    mutation_interval_secs=10.0,
                    mutation_strength=0.1,
                    enable_logging=True,
                    preset_dir=preset_dir,
                    debug=False,
                    enable_spatial_ecosystem=True,
                    spatial_population_size=18,
                )
                
                win = Gtk.ApplicationWindow(application=self)
                win.set_default_size(1200, 800)
                win.set_title("EvoRenderer - Real-time Evolutionary Visualization")

                area = EvoRenderer(config)
                area.set_hexpand(True)
                area.set_vexpand(True)
                area.set_halign(Gtk.Align.FILL)
                area.set_valign(Gtk.Align.FILL)
                
                box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
                box.set_hexpand(True)
                box.set_vexpand(True)
                box.append(area)
                
                win.set_child(box)
                
                win.present()
                
                self._logger.info("Application started successfully")
            
            except Exception as e:
                self._logger.error(f"Failed to activate application: {e}")
                raise

    app = App()
    exit_code = app.run(None)
    logger.info(f"Application exited with code: {exit_code}")
    exit(exit_code)
