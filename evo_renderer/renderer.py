from __future__ import annotations

import time
from typing import Any

import OpenGL.GL as GL

from .audio_state import AudioState
from .feedback import FeedbackBuffer
from .evolution import EvolutionEngine
from evo.modular_genome import ModularGenome
from evo.modular_shader_factory import ModularShaderFactory

from engine.gpu_profiler import GPUProfiler
from engine.optimizer import RendererOptimizer
from engine.renderer_genome import RendererGenome


class EvoRenderer:
    """GPU organism renderer (feedback ping-pong + genome-driven shader)."""

    VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

out vec2 v_uv;

void main() {
    v_uv = texcoord;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

    def __init__(self, feedback_res: int = 512, mutation_interval_secs: float = 15.0):
        self.genome = ModularGenome()
        self.shader_factory = ModularShaderFactory()

        # Stage 10: renderer genome + cost proxy
        self.base_feedback_res = int(feedback_res)
        self.renderer_genome = RendererGenome.random_initial(base_feedback_res=self.base_feedback_res)
        self.gpu_profiler = GPUProfiler()
        self.renderer_optimizer = RendererOptimizer()

        self.feedback = FeedbackBuffer(self.base_feedback_res)

        self.shader_program: int | None = None
        self._vao: int | None = None
        self._vbo: int | None = None

        self._loc_time = -1
        self._loc_res = -1
        self._loc_bass = -1
        self._loc_mid = -1
        self._loc_treble = -1
        self._loc_prev = -1

        # Stage 6 world-state uniforms (scalars summary MVP)
        self._loc_world_energy_mean = -1
        self._loc_world_energy_max = -1
        self._loc_world_entity_count = -1
        self._loc_world_age = -1

        # Stage 7 visual physics memory (audio -> persistent motion)
        self._loc_u_velocity = -1
        self._loc_u_impact = -1
        self._loc_u_turbulence = -1

        from .visual_physics import VisualPhysics

        self.visual_physics = VisualPhysics()


        self._last_dt_time = time.time()
        self.elapsed_time = 0.0
        self.last_audio_state: AudioState | None = None

        # CPU emergent physics world (driven by audio)
        from physics.ecosystem import Ecosystem

        self.world = Ecosystem(width=64, height=64)

        self.evolution = EvolutionEngine(self, interval_secs=mutation_interval_secs)

        # Procedural biology: metrics of the current champion's body graph.
        # Set via set_organism_morphology(); baked into the shader so the
        # evolved geometry warps the visual field.
        self.morphology_metrics: dict = {}

    def init_gl(self) -> None:
        # Fullscreen quad
        self._create_fullscreen_quad()

        # Feedback FBO ping-pong
        self.feedback.init_gl()

        # Compile initial shader
        self.evolve_shader()

        if self.shader_program is None:
            raise RuntimeError("EvoRenderer shader_program is not ready")

        # Cache uniform locations
        self._loc_time = GL.glGetUniformLocation(self.shader_program, "u_time")
        self._loc_res = GL.glGetUniformLocation(self.shader_program, "u_resolution")
        self._loc_bass = GL.glGetUniformLocation(self.shader_program, "bass")
        self._loc_mid = GL.glGetUniformLocation(self.shader_program, "mid")
        self._loc_treble = GL.glGetUniformLocation(self.shader_program, "treble")
        self._loc_prev = GL.glGetUniformLocation(self.shader_program, "previous_frame")

        # Stage 6 world-state uniforms
        self._loc_world_energy_mean = GL.glGetUniformLocation(
            self.shader_program, "u_world_energy_mean"
        )
        self._loc_world_energy_max = GL.glGetUniformLocation(
            self.shader_program, "u_world_energy_max"
        )
        self._loc_world_entity_count = GL.glGetUniformLocation(
            self.shader_program, "u_world_entity_count"
        )
        self._loc_world_age = GL.glGetUniformLocation(self.shader_program, "u_world_age")

    def _create_fullscreen_quad(self) -> None:
        # position(x,y) + texcoord(u,v) interleaved
        vertices = [
            -1.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            -1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            0.0,
            1.0,
        ]

        import array

        data = array.array("f", vertices)

        self._vao = int(GL.glGenVertexArrays(1))
        self._vbo = int(GL.glGenBuffers(1))

        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(data) * 4, data, GL.GL_STATIC_DRAW)

        stride = 4 * 4
        # position at loc 0
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(0)
        )
        # texcoord at loc 1
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(8)
        )

        GL.glBindVertexArray(0)

    def evolve_shader(self) -> None:
        # ModularShaderFactory generates a complete GLSL fragment shader.
        if self.morphology_metrics:
            source = self.shader_factory.generate(
                self.genome, brightness=1.0, feedback_alpha=0.85,
                morphology=self.morphology_metrics,
            )
        else:
            source = self.shader_factory.generate(self.genome, brightness=1.0, feedback_alpha=0.85)

        new_program = self._compile_program(self.VERTEX_SHADER, source)

        # If compilation/link fails, exceptions should bubble up to the caller.
        # Stage 10 optimizer will rollback by restoring parent genome.

        # Delete old program if present
        if self.shader_program is not None:
            try:
                GL.glDeleteProgram(self.shader_program)
            except Exception:
                pass

        self.shader_program = new_program

    def _compile_program(self, vs_source: str, fs_source: str) -> int:
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
            GL.glDeleteProgram(prog)
            raise RuntimeError(f"EvoRenderer shader link failed: {err}")

        return int(prog)

    def _compile_shader(self, shader_type: int, source: str) -> int:
        sh = GL.glCreateShader(shader_type)
        GL.glShaderSource(sh, source)
        GL.glCompileShader(sh)
        if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
            err = GL.glGetShaderInfoLog(sh)
            if isinstance(err, bytes):
                err = err.decode("utf-8", errors="ignore")
            GL.glDeleteShader(sh)
            raise RuntimeError(f"EvoRenderer shader compile failed: {err}")
        return int(sh)

    def set_feedback_resolution_from_genome(self) -> None:
        """Apply Stage 10 dynamic feedback resolution.

        Recreates the ping-pong FBO textures.
        """
        try:
            scale = float(self.renderer_genome.pipeline.get("feedback_resolution", 1.0))
        except Exception:
            scale = 1.0

        scale = max(0.35, min(1.0, scale))
        new_res = int(max(64, round(self.base_feedback_res * scale)))
        if int(self.feedback.res) == new_res:
            return

        # Reallocate feedback buffers
        self.feedback = FeedbackBuffer(new_res)
        self.feedback.init_gl()

    def render(self, audio: AudioState | dict) -> None:
        if isinstance(audio, dict):
            audio_state = AudioState()
            audio_state.set_levels(audio)
            audio = audio_state

        # dt & evolution
        now = time.time()
        dt = now - self._last_dt_time
        self._last_dt_time = now
        self.elapsed_time += float(dt)

        # expose audio + time context for learning layer
        self.last_audio_state = audio

        # Stage 6 physics update (audio -> universe conditions)
        self.world.step(
            dt,
            audio_bass=float(audio.bass),
            audio_mid=float(audio.mid),
            audio_treble=float(audio.treble),
        )

        # Stage 7 visual physics memory (audio -> persistent motion)
        try:
            self.visual_physics.update(audio)
        except Exception:
            pass


        # Stage 10 can update resolution before drawing
        # (safe because it only reallocates if changed)
        try:
            self.set_feedback_resolution_from_genome()
        except Exception:
            pass

        # Stage 12: evolution returns dream results when music returns from silence
        dream_results = []
        try:
            dream_results = self.evolution.update(dt) or []
        except Exception:
            dream_results = []

        if dream_results:
            try:
                self._apply_dream_genomes(dream_results)
            except Exception:
                pass

        # Stage 11: researcher runs on its own cadence, independent of evolution.
        researcher = getattr(self.evolution, "researcher", None)
        if researcher is not None:
            try:
                researcher.update(dt=float(dt), audio=audio)
            except Exception:
                pass

        if self.shader_program is None:
            return

        GL.glUseProgram(self.shader_program)
        GL.glBindVertexArray(self._vao)

        # Render organism into feedback dst FBO
        self.feedback.bind_write_fbo()

        t = float(now)
        GL.glUniform1f(self._loc_time, t)
        GL.glUniform2f(self._loc_res, float(self.feedback.res), float(self.feedback.res))
        GL.glUniform1f(self._loc_bass, float(audio.bass))
        GL.glUniform1f(self._loc_mid, float(audio.mid))
        GL.glUniform1f(self._loc_treble, float(audio.treble))

        # Smoothed derived signals for the audio feature pipeline
        # (new shader uniforms added in evo_renderer/shader_factory.py)
        loc_energy = GL.glGetUniformLocation(self.shader_program, "u_energy")
        loc_beat = GL.glGetUniformLocation(self.shader_program, "u_beat")

        # Visual physics uniforms (optional; shader may omit them)
        if self._loc_u_velocity >= 0:
            GL.glUniform1f(self._loc_u_velocity, float(self.visual_physics.velocity))
        if self._loc_u_impact >= 0:
            GL.glUniform1f(self._loc_u_impact, float(self.visual_physics.impact))
        if self._loc_u_turbulence >= 0:
            GL.glUniform1f(self._loc_u_turbulence, float(self.visual_physics.turbulence))

        loc_u_bass = GL.glGetUniformLocation(self.shader_program, "u_bass")
        loc_u_mid = GL.glGetUniformLocation(self.shader_program, "u_mid")
        loc_u_treble = GL.glGetUniformLocation(self.shader_program, "u_treble")

        if loc_u_bass >= 0:
            GL.glUniform1f(loc_u_bass, float(getattr(audio, "bass", 0.0)))
        if loc_u_mid >= 0:
            GL.glUniform1f(loc_u_mid, float(getattr(audio, "mid", 0.0)))
        if loc_u_treble >= 0:
            GL.glUniform1f(loc_u_treble, float(getattr(audio, "treble", 0.0)))

        if loc_energy >= 0:
            GL.glUniform1f(loc_energy, float(getattr(audio, "energy", 0.0)))
        if loc_beat >= 0:
            GL.glUniform1f(loc_beat, float(getattr(audio, "beat", 0.0)))


        # Stage 6 world summaries -> shader
        sums = self.world.world_summaries()
        GL.glUniform1f(self._loc_world_energy_mean, float(sums["world_energy_mean"]))
        GL.glUniform1f(self._loc_world_energy_max, float(sums["world_energy_max"]))
        GL.glUniform1f(self._loc_world_entity_count, float(sums["world_entity_count"]))
        GL.glUniform1f(self._loc_world_age, float(sums["world_age"]))

        # previous_frame texture
        self.feedback.bind_previous(texture_unit=0)
        GL.glUniform1i(self._loc_prev, 0)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

        # New frame becomes previous frame
        self.feedback.swap()

        # leave state clean
        self.feedback.unbind()
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def _apply_dream_genomes(self, dreams: list[dict[str, Any]]) -> None:
        """Apply best dream genome when music returns after silence."""
        if not dreams:
            return
        best = dreams[0]
        if not best.get("genome"):
            return
        try:
            self.genome = best["genome"]
            self.evolve_shader()
        except Exception:
            pass

    def set_organism_morphology(self, metrics: dict) -> None:
        """Feed the evolved body's metrics into the shader pipeline.

        ``metrics`` is the dict returned by ``VisualOrganism.body_metrics()``
        (branching / symmetry / complexity / density / regeneration / ...).
        The values are baked into the next compiled shader so the *geometry of
        the grown organism* shapes the rendered visual.  Passing an empty dict
        restores the classic shader.
        """
        self.morphology_metrics = dict(metrics or {})
        try:
            self.evolve_shader()
        except Exception:
            # If the morphology shader fails to compile, fall back to classic.
            self.morphology_metrics = {}
            try:
                self.evolve_shader()
            except Exception:
                pass

    def human_feedback(self, sentiment: str) -> None:
        """Pass human feedback into the researcher (Stage 11) and identity (Stage 12).

        sentiment:
          "more_alive" -> boost growth, motion, complexity
          "calmer"     -> boost slow evolution, reduce chaos
          "more_chaos" -> boost noise, particles, disruption
        """
        evolution = getattr(self, "evolution", None)
        if evolution is None:
            return

        # Stage 11: researcher
        if getattr(evolution, "researcher", None) is not None:
            try:
                nudges = evolution.researcher.human_feedback(sentiment)
                if nudges and hasattr(evolution, "brain"):
                    if "slow_evolution" in nudges:
                        self.interval_secs = max(5.0, self.interval_secs * 0.8)
                        evolution.interval_secs = self.interval_secs
                        evolution.timer = 0.0
                    if "reduce_chaos" in nudges:
                        if hasattr(evolution.brain, "weights"):
                            for key in list(evolution.brain.weights.keys()):
                                if "noise" in key or "particle" in key:
                                    evolution.brain.weights[key] = max(0.1, evolution.brain.weights[key] * 0.5)
                    evolution.researcher.curiosity_boost()
            except Exception:
                pass

        # Stage 12: identity mood update
        identity = getattr(evolution, "identity", None)
        if identity is not None:
            try:
                identity.human_feedback(sentiment)
            except Exception:
                pass

