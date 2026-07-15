from __future__ import annotations

from .genome import VisualGenome
from .modules import MODULES


class ShaderFactory:
    def build(self, genome: VisualGenome) -> str:
        # Numeric constants injected as literals.
        warp_strength = float(genome.params.get("warp_strength", 0.5))
        feedback = float(genome.params.get("feedback", 0.85))

        # Colour genes (simple palette inputs)
        ca = float(genome.params.get("colour_a", 0.2))
        cb = float(genome.params.get("colour_b", 0.6))
        cc = float(genome.params.get("colour_c", 0.9))

        # Module DNA -> shader body
        body_parts: list[str] = []
        for module_name in genome.modules:
            mod = MODULES.get(module_name)
            if mod is None:
                continue
            body_parts.append(mod.code)

        body = "\n".join(body_parts)

        # The assembled shader keeps the same interface expected by EvoRenderer.
        # We also ensure the module snippets can modify `uv` safely.
        return f"""
#version 330

uniform float u_time;
uniform vec2 u_resolution;

uniform float u_bass;
uniform float u_mid;
uniform float u_treble;
uniform float u_energy;
uniform float u_beat;

// Audio-driven visual physics memory (persistent momentum/disturbance)
uniform float u_velocity;
uniform float u_impact;
uniform float u_turbulence;


uniform sampler2D previous_frame;


// Stage 6 world-state interface (scalars summary MVP)
uniform float u_world_energy_mean;
uniform float u_world_energy_max;
uniform float u_world_entity_count;
uniform float u_world_age;

out vec4 fragColor;

void main()
{{
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    uv -= 0.5;

    // Audio-driven momentum advection (persistent)
    uv += u_velocity * 0.01;

    float warp_strength = {warp_strength};

    float feedback = {feedback};

    // Bass drives warp intensity (smoothed continuous evolutionary force)
    float bassT = clamp(u_bass, 0.0, 1.0);

    // Replace direct “make plasma” coupling with field-conditioned motion.
    float field_drive = clamp(u_world_energy_mean * 0.6 + u_world_energy_max * 0.15, 0.0, 2.0);

    // Geometry/warp pulse
    float pulse = 1.0 + bassT * 0.25;
    uv *= vec2(pulse, 1.0 + bassT * 0.10);

    uv.x += sin(uv.y * 10.0 + u_time) * warp_strength * field_drive * (1.0 + bassT * 0.35);



    // --- creature anatomy (assembled modules) ---
    {body}

    // colour genes (assembled-from-modules are allowed to influence n/p,
    // but we provide robust fallbacks)
    float hue = ({ca} + {cb}*uv.x + {cc}*uv.y) + 0.15 * sin(u_time + uv.x*3.0);


    vec3 old = texture(previous_frame, uv + 0.5).rgb;

    // Use audio to modulate palette and feedback response.
    vec3 palette = vec3(
        0.5 + 0.5*sin(6.28318*(hue + 0.00)),
        0.5 + 0.5*sin(6.28318*(hue + 0.33)),
        0.5 + 0.5*sin(6.28318*(hue + 0.66))
    );

    // Audio-driven feedback decay: quiet -> long trails, loud -> shorter/violent.
    float fbAudio = clamp(u_energy, 0.0, 1.0);
    float feedbackAlpha = clamp({feedback} - fbAudio * 0.25, 0.0, 0.999);

    vec3 col = old * feedbackAlpha;

    // Colour driven by audio bands.
    float bassT = clamp(u_bass, 0.0, 1.0);
    float midT = clamp(u_mid, 0.0, 1.0);
    float trebT = clamp(u_treble, 0.0, 1.0);

    // palette intensity reacts to bass/mid, with energy bias.
    col += palette * (0.20 + 0.25*bassT + 0.15*midT + 0.10*fbAudio);

    // soft motion energy
    col += vec3(
        0.05*sin(u_time + uv.x*5.0 + trebT),
        0.05*sin(u_time + uv.y*7.0 + midT),
        0.05*sin(u_time + bassT*2.0)
    );

    // Beat impulse: physically bend image based on stored impact memory
    float beatForce = u_beat * 0.15 + u_impact * 0.02;
    uv.x += sin(uv.y * 20.0) * beatForce;
    uv.y += cos(uv.x * 20.0) * beatForce;

    // Beat shockwave: ring impulse around center
    float shock = sin(distance(uv, vec2(0.0)) * 30.0 - u_time * 10.0) * (0.20 * clamp(u_beat, 0.0, 1.0));
    col += shock;


    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);


}}
"""


