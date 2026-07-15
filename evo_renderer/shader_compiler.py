from __future__ import annotations


from .shader_dna import ShaderDNA


class ShaderCompiler:
    """Compile ShaderDNA into a full GLSL fragment shader.

    Safety requirements:
      - Deterministic codegen from safe IR (no free-form GLSL mutation)
      - Generated shader must define expected uniforms & varyings.
      - Keep variable names consistent.
    """

    VERSION = "#version 330 core"

    def compile(self, dna: ShaderDNA) -> str:
        body_parts: list[str] = []

        # Gene ops append to a `body_parts` which runs inside main().
        for gene in dna.operations:
            op = gene.get("op")
            if op == "noise":
                scale = float(gene.get("scale", 6.0))
                body_parts.append(
                    f"""
                    float n = sin(
                        uv.x*({scale}) +
                        uv.y*({scale*0.75:.4f}) +
                        u_time
                    );
                    uv += 0.01 * n;
                    """
                )

            elif op == "warp":
                amount = float(gene.get("amount", 0.2))
                body_parts.append(
                    f"""
                    uv += sin(uv.yx*12.0 + u_time) * {amount:.6f};
                    """
                )

            elif op == "distort":
                strength = float(gene.get("strength", 0.1))
                body_parts.append(
                    f"""
                    uv = mix(uv, vec2(uv.x, uv.y) + vec2(sin(uv.y*18.0+u_time), cos(uv.x*14.0-u_time))*{strength:.6f}, 0.7);
                    """
                )

            elif op == "fractal":
                it = int(gene.get("iter", 4))
                it = max(1, min(8, it))
                body_parts.append(
                    f"""
                    for(int i=0;i<{it};i++)
                    {{
                        uv = abs(uv) / max(dot(uv,uv), 1e-4) - 0.5;
                    }}
                    """
                )

            elif op == "feedback":
                decay = float(gene.get("decay", 0.9))
                decay = max(0.5, min(0.999, decay))
                body_parts.append(
                    f"""
                    vec3 oldc = texture(previous_frame, uv + 0.5).rgb;
                    colour += oldc * {decay:.6f};
                    """
                )

            elif op == "colour":
                mode = int(gene.get("mode", 0)) % 3
                if mode == 0:
                    body_parts.append(
                        """
                        vec3 palette = 0.5 + 0.5*sin(vec3(uv.x*3.0, uv.y*4.0, u_time*0.2) + u_time);
                        colour += palette * 0.35;
                        """
                    )
                elif mode == 1:
                    body_parts.append(
                        """
                        float l = length(uv);
                        vec3 palette = 0.5 + 0.5*cos(vec3(uv.x, uv.y, l)*5.0 + u_time);
                        colour += palette * 0.35;
                        """
                    )
                else:
                    body_parts.append(
                        """
                        float a = atan(uv.y, uv.x);
                        vec3 palette = vec3(
                            0.5 + 0.5*sin(6.28318*(a/3.14159) + u_time),
                            0.5 + 0.5*sin(6.28318*(a/2.71828) + u_time*1.3),
                            0.5 + 0.5*sin(6.28318*(a/1.61803) + u_time*0.7)
                        );
                        colour += palette * 0.35;
                        """
                    )

            else:
                # Unknown gene op: ignore (keeps compilation stable).
                continue

        body = "\n".join(body_parts)

        # Template shader: feedback is handled via `colour` accumulation.
        # We also include a bit of audio coupling by using bass/mid/treble.
        # Use a normal triple-quoted template and only inject `body`.
        return (
            """\
{version}


uniform float u_time;
uniform vec2 u_resolution;

uniform float bass;
uniform float mid;
uniform float treble;

uniform sampler2D previous_frame;

out vec4 fragColor;

void main()
{{
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    uv -= 0.5;

    vec3 colour = vec3(0.0);

    // Audio-driven base motion so population has a consistent evaluation surface.
    uv.x += sin(uv.y * 10.0 + u_time) * bass * 0.04;
    uv.y += cos(uv.x * 8.0 + u_time*0.7) * mid * 0.03;

    {body}

    // Final shaping: clamp and add a soft glow.
    float energy = clamp(dot(colour, vec3(0.3333)), 0.0, 1.0);
    colour += energy * (0.15 + 0.25*treble);
    colour = clamp(colour, 0.0, 1.0);

    fragColor = vec4(colour, 1.0);
}}
"""
            .format(version=self.VERSION, body=body)
        )


