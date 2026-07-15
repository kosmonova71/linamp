from __future__ import annotations


class ShaderModule:
    def __init__(self, name: str, code: str):
        self.name = name
        self.code = code


# NOTE:
# Modules are concatenated into the fragment shader body.
# They assume the following identifiers exist in the generated shader:
#   - vec2 uv (working UV), vec2 gl_FragCoord-based space already done
#   - float u_time
#   - float warp_strength
#   - vec2 uv will be modified in-place where appropriate
#   - (optionally) floats like p/n/col for intermediate usage

MODULES: dict[str, ShaderModule] = {
    "noise": ShaderModule(
        "noise",
        """
float n = sin(
    uv.x * 12.0 +
    uv.y * 8.0 +
    u_time
);
""",
    ),
    "warp": ShaderModule(
        "warp",
        """
uv += sin(
    uv.yx * 10.0 +
    u_time
) * warp_strength;
""",
    ),
    "fractal": ShaderModule(
        "fractal",
        """
for(int i=0;i<5;i++)
{
    uv = abs(uv) / dot(uv,uv)
          - 0.5;
}
""",
    ),
    "particles": ShaderModule(
        "particles",
        """
float p =
sin(
 length(uv) * 40.0
 - u_time*5.0
);
""",
    ),
}

