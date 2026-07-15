from __future__ import annotations

from dataclasses import dataclass

import OpenGL.GL as GL


@dataclass
class _PingPongTarget:
    fbo: int | None = None
    tex: int | None = None


class FeedbackBuffer:
    """Ping-pong feedback buffer.

    Rendering model (matches requested ownership semantics):
      - shader samples from `read` texture (previous frame)
      - shader renders into `write` FBO (new frame)
      - caller calls swap() after draw
    """

    def __init__(self, res: int = 512):
        self.res = int(res)
        self.read = _PingPongTarget()
        self.write = _PingPongTarget()

    def init_gl(self) -> None:
        self._delete_if_needed()

        # Create two targets for ping-pong.
        self.read = self._create_target()
        self.write = self._create_target()

        # Clear both.
        for target in (self.read, self.write):
            if target.fbo is None:
                continue
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(target.fbo))
            GL.glViewport(0, 0, self.res, self.res)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _create_target(self) -> _PingPongTarget:
        fbo = int(GL.glGenFramebuffers(1))
        tex = int(GL.glGenTextures(1))

        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA16F,
            self.res,
            self.res,
            0,
            GL.GL_RGBA,
            GL.GL_HALF_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            tex,
            0,
        )

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FeedbackBuffer FBO incomplete: {hex(status)}")

        return _PingPongTarget(fbo=fbo, tex=tex)

    def _delete_if_needed(self) -> None:
        for target in (self.read, self.write):
            try:
                if target.tex is not None:
                    GL.glDeleteTextures(1, [int(target.tex)])
            except Exception:
                pass
            try:
                if target.fbo is not None:
                    GL.glDeleteFramebuffers(1, [int(target.fbo)])
            except Exception:
                pass

        self.read = _PingPongTarget()
        self.write = _PingPongTarget()

    def swap(self) -> None:
        self.read, self.write = self.write, self.read

    def bind_previous(self, texture_unit: int = 0) -> None:
        """Bind previous-frame texture to a texture unit."""
        if self.read.tex is None:
            raise RuntimeError("FeedbackBuffer not initialized")
        GL.glActiveTexture(GL.GL_TEXTURE0 + int(texture_unit))
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(self.read.tex))

    def bind_write_fbo(self) -> None:
        if self.write.fbo is None:
            raise RuntimeError("FeedbackBuffer not initialized")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(self.write.fbo))
        GL.glViewport(0, 0, self.res, self.res)

    def unbind(self) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

