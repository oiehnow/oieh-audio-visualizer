#version 330
// Same y-down pixel-space projection as rect.vert (orientation invariant).
uniform vec2  u_res;
uniform float u_half_w;   // thickness / 2, px
uniform float u_pad;      // AA pad, px

in vec2 in_unit;          // corners of the unit quad (0|1, 0|1)
in vec4 in_seg;           // per-instance: p0.xy, p1.xy px
in vec2 in_t01;           // per-instance: polyline t at p0, p1

out vec2 v_p;
flat out vec2 v_a;
flat out vec2 v_b;
out float v_t;

void main() {
    vec2 a = in_seg.xy, b = in_seg.zw;
    vec2 d = b - a;
    float len = max(length(d), 1e-6);
    vec2 dir = d / len, nrm = vec2(-dir.y, dir.x);
    float ext = u_half_w + u_pad;                        // covers round caps + AA
    vec2 p = a + dir * mix(-ext, len + ext, in_unit.x)
               + nrm * mix(-ext, ext, in_unit.y);
    v_p = p; v_a = a; v_b = b;
    v_t = mix(in_t01.x, in_t01.y, in_unit.x);
    gl_Position = vec4(p / u_res * 2.0 - 1.0, 0.0, 1.0);
}
