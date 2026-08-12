#version 330
uniform vec3  u_color_a;
uniform vec3  u_color_b;
uniform int   u_grad;        // 0 solid, 1 along spectrum (v_t), 2 along bar (v_along)
uniform float u_corner_r;    // px
uniform float u_rms;         // frame loudness 0..1 (available to styles; currently unused)

in vec2  v_local;
in vec2  v_half;
in float v_t;
in float v_along;
out vec4 fragColor;

float sdRoundBox(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

void main() {
    float r = min(u_corner_r, min(v_half.x, v_half.y));
    float d = sdRoundBox(v_local, v_half, r);
    float cov = clamp(0.5 - d, 0.0, 1.0);            // 1px analytic AA
    vec3 col = (u_grad == 0) ? u_color_a
             : mix(u_color_a, u_color_b, (u_grad == 1) ? v_t : v_along);
    fragColor = vec4(col * cov, cov);                // PREMULTIPLIED
}
