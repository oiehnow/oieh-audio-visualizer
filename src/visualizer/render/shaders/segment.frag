#version 330
uniform vec3  u_color_a;
uniform vec3  u_color_b;
uniform int   u_grad;      // 0 solid, 1 gradient along polyline t
uniform float u_half_w;
uniform float u_rms;       // frame loudness 0..1 (available to styles; currently unused)

in vec2 v_p;
flat in vec2 v_a;
flat in vec2 v_b;
in float v_t;
out vec4 fragColor;

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    float d = sdSegment(v_p, v_a, v_b) - u_half_w;      // capsule: round caps/joints
    float cov = clamp(0.5 - d, 0.0, 1.0);
    vec3 col = (u_grad == 0) ? u_color_a : mix(u_color_a, u_color_b, v_t);
    fragColor = vec4(col * cov, cov);                   // PREMULTIPLIED
}
