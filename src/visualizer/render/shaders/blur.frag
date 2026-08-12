#version 330
uniform sampler2D u_tex;
uniform vec2  u_dir;       // (1,0) horizontal pass, (0,1) vertical pass
uniform float u_spacing;   // tap spacing in texels (sigma/2)
uniform float u_w[13];     // normalized gaussian weights

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec2 texel = u_dir * u_spacing / vec2(textureSize(u_tex, 0));
    vec4 acc = vec4(0.0);
    for (int i = -6; i <= 6; i++) {
        acc += u_w[i + 6] * texture(u_tex, v_uv + float(i) * texel);
    }
    fragColor = acc;   // blur of premultiplied stays premultiplied — no fringes
}
