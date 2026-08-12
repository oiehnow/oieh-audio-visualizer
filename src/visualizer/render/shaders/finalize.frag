#version 330
// Composites glow under the crisp layer over the background, then UNPREMULTIPLIES.
// The wire format to ffmpeg is STRAIGHT-alpha RGBA — this unpremultiply is not optional.
uniform sampler2D u_layer;        // premultiplied, full res
uniform sampler2D u_glow;         // premultiplied, quarter res (bilinear upsample) or 1x1 black
uniform sampler2D u_bg_tex;       // background image (opaque), or 1x1 black when unused
uniform vec4  u_bg;               // PREMULTIPLIED bg: (0,0,0,0) transparent, or (r,g,b,1) solid
uniform int   u_bg_mode;          // 0 = u_bg color, 1 = u_bg_tex cover-fit image
uniform vec2  u_bg_uv_scale;      // cover-fit crop: visible fraction of the texture per axis
uniform float u_glow_strength;    // 0 when glow disabled

in vec2 v_uv;
out vec4 fragColor;

vec4 over(vec4 src, vec4 dst) { return src + dst * (1.0 - src.a); }

void main() {
    vec4 c;
    if (u_bg_mode == 1) {
        // v_uv.y == 0 is the image top in this app's y-down convention, matching
        // the texture's row order (PIL top row first) — no flip.
        c = vec4(texture(u_bg_tex, (v_uv - 0.5) * u_bg_uv_scale + 0.5).rgb, 1.0);
    } else {
        c = u_bg;
    }
    vec4 g = clamp(texture(u_glow, v_uv) * u_glow_strength, 0.0, 1.0);
    g.rgb  = min(g.rgb, vec3(g.a));   // keep premultiplied valid (rgb <= a) after boost
    c = over(g, c);                   // glow sits UNDER the crisp layer
    c = over(texture(u_layer, v_uv), c);
    fragColor = vec4(c.a > 0.0 ? c.rgb / c.a : vec3(0.0), c.a);   // -> STRAIGHT alpha
}
