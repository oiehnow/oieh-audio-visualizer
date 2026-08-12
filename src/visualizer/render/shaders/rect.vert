#version 330
// Pixel space is y-DOWN; the projection ndc = px/res*2-1 puts image-top at GL-bottom
// so fbo reads stream top-row-first (orientation invariant — never flip elsewhere).
uniform vec2  u_res;      // (W, H) px
uniform vec2  u_anchor;   // BARS: (0, baseline_y_px) or (0, H/2) mirrored; RADIAL: center px
uniform float u_bar_w;    // bar width px
uniform float u_pad;      // AA pad, px
uniform int   u_mode;     // 0 = linear bars, 1 = radial bars
uniform int   u_mirror;

in vec2  in_unit;                  // corners of the unit quad (0|1, 0|1)
in float in_x;                     // static per-instance: bar center x (linear)
in float in_angle;                 // static per-instance: bar angle (radial)
in float in_r0;                    // static per-instance: inner radius (radial)
in float in_t;                     // static per-instance: spectrum position 0..1
in float in_h;                     // dynamic per-instance: bar length px

out vec2  v_local;
out vec2  v_half;
out float v_t;
out float v_along;                 // 0 at bar base, 1 at bar tip (both modes)

void main() {
    float h = max(in_h, u_bar_w);                 // never below a pill "dot"
    v_half  = vec2(u_bar_w, h) * 0.5;
    vec2 corner = (in_unit - 0.5) * 2.0 * (v_half + u_pad);  // padded so AA isn't clipped
    v_local = corner;
    v_t     = in_t;
    v_along = (u_mode == 0) ? (1.0 - in_unit.y) : in_unit.y;
    vec2 p;
    if (u_mode == 0) {
        // y-down: "up" on screen is -y
        float cy = (u_mirror == 1) ? u_anchor.y : u_anchor.y - v_half.y;
        p = vec2(in_x + corner.x, cy + corner.y);
    } else {
        float rmid = in_r0 + ((u_mirror == 1) ? 0.0 : v_half.y);
        vec2 lp = vec2(corner.x, rmid + corner.y);   // +y = outward before rotation
        float c = cos(in_angle), s = sin(in_angle);
        p = u_anchor + vec2(c * lp.x - s * lp.y, s * lp.x + c * lp.y);
    }
    gl_Position = vec4(p / u_res * 2.0 - 1.0, 0.0, 1.0);   // y-down mapping
}
