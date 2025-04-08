#version 300 es

precision mediump int;

in vec3 a_position;
in vec3 a_texcoord;
in vec2 a_dotcoord;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform int u_bpcmask;
uniform float u_brightness;

out vec3 v_texcoord;
out vec2 v_dotcoord;
out vec3 v_bpcscale;

void main() {
    gl_Position = u_proj * u_view * vec4(a_position, 1.0);
    v_texcoord = a_texcoord;
    v_dotcoord = a_dotcoord;

    v_bpcscale = u_brightness / vec3(float(u_bpcmask & 0xe0), float(u_bpcmask & 0x1c), float(u_bpcmask & 0x03));

}
