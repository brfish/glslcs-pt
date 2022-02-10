#version 450 core

in vec2 vTexCoords;

out vec4 fColor;

uniform sampler2D uImage;

vec3 linearTosRGB(in vec3 c) {
    const float p = 1.0 / 2.4;
    bvec3 less = lessThan(c, vec3(0.0031308));
    c.x = less.x ? c.x * 12.92 : pow(c.x, p) * 1.055 - 0.055;
    c.y = less.y ? c.y * 12.92 : pow(c.y, p) * 1.055 - 0.055;
    c.z = less.z ? c.z * 12.92 : pow(c.z, p) * 1.055 - 0.055;
    return c;
}

void main() {
    vec3 color = texture(uImage, vTexCoords).rgb;
    fColor = vec4(linearTosRGB(color), 1.0);
}