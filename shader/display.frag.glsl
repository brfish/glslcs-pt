#version 450 core

in vec2 vTexCoords;

out vec4 fColor;

uniform sampler2D uImage;

void main() {
    vec3 color = texture(uImage, vTexCoords).rgb;
    color = pow(color, vec3(1.0 / 2.2));
    fColor = vec4(color, 1.0);
}