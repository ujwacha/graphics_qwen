#version 330 core

// Output color
out vec4 FragColor;

// Input from vertex shader
in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;

// Uniforms
uniform sampler2D ourTexture;
uniform vec3 lightPos;      // Light position
uniform vec3 viewPos;       // Camera position
uniform vec3 lightColor;    // Light color

void main()
{
    // Ambient lighting
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse lighting
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Combine results
    vec3 result = (ambient + diffuse) * texture(ourTexture, TexCoord).rgb;
    
    FragColor = vec4(result, 1.0);
}