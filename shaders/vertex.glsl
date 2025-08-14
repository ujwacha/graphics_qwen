#version 330 core

// Input vertex attributes
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aNormal;

// Output to fragment shader
out vec2 TexCoord;
out vec3 Normal;
out vec3 FragPos;

// Uniform matrices
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // Calculate fragment position in world space
    FragPos = vec3(model * vec4(aPos, 1.0));
    
    // Transform normal to world space
    Normal = mat3(transpose(inverse(model))) * aNormal;
    
    // Standard vertex transformation
    gl_Position = projection * view * vec4(FragPos, 1.0);
    
    // Pass texture coordinates to fragment shader
    TexCoord = aTexCoord;
}