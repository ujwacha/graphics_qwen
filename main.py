import glfw
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# Vertex Shader
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

# Fragment Shader
fragment_shader_source = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D ourTexture;

void main()
{
    FragColor = texture(ourTexture, TexCoord);
}
"""

def generate_sphere(radius, sectors, stacks):
    vertices = []
    indices = []
    tex_coords = []

    for i in range(stacks + 1):
        phi = np.pi * i / stacks  # [0, pi]
        for j in range(sectors + 1):
            theta = 2 * np.pi * j / sectors  # [0, 2pi]

            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.cos(phi)
            z = radius * np.sin(phi) * np.sin(theta)

            u = 1 - (j / sectors)
            v = 1 - (i / stacks)

            vertices.extend([x, y, z])
            tex_coords.extend([u, v])

    for i in range(stacks):
        for j in range(sectors):
            first = i * (sectors + 1) + j
            second = first + sectors + 1

            indices.append(first)
            indices.append(second)
            indices.append(first + 1)

            indices.append(second)
            indices.append(second + 1)
            indices.append(first + 1)

    return np.array(vertices, dtype=np.float32), np.array(tex_coords, dtype=np.float32), np.array(indices, dtype=np.uint32)

def load_texture(path):
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    # Load image
    image = Image.open(path)
    img_data = np.array(list(image.getdata()), np.uint8)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return texture

def main():
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "Planet Renderer", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    shader = compileProgram(
        compileShader(vertex_shader_source, GL_VERTEX_SHADER),
        compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    )

    vertices, tex_coords, indices = generate_sphere(1.0, 50, 50)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes + tex_coords.nbytes, None, GL_STATIC_DRAW)
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
    glBufferSubData(GL_ARRAY_BUFFER, vertices.nbytes, tex_coords.nbytes, tex_coords)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(vertices.nbytes))
    glEnableVertexAttribArray(1)

    texture = load_texture("earth_texture.jpg")  # Replace with your actual path

    glEnable(GL_DEPTH_TEST)

    projection = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.02, -1.0],
        [0.0, 0.0, -0.2, 0.0]
    ], dtype=np.float32)

    view = np.eye(4, dtype=np.float32)
    model = np.eye(4, dtype=np.float32)

    angle = 0.0

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)

        # Set matrices
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection)
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model)

        glBindTexture(GL_TEXTURE_2D, texture)
        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)
        glfw.poll_events()

        angle += 0.01
        model[0][0] = np.cos(angle)
        model[0][2] = np.sin(angle)
        model[2][0] = -np.sin(angle)
        model[2][2] = np.cos(angle)

    glfw.terminate()

if __name__ == "__main__":
    main()
