import subprocess
import sys

# List of required packages
packages = [
    "PyOpenGL",
    "PyOpenGL_accelerate",
    "glfw",
    "numpy",
    "Pillow"
]

def install_packages():
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

if __name__ == "__main__":
    print("Installing required packages...")
    install_packages()
    print("All packages installed successfully!")
