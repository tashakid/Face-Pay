"""
Environment setup script for Face Recognition Payment System
This script helps set up the development environment on Windows
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python():
    """Check Python installation"""
    print("🐍 Checking Python installation...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_miniconda():
    """Install Miniconda for easier package management"""
    print("\n📦 Installing Miniconda for easier package management...")
    print("Please download and install Miniconda from: https://docs.conda.io/en/latest/miniconda.html")
    print("After installation, restart your terminal/command prompt")
    
    input("Press Enter after you have installed Miniconda...")
    
    # Check if conda is available
    try:
        result = subprocess.run("conda --version", shell=True, check=True, capture_output=True, text=True)
        print("✅ Conda is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ Conda is not available. Please install Miniconda first.")
        return False

def create_conda_env():
    """Create conda environment"""
    commands = [
        ("conda create -n face-payment python=3.9 -y", "Creating conda environment"),
        ("conda activate face-payment", "Activating conda environment")
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    return True

def install_with_conda():
    """Install packages using conda"""
    packages = [
        "numpy",
        "opencv",
        "mediapipe",
        "pip",
        "requests"
    ]
    
    for package in packages:
        if not run_command(f"conda install {package} -y", f"Installing {package} with conda"):
            return False
    
    # Install Python packages with pip
    pip_packages = [
        "fastapi",
        "uvicorn",
        "pillow",
        "pydantic",
        "python-jose[cryptography]",
        "python-dotenv",
        "firebase-admin"
    ]
    
    for package in pip_packages:
        if not run_command(f"pip install {package}", f"Installing {package} with pip"):
            return False
    
    return True

def setup_manual_instructions():
    """Provide manual setup instructions"""
    print("\n" + "="*60)
    print("🚨 AUTOMATIC SETUP FAILED - MANUAL SETUP REQUIRED")
    print("="*60)
    
    print("\n📋 MANUAL SETUP INSTRUCTIONS:")
    print("\n1. Install Miniconda:")
    print("   Download from: https://docs.conda.io/en/latest/miniconda.html")
    print("   Choose Python 3.9 version for Windows")
    
    print("\n2. Create conda environment:")
    print("   conda create -n face-payment python=3.9 -y")
    print("   conda activate face-payment")
    
    print("\n3. Install packages with conda:")
    print("   conda install numpy opencv mediapipe requests -y")
    
    print("\n4. Install remaining packages with pip:")
    print("   pip install fastapi uvicorn pillow pydantic python-jose[cryptography] python-dotenv firebase-admin")
    
    print("\n5. Navigate to project directory:")
    print("   cd face-payment-backend")
    
    print("\n6. Run the application:")
    print("   python src/main.py")
    
    print("\n7. Alternative - Use pre-compiled wheels:")
    print("   Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/")
    print("   Install with: pip install package_name.whl")
    
    print("\n📚 For more help, visit:")
    print("   - https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html")
    print("   - https://opencv.org/releases/")
    print("   - https://google.github.io/mediapipe/getting_started/install.html")

def main():
    """Main setup function"""
    print("🚀 Face Recognition Payment System - Environment Setup")
    print("="*60)
    
    # Check system
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    # Check Python
    if not check_python():
        setup_manual_instructions()
        return
    
    # Try different setup methods
    print("\n🔧 Trying automatic setup...")
    
    # Method 1: Try conda
    if install_miniconda():
        if create_conda_env():
            if install_with_conda():
                print("\n✅ Setup completed successfully!")
                print("\n🎯 Next steps:")
                print("1. Activate environment: conda activate face-payment")
                print("2. Navigate to project: cd face-payment-backend")
                print("3. Run application: python src/main.py")
                return
    
    # If automatic setup fails, provide manual instructions
    setup_manual_instructions()

if __name__ == "__main__":
    main()