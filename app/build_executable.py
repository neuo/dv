#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Script - Package application as executable

Use PyInstaller to package Python application as standalone executable,
supporting Windows (.exe), macOS (.app), Linux platforms.

Usage:
    python app/build_executable.py

Dependencies:
    pip install pyinstaller
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller
        print(f"✓ PyInstaller installed (version: {PyInstaller.__version__})")
        return True
    except ImportError:
        print("✗ PyInstaller not installed")
        return False

def install_pyinstaller():
    """Install PyInstaller"""
    print("Installing PyInstaller...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], 
                      check=True)
        print("✓ PyInstaller installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ PyInstaller installation failed")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("Checking dependencies...")
    
    required_modules = [
        'torch', 'numpy', 'pandas', 'openpyxl', 
        'sklearn', 'tkinter', 'scipy'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            if module == 'tkinter':
                import tkinter
            elif module == 'sklearn':
                import sklearn
            else:
                __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"Missing dependencies: {', '.join(missing_modules)}")
        return False
    
    return True

def build_executable():
    """Build executable file"""
    print("Starting executable build...")
    
    # Determine operating system
    system = platform.system()
    print(f"Current system: {system}")
    
    # Basic PyInstaller command parameters
    cmd = [
        "pyinstaller",
        "--onefile",                    # Package as single file
        "--name=DoseVerificationSystem", # Executable file name
        "--clean",                      # Clean cache
    ]
    
    # Add system-specific parameters
    if system == "Windows":
        cmd.append("--windowed")        # Hide console on Windows
    elif system == "Darwin":  # macOS
        cmd.append("--windowed")        # Create .app bundle on macOS
    
    # Don't package model files into executable (avoid file size issues)
    # Model files will be provided separately in installation package
    result_dir = Path("../result")
    if result_dir.exists():
        print("✓ Model files detected, will be provided separately (not packaged in executable)")
    else:
        print("Warning: result directory not found")
    
    # Add additional Python files
    cmd.extend(["--add-data", f"model_loader.py{os.pathsep}."])
    
    # Add hidden imports
    hidden_imports = [
        "torch", "torch.nn", "torch.optim",
        "numpy", "pandas", "openpyxl", 
        "sklearn", "scipy", "scipy.interpolate",
        "tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox",
        "pathlib", "threading", "datetime"
    ]
    
    for module in hidden_imports:
        cmd.extend(["--hidden-import", module])
    
    # Add main file
    cmd.append("gui_fixed.py")
    
    try:
        print("Executing build command...")
        print(f"Command: {' '.join(cmd)}")
        
        # Execute build command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Executable build successful")
        
        # Show output location
        dist_dir = Path("dist")
        if dist_dir.exists():
            exe_files = list(dist_dir.glob("*"))
            if exe_files:
                exe_file = exe_files[0]
                file_size = exe_file.stat().st_size / (1024 * 1024)  # MB
                print(f"Executable file: {exe_file.absolute()}")
                print(f"File size: {file_size:.1f} MB")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e}")
        if e.stdout:
            print("Standard output:", e.stdout)
        if e.stderr:
            print("Error output:", e.stderr)
        return False
    except FileNotFoundError:
        print("✗ pyinstaller command not found")
        return False

def create_installer_package():
    """Create installation package"""
    print("Creating installation package...")
    
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("✗ dist directory not found")
        return False
    
    # Find executable file
    exe_files = list(dist_dir.glob("*"))
    if not exe_files:
        print("✗ No executable file found")
        return False
    
    exe_file = exe_files[0]
    
    # Create installation package directory
    package_dir = Path("DoseVerificationSystem_Package")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    # Copy executable file
    shutil.copy2(exe_file, package_dir)
    print(f"✓ Copied executable file: {exe_file.name}")
    
    # Copy model files
    result_dir = Path("../result")
    if result_dir.exists():
        package_result_dir = package_dir / "result"
        shutil.copytree(result_dir, package_result_dir)
        
        # Calculate total model file size
        total_size = sum(f.stat().st_size for f in package_result_dir.glob("*.pth"))
        total_size_gb = total_size / (1024 * 1024 * 1024)
        print(f"✓ Copied model files: {len(list(package_result_dir.glob('*.pth')))} files, {total_size_gb:.1f} GB")
    else:
        print("Warning: Model files not found")
    
    # Create instruction file
    readme_content = """Dose Verification Prediction System
====================================

This is a deep learning-based dose verification prediction system.

Installation Instructions:
1. Copy the entire folder to your desired location
2. Double-click the executable file to run
3. Keep the executable file and result folder in the same directory

Usage Instructions:
1. Select model type (CNN, Transformer, CNNTransformer, CNN_SINGLE)
2. Select precision configuration
3. Select folder containing data files (must include ADT_plan.xlsx, Ref_measure.xlsx, Ref_plan.xlsx)
4. Click "Start Prediction"

File Structure:
- DoseVerificationSystem (executable file)
- result/ (model file directory, required)
- Instructions.txt (this file)

Supported File Formats:
- Excel files (.xlsx, .xls)

System Requirements:
- Windows 10/11, macOS 10.14+, or Linux
- At least 4GB RAM
- At least 3GB available disk space

Notes:
- First run may take longer to load models
- Ensure sufficient memory for deep learning models
- Do not delete or move model files in result folder

Technical Support:
Contact development team for issues.

Version: 1.0
Build time: {build_time}
"""
    
    from datetime import datetime
    readme_path = package_dir / "Instructions.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content.format(build_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    # Create quick installation script
    if platform.system() == "Windows":
        install_script = """@echo off
echo Dose Verification System Quick Install
echo =======================================

set "desktop=%USERPROFILE%\\Desktop"
set "target_dir=%desktop%\\DoseVerificationSystem"

echo Copying to desktop...
if not exist "%target_dir%" mkdir "%target_dir%"
xcopy /E /I /Y "%~dp0*" "%target_dir%\\"

if %errorlevel% == 0 (
    echo ✓ Installation successful!
    echo Program installed to desktop "DoseVerificationSystem" folder
    echo You can now run the executable file
) else (
    echo ✗ Installation failed
)

pause
"""
        script_path = package_dir / "QuickInstallToDesktop.bat"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(install_script)
    
    else:  # macOS/Linux
        install_script = """#!/bin/bash
echo "Dose Verification System Quick Install"
echo "====================================="

desktop="$HOME/Desktop"
target_dir="$desktop/DoseVerificationSystem"

echo "Copying to desktop..."
mkdir -p "$target_dir"
cp -r "$(dirname "$0")"/* "$target_dir/"

if [ $? -eq 0 ]; then
    echo "✓ Installation successful!"
    echo "Program installed to desktop 'DoseVerificationSystem' folder"
    echo "You can now run the executable file"
    chmod +x "$target_dir/DoseVerificationSystem"
else
    echo "✗ Installation failed"
fi

read -p "Press Enter to continue..."
"""
        script_path = package_dir / "QuickInstallToDesktop.sh"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(install_script)
        os.chmod(script_path, 0o755)
    
    # Calculate total package size
    total_package_size = sum(f.stat().st_size for f in package_dir.rglob("*") if f.is_file())
    total_package_size_gb = total_package_size / (1024 * 1024 * 1024)
    
    print(f"✓ Installation package created: {package_dir.absolute()}")
    print(f"Package total size: {total_package_size_gb:.1f} GB")
    return True

def clean_build_files():
    """Clean build files"""
    print("Cleaning build files...")
    
    dirs_to_remove = ["build", "__pycache__"]
    files_to_remove = ["*.spec"]
    
    for dir_name in dirs_to_remove:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"✓ Removed directory: {dir_name}")
    
    for pattern in files_to_remove:
        for file_path in Path(".").glob(pattern):
            file_path.unlink()
            print(f"✓ Removed file: {file_path}")

def main():
    """Main function"""
    print("Dose Verification System Build Tool")
    print("=" * 50)
    
    # Check if running in CI environment
    is_ci = os.environ.get('CI') == 'true' or os.environ.get('GITHUB_ACTIONS') == 'true'
    if is_ci:
        print("Running in CI environment - non-interactive mode")
    
    # Check current directory
    if not Path("gui_fixed.py").exists():
        print("✗ Please run this script in the app directory")
        print("Usage: cd app && python build_executable.py")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("✗ Please install missing dependencies first")
        sys.exit(1)
    
    # Check PyInstaller
    if not check_pyinstaller():
        if not install_pyinstaller():
            sys.exit(1)
    
    # Build executable
    if not build_executable():
        sys.exit(1)
    
    # Create installation package
    if not create_installer_package():
        print("Warning: Installation package creation failed, but executable was generated")
    
    # Ask whether to clean build files (skip in CI)
    if is_ci:
        print("CI environment detected - skipping build file cleanup")
    else:
        try:
            choice = input("\nClean build files? (y/n): ").lower().strip()
            if choice in ['y', 'yes']:
                clean_build_files()
        except KeyboardInterrupt:
            print("\nUser cancelled operation")
    
    print("\n✓ Build complete!")
    print("You can find files at:")
    print("- Executable file: dist/")
    print("- Installation package: DoseVerificationSystem_Package/")

if __name__ == "__main__":
    main() 