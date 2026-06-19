import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {"torch": "torch", "numpy": "numpy", "pygame": "pygame", "scipy": "scipy"}

def is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None

def install_package(package_name: str):
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}!\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing {package_name}: {e}")
        sys.exit(1)

def main():
    print("Checking dependencies...")
    missing_packages = []
    for module, package in REQUIRED_PACKAGES.items():
        if not is_installed(module):
            print(f"  [MISSING] {module} (PyPI: {package})")
            missing_packages.append(package)
        else:
            print(f"  [OK]      {module}")
    if not missing_packages:
        print("\nAll dependencies are already installed! You're good to go.")
        return
    print(f"\nFound {len(missing_packages)} missing dependencies. Starting installation...\n")
    for package in missing_packages:
        install_package(package)
    print("All dependencies have been successfully installed!")

if __name__ == "__main__":
    main()