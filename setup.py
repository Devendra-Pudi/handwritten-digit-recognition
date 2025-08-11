import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies. Please install manually:")
        print("pip install -r requirements.txt")
        return False

def check_tensorflow():
    """Check if TensorFlow is working correctly"""
    print("Checking TensorFlow installation...")
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} installed successfully!")
        
        # Check for GPU support
        if tf.config.list_physical_devices('GPU'):
            print("✓ GPU support available!")
        else:
            print("ℹ Running on CPU (GPU not available)")
            
        return True
    except ImportError as e:
        print("✗ TensorFlow not installed correctly")
        print(f"Error: {e}")
        print("\nTrying to fix TensorFlow installation...")
        return fix_tensorflow_installation()
    except Exception as e:
        print(f"✗ TensorFlow has issues: {e}")
        print("\nTrying to fix TensorFlow installation...")
        return fix_tensorflow_installation()

def fix_tensorflow_installation():
    """Attempt to fix TensorFlow installation"""
    print("Attempting to fix TensorFlow installation...")
    
    try:
        # Try to uninstall any existing problematic installations
        print("Cleaning up existing TensorFlow installations...")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'tensorflow', 'tensorflow-intel', 'tensorflow-cpu', '-y'], 
                      capture_output=True)
        
        # Install a stable version
        print("Installing TensorFlow (stable version)...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.13.0', '--no-cache-dir'], 
                               capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print("Standard installation failed, trying CPU-only version...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow-cpu==2.13.0', '--no-cache-dir'], 
                                   capture_output=True, text=True, timeout=300)
            
        if result.returncode == 0:
            print("✓ TensorFlow installation fixed!")
            # Test the installation
            try:
                import tensorflow as tf
                print(f"✓ TensorFlow {tf.__version__} working correctly!")
                return True
            except:
                print("✗ TensorFlow still not working after fix attempt")
                return False
        else:
            print(f"✗ Failed to install TensorFlow: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ TensorFlow installation timed out")
        print("Please try installing manually with: pip install tensorflow==2.13.0")
        return False
    except Exception as e:
        print(f"✗ Error during TensorFlow fix: {e}")
        print("Please try installing manually with: pip install tensorflow==2.13.0")
        return False

def check_other_dependencies():
    """Check other required dependencies"""
    print("Checking other dependencies...")
    required_packages = ['numpy', 'matplotlib', 'sklearn', 'seaborn', 'cv2', 'PIL']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} not available")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def create_test_structure():
    """Create directory structure for testing"""
    print("Creating test directories...")
    
    test_dirs = ['test_images', 'models', 'outputs']
    for dir_name in test_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✓ Created {dir_name}/ directory")
        else:
            print(f"ℹ {dir_name}/ directory already exists")

def run_quick_test():
    """Run a quick test to verify everything works"""
    print("Running quick system test...")
    try:
        from main import DigitRecognitionModel
        
        # Create a minimal test
        model = DigitRecognitionModel()
        print("✓ Model class imported successfully")
        
        # Test data loading
        model.load_and_preprocess_data()
        print("✓ Data loading works")
        
        # Test model building
        model.build_cnn_model()
        print("✓ Model building works")
        
        print("✓ All components working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 50)
    print("Handwritten Digit Recognition System Setup")
    print("=" * 50)
    
    # Step 1: Install dependencies
    if not install_requirements():
        print("Please fix dependency issues and run setup again.")
        return
    
    print()
    
    # Step 2: Check TensorFlow
    if not check_tensorflow():
        print("Please fix TensorFlow installation and run setup again.")
        return
    
    print()
    
    # Step 3: Check other dependencies
    if not check_other_dependencies():
        print("Please install missing packages and run setup again.")
        return
    
    print()
    
    # Step 4: Create directory structure
    create_test_structure()
    
    print()
    
    # Step 5: Run quick test
    if not run_quick_test():
        print("System test failed. Please check your installation.")
        return
    
    print()
    print("=" * 50)
    print("Setup completed successfully! 🎉")
    print("=" * 50)
    print()
    print("Next steps:")
    print("1. Run 'python main.py' to train the model")
    print("2. Run 'python test_model.py' for interactive testing")
    print("3. Check README.md for detailed usage instructions")
    print()
    print("Training time: ~5-10 minutes depending on your hardware")
    print("Expected accuracy: >99% on MNIST dataset")

if __name__ == "__main__":
    main()
