#!/usr/bin/env python3
"""
Check if PEFT is installed and provide installation instructions
"""

def check_peft_installation():
    """Check if PEFT is properly installed."""
    try:
        import peft
        print("✅ PEFT is installed successfully!")
        print(f"PEFT version: {peft.__version__}")
        return True
    except ImportError:
        print("❌ PEFT is not installed!")
        print("\nTo install PEFT, run:")
        print("pip install peft")
        print("\nOr for the latest version:")
        print("pip install git+https://github.com/huggingface/peft.git")
        return False
    except Exception as e:
        print(f"❌ Error checking PEFT: {e}")
        return False

if __name__ == "__main__":
    print("Checking PEFT installation...")
    check_peft_installation() 