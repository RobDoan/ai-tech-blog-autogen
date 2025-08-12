#!/usr/bin/env python3
"""
Simple test runner script for the Trend Spotter module.
Run this script to execute all tests for the trend_spotter.py module.
"""

import subprocess
import sys
import os

def main():
    """Run tests for the trend spotter module"""
    print("🧪 Running Trend Spotter Tests")
    print("=" * 50)

    # Change to project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    try:
        # Install test dependencies if needed
        print("📦 Installing test dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", ".[test]"
        ], check=True)

        # Run the tests
        print("\n🔍 Running tests...")
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/test_trend_spotter.py",
            "-v",  # verbose output
            "--tb=short"  # shorter traceback format
        ], check=False)

        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(result.returncode)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running tests: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ pytest not found. Please install test dependencies:")
        print("   uv add --dev pytest pytest-mock")
        print("   or")
        print("   pip install pytest pytest-mock")
        sys.exit(1)

if __name__ == "__main__":
    main()