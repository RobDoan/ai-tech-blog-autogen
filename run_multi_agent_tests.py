#!/usr/bin/env python3
"""
Comprehensive test runner for the Multi-Agent Blog Writer system.

This script runs different categories of tests with appropriate configuration
and provides detailed reporting for the multi-agent blog generation system.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run tests for the Multi-Agent Blog Writer system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Categories:
  unit         - Fast unit tests for individual components
  integration  - Integration tests requiring component interaction
  all          - All tests (unit + integration)
  api          - Tests requiring real OpenAI API access
  slow         - Long-running performance tests
  coverage     - Generate detailed coverage report

Examples:
  python run_multi_agent_tests.py unit
  python run_multi_agent_tests.py integration
  python run_multi_agent_tests.py all --verbose
  python run_multi_agent_tests.py api  # Requires OPENAI_API_KEY
        """
    )
    
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "all", "api", "slow", "coverage"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--parallel", "-n",
        type=int,
        help="Run tests in parallel (number of workers)"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if we're in a virtual environment
    if not (sys.prefix != sys.base_prefix or hasattr(sys, 'real_prefix')):
        print("⚠️  Warning: Not running in a virtual environment")
        print("   Consider running: uv sync && source .venv/bin/activate")
    
    # Build base command
    base_cmd = ["uv", "run", "pytest"]
    
    # Add common options
    if args.verbose:
        base_cmd.append("-v")
    if args.fail_fast:
        base_cmd.append("-x")
    if args.parallel:
        base_cmd.extend(["-n", str(args.parallel)])
    
    # Configure test selection and options based on type
    success = True
    
    if args.test_type == "unit":
        cmd = base_cmd + [
            "-m", "unit",
            "tests/test_multi_agent_models.py",
            "tests/test_base_agent.py",
            "tests/test_content_planner_agent.py",
            "--tb=short"
        ]
        success &= run_command(" ".join(cmd), "Unit Tests")
    
    elif args.test_type == "integration":
        cmd = base_cmd + [
            "-m", "integration",
            "tests/test_blog_writer_orchestrator.py",
            "tests/test_multi_agent_integration.py",
            "--tb=line",
            "--timeout=300"
        ]
        success &= run_command(" ".join(cmd), "Integration Tests")
    
    elif args.test_type == "all":
        # Run unit tests first
        unit_cmd = base_cmd + [
            "-m", "unit",
            "tests/test_multi_agent_models.py",
            "tests/test_base_agent.py",
            "tests/test_content_planner_agent.py",
            "--tb=short"
        ]
        success &= run_command(" ".join(unit_cmd), "Unit Tests")
        
        if success:
            # Run integration tests
            integration_cmd = base_cmd + [
                "-m", "integration",
                "tests/test_blog_writer_orchestrator.py",
                "tests/test_multi_agent_integration.py",
                "--tb=line",
                "--timeout=300"
            ]
            success &= run_command(" ".join(integration_cmd), "Integration Tests")
    
    elif args.test_type == "api":
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ API tests require OPENAI_API_KEY environment variable")
            print("   Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
            return False
        
        cmd = base_cmd + [
            "-m", "api",
            "tests/test_multi_agent_integration.py::TestMultiAgentIntegration::test_real_api_blog_generation",
            "--tb=long",
            "--timeout=600"
        ]
        success &= run_command(" ".join(cmd), "API Integration Tests")
    
    elif args.test_type == "slow":
        cmd = base_cmd + [
            "-m", "slow",
            "tests/",
            "--tb=line",
            "--timeout=600"
        ]
        success &= run_command(" ".join(cmd), "Slow/Performance Tests")
    
    elif args.test_type == "coverage":
        cmd = base_cmd + [
            "tests/",
            "--cov=src/autogen_blog",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=75",
            "--tb=short"
        ]
        success &= run_command(" ".join(cmd), "All Tests with Coverage")
        
        if success:
            print(f"\n📊 Coverage report generated:")
            print(f"   HTML: file://{project_root}/htmlcov/index.html")
            print(f"   XML:  {project_root}/coverage.xml")
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests PASSED!")
        print("✅ Multi-Agent Blog Writer system is working correctly")
    else:
        print("💥 Some tests FAILED!")
        print("❌ Check the output above for details")
    print(f"{'='*60}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)