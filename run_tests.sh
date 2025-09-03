#!/bin/bash

# Test runner script for Semantic Retrieval System API
# This script provides easy ways to run different types of tests

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if pytest is installed
check_pytest() {
    if ! command -v pytest &> /dev/null; then
        print_error "pytest is not installed. Installing test dependencies..."
        pip install -r requirements.txt
    fi
}

# Check if API server is running
check_api_server() {
    print_header "Checking API Server"
    
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        print_success "API server is running"
        return 0
    else
        print_warning "API server is not running on localhost:8080"
        echo "Please start the API server first:"
        echo "  python backendapi.py"
        echo "  # or"
        echo "  ./start_api.sh"
        return 1
    fi
}

# Install test dependencies
install_deps() {
    print_header "Installing Test Dependencies"
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Run basic tests
run_basic_tests() {
    print_header "Running Basic API Tests"
    pytest tests/test_api.py -v --tb=short
}

# Run performance tests
run_performance_tests() {
    print_header "Running Performance Tests"
    pytest tests/test_performance.py -m "performance" -v --tb=short
}

# Run stress tests
run_stress_tests() {
    print_header "Running Stress Tests"
    pytest tests/test_performance.py -m "stress" -v --tb=short
}

# Run all tests
run_all_tests() {
    print_header "Running All Tests"
    pytest tests/ -v --tb=short
}

# Run fast tests only
run_fast_tests() {
    print_header "Running Fast Tests"
    pytest tests/ -m "not slow and not stress" -v --tb=short
}

# Run tests with coverage
run_coverage_tests() {
    print_header "Running Tests with Coverage"
    pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v --tb=short
    print_success "Coverage report generated in htmlcov/"
}

# Generate test report
generate_report() {
    print_header "Generating Test Report"
    mkdir -p reports
    pytest tests/ --html=reports/test_report.html --self-contained-html -v --tb=short
    print_success "Test report generated in reports/test_report.html"
}

# Clean test artifacts
clean_artifacts() {
    print_header "Cleaning Test Artifacts"
    rm -rf .pytest_cache/
    rm -rf htmlcov/
    rm -rf reports/
    rm -rf .coverage
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    print_success "Test artifacts cleaned"
}

# Show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Test runner for Semantic Retrieval System API"
    echo ""
    echo "Options:"
    echo "  basic       Run basic functional tests (default)"
    echo "  performance Run performance tests"
    echo "  stress      Run stress/load tests"
    echo "  all         Run all tests"
    echo "  fast        Run fast tests only (exclude slow tests)"
    echo "  coverage    Run tests with coverage report"
    echo "  report      Generate HTML test report"
    echo "  install     Install test dependencies"
    echo "  clean       Clean test artifacts"
    echo "  check       Check if API server is ready"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Run basic tests"
    echo "  $0 basic        # Run basic tests"
    echo "  $0 performance  # Run performance tests"
    echo "  $0 all          # Run all tests"
    echo "  $0 coverage     # Run tests with coverage"
}

# Main logic
main() {
    local command=${1:-basic}
    
    # Check pytest installation
    check_pytest
    
    case $command in
        "basic"|"")
            if check_api_server; then
                run_basic_tests
            fi
            ;;
        "performance")
            if check_api_server; then
                run_performance_tests
            fi
            ;;
        "stress")
            if check_api_server; then
                run_stress_tests
            fi
            ;;
        "all")
            if check_api_server; then
                run_all_tests
            fi
            ;;
        "fast")
            if check_api_server; then
                run_fast_tests
            fi
            ;;
        "coverage")
            if check_api_server; then
                run_coverage_tests
            fi
            ;;
        "report")
            if check_api_server; then
                generate_report
            fi
            ;;
        "install")
            install_deps
            ;;
        "clean")
            clean_artifacts
            ;;
        "check")
            check_api_server
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
