#!/bin/bash

# Configuration
BLENDER_PATH="/home/ubuntu/xuweiyi-us-south-3/generate_objaverse/rendering/blender-3.2.2-linux-x64/blender"
SCRIPT_PATH="/home/ubuntu/xuweiyi-us-south-3/generate_objaverse/rendering/blender_cpu.py"
GLB_DIR="/home/ubuntu/xuweiyi-us-south-3/generate_objaverse/rendering/objaverse_curated_v2/glbs"
OUTPUT_DIR="/home/ubuntu/xuweiyi-us-south-3/generate_objaverse/rendering/output"
PARALLEL_JOBS=100

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install parallel if it doesn't exist
install_parallel() {
    if ! command_exists parallel; then
        print_warning "GNU parallel not found. Installing..."
        if command_exists apt-get; then
            sudo apt-get update
            sudo apt-get install -y parallel
        elif command_exists yum; then
            sudo yum install -y parallel
        else
            print_error "Cannot install GNU parallel automatically. Please install it manually."
            exit 1
        fi
    fi
}

# Function to create render command
create_render_command() {
    local glb_file="$1"
    local filename=$(basename "$glb_file")
    local name_without_ext="${filename%.glb}"
    local output_path="${OUTPUT_DIR}/${name_without_ext}"
    
    echo "\"$BLENDER_PATH\" --background --python \"$SCRIPT_PATH\" -- --object_path \"$glb_file\" --output_dir \"$output_path\" --mode_multi 1 --frame_num 48"
}

# Main function
main() {
    print_info "Starting parallel GLB rendering script"
    
    # Check if required paths exist
    if [[ ! -d "$GLB_DIR" ]]; then
        print_error "GLB directory not found: $GLB_DIR"
        exit 1
    fi
    
    if [[ ! -f "$BLENDER_PATH" ]]; then
        print_error "Blender executable not found: $BLENDER_PATH"
        exit 1
    fi
    
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        print_error "Blender script not found: $SCRIPT_PATH"
        exit 1
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # Install parallel if needed
    install_parallel
    
    # Find all GLB files
    print_info "Searching for GLB files in $GLB_DIR..."
    
    # Create temporary file to store commands
    local commands_file="render_commands_$$.txt"
    
    # Find all GLB files and create commands
    find "$GLB_DIR" -name "*.glb" -type f | while read -r glb_file; do
        create_render_command "$glb_file"
    done > "$commands_file"
    
    # Count total files
    local total_files=$(wc -l < "$commands_file")
    print_info "Found $total_files GLB files to render"
    
    if [[ $total_files -eq 0 ]]; then
        print_error "No GLB files found!"
        rm -f "$commands_file"
        exit 1
    fi
    
    # Check if dry run
    if [[ "$1" == "--dry-run" ]]; then
        print_info "Dry run mode - showing first 5 commands:"
        head -n 5 "$commands_file" | nl
        print_info "Total commands: $total_files"
        rm -f "$commands_file"
        exit 0
    fi
    
    # Run parallel processing
    print_info "Starting parallel rendering with $PARALLEL_JOBS jobs..."
    print_info "Progress will be shown below..."
    
    # Run the parallel command
    if parallel -j "$PARALLEL_JOBS" --progress --joblog "parallel_jobs_$(date +%Y%m%d_%H%M%S).log" < "$commands_file"; then
        print_info "Parallel rendering completed successfully!"
        
        # Show summary
        local logfile=$(ls -t parallel_jobs_*.log 2>/dev/null | head -n 1)
        if [[ -f "$logfile" ]]; then
            print_info "Job summary from $logfile:"
            tail -n 10 "$logfile"
        fi
        
        print_info "Results saved to $OUTPUT_DIR"
    else
        print_error "Parallel rendering failed!"
        exit 1
    fi
    
    # Clean up
    rm -f "$commands_file"
    
    print_info "Rendering complete!"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --dry-run    Show commands without executing
    --help       Show this help message

Configuration (edit script to modify):
    BLENDER_PATH:   $BLENDER_PATH
    SCRIPT_PATH:    $SCRIPT_PATH
    GLB_DIR:        $GLB_DIR
    OUTPUT_DIR:     $OUTPUT_DIR
    PARALLEL_JOBS:  $PARALLEL_JOBS

Examples:
    $0                    # Run full rendering
    $0 --dry-run         # Show commands without executing
    $0 --help            # Show this help

EOF
}

# Parse command line arguments
case "$1" in
    --help|-h)
        show_help
        exit 0
        ;;
    --dry-run)
        main --dry-run
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac 