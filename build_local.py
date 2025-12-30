import os
import sys
import shutil
import subprocess
import platform

def build_and_copy():
    # Define paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    rust_dir = os.path.join(root_dir, "rust_ga_lib")
    python_pkg_dir = os.path.join(root_dir, "gwgan_edgerefine")
    
    # Check if rust dir exists
    if not os.path.exists(rust_dir):
        print(f"Error: Cannot find directory '{rust_dir}'")
        sys.exit(1)

    # 1. Compile with Cargo
    print(f"Building Rust library in {rust_dir}...")
    try:
        # We build inside the rust_ga_lib directory
        subprocess.check_call(["cargo", "build", "--release"], cwd=rust_dir)
    except subprocess.CalledProcessError:
        print("Error: Cargo build failed.")
        sys.exit(1)

    # 2. Identify the compiled artifact
    # Cargo puts output in rust_ga_lib/target/release/
    target_dir = os.path.join(rust_dir, "target", "release")
    
    system = platform.system()
    if system == "Windows":
        src_filename = "gwgan_edgerefine_rs.dll"
        dest_filename = "gwgan_edgerefine_rs.pyd"
    elif system == "Darwin": # MacOS
        src_filename = "libgwgan_edgerefine_rs.dylib"
        dest_filename = "gwgan_edgerefine_rs.so"
    else: # Linux
        src_filename = "libgwgan_edgerefine_rs.so"
        dest_filename = "gwgan_edgerefine_rs.so"

    src_path = os.path.join(target_dir, src_filename)
    dest_path = os.path.join(python_pkg_dir, dest_filename)

    # 3. Copy to the Python Package directory
    if os.path.exists(src_path):
        print(f"Copying artifact:\n  From: {src_path}\n  To:   {dest_path}")
        shutil.copy2(src_path, dest_path)
        print("\nSUCCESS: Library updated. You can now run the main script.")
    else:
        print(f"Error: Compiled file not found at {src_path}")
        print("Check if Cargo.toml [lib] name is 'gwgan_edgerefine_rs'")
        sys.exit(1)

if __name__ == "__main__":
    build_and_copy()