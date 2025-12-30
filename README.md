# **GWGAN Edge Refinement**

This research project integrates a Generative Adversarial Network (GAN) with an Evolutionary Algorithm (EA) to refine graph edge structures. The system uses a Python-based GAN for initial generation and a high-performance Rust extension for evolutionary refinement.

## **Project Structure**

* **gwgan\_edgerefine/**: The main Python application containing the GAN logic, training scripts, and models.  
* **rust\_ga\_lib/**: The Rust library extension that powers the Evolutionary Algorithm. This is compiled into a Python module.

## **Setup & Installation**

This project uses maturin to build the high-performance Rust extension.

### **1\. Prerequisites**

Ensure you have the following installed:

* Python (3.8+)  
* Rust (latest stable toolchain)

Install the build tool:

pip install maturin

### **2\. Build and Install the Rust Extension**

You must install the Rust library into your Python environment before running the main application.

cd rust\_ga\_lib  
maturin develop \--release  
cd ..

**Note:** maturin develop compiles the Rust code and installs it directly into your currently active Python virtual environment. If you want to build a wheel for distribution, use maturin build \--release.

### **3\. Run the Application**

Once the extension is installed, you can run the WGAN training script normally.

python gwgan\_edgerefine/main.py  
