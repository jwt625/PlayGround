# Dockerfile for qiskit-metal with PySide2 GUI support
FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies for PySide2 and GUI
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # Python and pip
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    # Qt5 and GUI dependencies
    qt5-default \
    qttools5-dev-tools \
    libqt5widgets5 \
    libqt5gui5 \
    libqt5core5a \
    libqt5dbus5 \
    libqt5network5 \
    libqt5opengl5 \
    libqt5printsupport5 \
    libqt5svg5 \
    libqt5test5 \
    libqt5xml5 \
    # X11 and GUI support
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxrandr2 \
    libxss1 \
    libgconf-2-4 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    # LLVM/Clang for PySide2
    llvm-10 \
    clang-10 \
    libclang-10-dev \
    # Additional dependencies
    libxml2-dev \
    libxslt1-dev \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    # For numerical computing
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    # Text editors for development
    nano \
    vim \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Install pip for Python 3.9
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.9

# Set environment variables for Qt and LLVM
ENV QT_QPA_PLATFORM=xcb
ENV DISPLAY=:0
ENV LLVM_INSTALL_DIR=/usr/lib/llvm-10
ENV PATH="/usr/lib/llvm-10/bin:$PATH"

# Create working directory
WORKDIR /workspace

# Create Python virtual environment
RUN python3.9 -m venv /workspace/venv

# Activate virtual environment and install compatible versions
RUN /workspace/venv/bin/pip install --upgrade pip setuptools wheel

# Install compatible versions for PySide2 building
RUN /workspace/venv/bin/pip install \
    "numpy==1.20.3" \
    "Cython==0.29.32" \
    "setuptools==59.8.0" \
    "wheel==0.37.1"

# Copy requirements and install script
COPY requirements-docker.txt /workspace/
COPY install-qiskit-metal.sh /workspace/
COPY run-qiskit-metal.sh /workspace/
COPY build-container.sh /workspace/
COPY setup-x11-forwarding.sh /workspace/

# Copy examples folder with quantum chip designs
COPY examples/ /workspace/examples/

# Make all shell scripts executable
RUN chmod +x /workspace/*.sh

# Set proper permissions for the workspace
RUN chmod -R 755 /workspace

# Install PySide2 from wheel during build
RUN /workspace/venv/bin/pip install \
    https://files.pythonhosted.org/packages/c2/9a/78ca8bada6cf4d2798e0c823c025c590517d74445837f4eb50bfddce8737/PySide2-5.15.2.1-5.15.2-cp35.cp36.cp37.cp38.cp39.cp310-abi3-manylinux1_x86_64.whl

# Install all requirements during build
RUN /workspace/venv/bin/pip install -r /workspace/requirements-docker.txt

# Install qiskit-metal during build
RUN /workspace/venv/bin/pip install qiskit-metal

# Install X11 apps for testing (do this during build to save time)
RUN apt-get update && apt-get install -y x11-apps && apt-get clean && rm -rf /var/lib/apt/lists/*

# Verify installation during build
RUN /workspace/venv/bin/python -c "import qiskit_metal, scqubits, PySide2; print('✅ All packages installed successfully in Docker image')"

# Set the entrypoint to bash so we can run commands interactively
CMD ["/bin/bash"]
