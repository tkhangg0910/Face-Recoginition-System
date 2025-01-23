# Face Recoginition System

## Installation

Follow these steps to install the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/tkhangg0910/Face-Recoginition-System
    ```
2. Navigate to the project directory:
    ```bash
    cd src
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
Setup milvus in BE:
1. Navigate to the DB Backend directory:
    ```bash
    cd src/BE/db
    ```
2. Download the installation script and save it as standalone.bat:
    ```bash
    Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat
    ```
3. Run the downloaded script to start Milvus as a Docker container.
    ```bash
    standalone.bat start
    ```
4. Run the container
