# Face Recognition System

## Demo

Here is a demonstration of the system in action:

[![IMAGE ALT TEXT HERE](https://i.sstatic.net/Vp2cE.png)](https://www.youtube.com/watch?v=Ju-ulKgFzLQ)


Alternatively, you can directly access the video using the link below:

[Demo Video](https://github.com/tkhangg0910/Face-Recoginition-System/blob/main/src/Demo/demo.mp4)
## Installation

Follow these steps to install the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/tkhangg0910/Face-Recognition-System
    ```
2. Navigate to the project directory:
    ```bash
    cd src
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Setup Milvus in BE:
1. Navigate to the DB Backend directory:
    ```bash
    cd src/BE/db
    ```
2. Download the installation script and save it as `standalone.bat`:
    ```bash
    Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat
    ```
3. Run the downloaded script to start Milvus as a Docker container:
    ```bash
    standalone.bat start
    ```
4. Run the container as needed.

---

## Architecture/Pipeline

Below is the pipeline architecture for the **Face Recognition System**:

![image](https://github.com/user-attachments/assets/84c0ac14-5aed-4091-a1f8-01255524298f)

## Reference
https://phamdinhkhanh.github.io/2020/03/12/faceNetAlgorithm.html
https://arxiv.org/abs/1503.03832
https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf
https://medium.com/pythons-gurus/what-is-the-best-face-detector-ab650d8c1225
https://viblo.asia/p/facial-recognition-system-face-alignment-eW65G2BalDO
