# CVAT Polygon-to-Mask API Converter

This tool automates the process of converting vector polygons into RLE masks within a CVAT (Computer Vision Annotation Tool) task. It uses the CVAT SDK to fetch existing annotations, perform the conversion, and upload the results back to the server without manual file handling.

## 🚀 Key Features
- **Direct API Integration**: Connects to your CVAT instance using the official SDK.
- **Auto-Conversion**: Transforms XML polygon data into frame-perfect RLE masks using OpenCV.
- **Duplicate Protection**: Checks existing annotations to prevent uploading redundant shapes.
- **Full Automation**: Automatically downloads, extracts, processes, and updates annotations in one go.
- **Cleanup**: Uses temporary directories to ensure no junk files are left on your local machine.

## 🛠 Tech Stack
- **Python 3.12.9**
- **CVAT SDK**
- **OpenCV** (for mask rasterization)
- **NumPy**

## 📖 How It Works
1. Connects to the CVAT server and retrieves task metadata.
2. Downloads the current annotations in `CVAT for images 1.1` format.
3. Parses polygons and draws them onto a binary mask.
4. Encodes the masks into RLE format.
5. Uploads the new mask annotations back to the specified Task ID.

## ⚙️ Configuration
Update the following constants in the script:
- `CV_URL`: Your CVAT server address.
- `TASK_ID`: The ID of the task you want to process.
- `USERNAME` / `PASSWORD`: Your credentials.

## 🛠 Installation & Usage

### 1. Install dependencies
It is crucial to install the `cvat-sdk` with `[masks]` support for proper RLE handling:
```
pip install "cvat-sdk[masks]" opencv-python numpy
```

### 2. Configuration
Open the script and edit the **SETTINGS** section:
- `CVAT_URL`: Your server address (e.g., `http://localhost:8080`).
- `TASK_ID`: The ID of your task (found in the CVAT task URL).
- `USERNAME` / `PASSWORD`: Your CVAT credentials.

### 3. Run the converter
```
python main.py
```

## 📸 Screenshots & Workflow

### 1. Initial Annotations
<img width="2977" height="1747" alt="image" src="https://github.com/user-attachments/assets/f5081006-a222-415b-8c69-a2b968ba2465" />

__Description: The task state before running the script, showing objects annotated with standard **Polygons**.__

### 2. Final Result
<img width="3167" height="1857" alt="image" src="https://github.com/user-attachments/assets/09edaf13-5fd4-47d0-85dc-3c4d2c83cbd2" />

__Description: The updated task in the CVAT interface. All polygons are now converted to **Mask** types, ready for bitmask export.__

## ⚠️ Disclaimer
This script uses the `CVAT SDK`. Ensure your CVAT server version is compatible with the SDK version installed.
