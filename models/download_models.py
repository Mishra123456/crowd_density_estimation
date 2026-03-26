import os
import urllib.request

def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {os.path.basename(filepath)}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                target_size = int(response.info().get('Content-Length', 0))
                downloaded = 0
                block_size = 1024 * 8
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if target_size > 0:
                        progress = (downloaded / target_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end="")
            print(f"\n{os.path.basename(filepath)} download complete.\n")
        except Exception as e:
            print(f"Failed to download {url}. Error: {e}")
    else:
        print(f"File {os.path.basename(filepath)} already exists.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # YOLOv4-tiny files for vastly improved accuracy
    cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
    weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
    names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    
    cfg_path = os.path.join(current_dir, "yolov4-tiny.cfg")
    weights_path = os.path.join(current_dir, "yolov4-tiny.weights")
    names_path = os.path.join(current_dir, "coco.names")
    
    print("Fetching YOLOv4-tiny weights and configs for higher accuracy...")
    download_file(names_url, names_path)
    download_file(cfg_url, cfg_path)
    download_file(weights_url, weights_path)
    
    print("YOLOv4-tiny models downloaded successfully into the 'models' directory.")
