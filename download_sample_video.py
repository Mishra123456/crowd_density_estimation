import urllib.request
import os

def main():
    # URL for a widely used public sample video of a street with people
    video_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
    output_path = "sample_video.mp4"
    
    print(f"Downloading sample video from {video_url}...")
    try:
        urllib.request.urlretrieve(video_url, output_path)
        print(f"\nSuccess! Saved to: {os.path.abspath(output_path)}")
        print("\nYou can now test it with:")
        print(f"python main.py --source \"{output_path}\"")
    except Exception as e:
        print(f"\nFailed to download the video: {e}")
        
if __name__ == "__main__":
    main()
