import numpy as np
import cv2
from copy import copy
from pytube import YouTube
from os import listdir
from os.path import isfile, join
import pickle

video_links = [
    "https://www.youtube.com/watch?app=desktop&v=N0K5llcc6r8",
    "https://www.youtube.com/watch?v=ZeuclUa8QWY",
    "https://www.youtube.com/watch?v=AgBryRrdxx4",
    "https://www.youtube.com/watch?app=desktop&v=5mhlh7Qopj4",
    "https://www.youtube.com/watch?app=desktop&v=5mhlh7Qopj4",
    "https://www.youtube.com/watch?v=WusmMZqT2bQ",
    "https://www.youtube.com/watch?v=XFH7tm7kQA8",
    "https://www.youtube.com/watch?v=lbS5SAI6TRE",
    "https://www.youtube.com/watch?v=bUViuc4HVZ0",
    "https://www.youtube.com/watch?v=TjWouSqe0uc",
    "https://www.youtube.com/watch?v=c0OruxJB-0k",
    "https://www.youtube.com/watch?v=rHPWkoXFIKM",
]

for v in range(len(video_links)):
    name = video_links[v]
    
    try:
        print(f"Downloading {name}")

        yt = YouTube(name)
        filt = yt.streams.filter(file_extension='mp4')

        if len(filt) == 0:
            print("No .mp4 stream found, skipping...")

            continue

        video = filt.get_lowest_resolution()

        print(f"Resolution: {video.resolution}")

        videoName = f"video{v + 1}.mp4"

        video.download("video_data/", videoName, timeout=100000)

        print("Download complete.")
    except Exception as e:
        print(e)
        print("Continuing...")

        continue

    print(f"Processed video {v}")

print("Done.")

