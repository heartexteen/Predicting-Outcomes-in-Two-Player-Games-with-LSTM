# ArcadeVit
ArcadeVit is an open-source database designed for making binary winner predictions in the Street Fighter II video game. It operates by using data extracted from the percentage change in the health bar every 5 frames and the video frames. This platform enables the study and improvement of prediction models in gaming analytics.
#Installation 
#To set up the ArcadeVit platform, please follow these steps:
# Clone the repository
git clone https://github.com/yourgithub/arcadevit.git

# Navigate to the project directory
cd arcadevit

# Install required dependencies
pip install -r requirements.txt

#Video Data Preparation
#To get the videos and prepare them for processing:

#Download the videos by running the download.py script:
python download.py
#Ensure there is a folder called video_data to store the downloaded videos.
#Crop and resize the videos using the preprocessor.py script:
python preprocessor.py
#Label the videos with the labeler.py script:
python labeler.py

#Configuration
#Configure all other scripts globally using config.py. For example, to disable health masking, set ENABLE_HEALTH_MASK to False in config.py:
ENABLE_HEALTH_MASK = False
#Changes to config.py take effect at runtime, so you do not need to re-run download, process, or label scripts when modifying settings.

#Model Training
#To utilize the algorithm for making predictions:

#Download the dataset Ultimate_sheets.xlsx.

#Open runlstm.py and execute it:
python runlstm.py

#Modify the round_interv variable in runlstm.py to any value between 0 and 1. This adjustment specifies the round progression percentage you wish to use as test data for making predictions:
round_interv = 0.75

# Set random seed for reproducibility
seed = 42
round_interv = 0.75
random.seed(a=seed)
np.random.seed(seed=seed)
torch.manual_seed(seed=seed)

# Set compute device to GPU if available or CPU otherwise
device = ("cuda" if torch.cuda.is_available() else "cpu")
```

#Features

#Video Processing: Download, crop, resize, and label video data for analysis.
#Model Training: Train LSTM models to predict winners based on health bar changes.
#Customizable Configuration: Adjust global settings in config.py.
#Performance Metrics: Evaluate model performance with various metrics.
#Reproducibility: Ensure consistent results with seed settings.

#Author Kittimate Chulajata, Sean Wu, Eric Laukien
#Feel free to contribute to the project or report any issues you encounter. Happy coding!






