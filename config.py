OUTPUT_SIZE = (256, 256)
MODEL_IMG_SIZE = (64, 64)
HEALTHBAR_OFFSET_Y = 0.1 # ratio of image down from top where center of health bar is

# downloaded video names
VIDEO_NAMES = [
    "video_data/video1.mp4",
    "video_data/video2.mp4",
    "video_data/video3.mp4",
    "video_data/video4.mp4",
    "video_data/video5.mp4",
    "video_data/video6.mp4",
    "video_data/video7.mp4",
    "video_data/video8.mp4",
    "video_data/video9.mp4",
    "video_data/video10.mp4",
    "video_data/video11.mp4",
]

# determined at square resolution
CROP_BOUNDS = [
    (0.0864583, 0.588542, 0.740625, 0.084375),
    (0.0, 0.714583, 1.0, 0.0),
    (0.135417, 0.8625, 1.0, 0.0),
    (0.291667, 1.0, 0.939583, 0.0208333),
    (0.291667, 1.0, 0.939583, 0.0208333),
    (0.157292, 0.845833, 0.946875, 0.0),
    (0.157292, 0.845833, 0.946875, 0.0),
    (0.123958, 0.872917, 1.0, 0.0),
    (0.0, 0.738542, 1.0, 0.0),
    (0.291667, 1.0, 0.939583, 0.0208333),
    (0.316667, 1.0, 0.923958, 0.0177083),
]

SKIP_VIDEOS = { 0, 1, 2, 7, 9 } # indices of videos that we want to skip (bad quality, for instance)
TEST_VIDEOS = { 0 }
HEALTHBAR_TOLERANCE = 0.05

ENABLE_HEALTH_MASK = False
EVAL_PERCENTAGES = [50, 75, 95]

