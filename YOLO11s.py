from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("best.pt")

# Define path to the image file
# The source can be replaced detected_LA_c0.5.part2 or detected_LA_c0.5.part3
source = "detected_LA_c0.5.part1"


# Run inference on the source
results = model(source, save = True)  # list of Results objects