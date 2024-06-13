#%%

import cv2


# %%
import cv2
import supervision as sv
from ultralytics import YOLO

image = cv2.imread('GPlQ1Tca8AAW_U1.jfif')
model = YOLO('yolov8s.pt')
result = model(image)[0]
detections = sv.Detections.from_ultralytics(result)

len(detections)

#%%
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 0.5
# font_color = (255, 255, 255)
# thickness = 2
# line_type = 2

# # Loop through the detections and draw each one
# for ii in range(len(detections)):
#     detection = detections[ii]
#     bbox = detection.xyxy  # Bounding box in format [x1, y1, x2, y2]
#     class_id = detection.class_id
#     confidence = detection.confidence

#     # Draw the bounding box
#     x1, y1, x2, y2 = map(int, bbox[0])
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)

#     # Put the label and confidence on the image
#     label = f'{sv.classes[class_id]}: {confidence:.2f}'
#     text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
#     text_x = x1
#     text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10

#     cv2.putText(image, label, (text_x, text_y), font, font_scale, font_color, thickness, line_type)


#%% box annotation
box_annotator = sv.BoxAnnotator()

labels = [
    f"{model.model.names[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]
annotated_image = box_annotator.annotate(
    image.copy(), detections=detections, labels=labels
)

sv.plot_image(image=annotated_image, size=(16, 16))


#%%
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

annotated_image = mask_annotator.annotate(image.copy(), detections=detections)

sv.plot_image(image=annotated_image, size=(16, 16))

# %%
import tbb

print(tbb.__file__)
# %%
