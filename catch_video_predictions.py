import json
import os
from roboflow import Roboflow
import numpy as np
import supervision as sv

PROJECT_NAME = "logotracker"
VIDEO_FILE = "C:\\Projects\\AdsHunter\\src\\sesentaseg.mp4"

'''
#matplotlib.pyplot.ioff()
# Initialize the Roboflow object with your API key
rf = Roboflow(api_key="GShvoXqnCrQbVtWOrVcs")

# Retrieve your current workspace and proje ct name
print(rf.workspace())
#exit()

rf = Roboflow(api_key='GShvoXqnCrQbVtWOrVcs')
project = rf.workspace().project(PROJECT_NAME)
model = project.version(2).model
job_id, signed_url, expire_time = model.predict_video(
    VIDEO_FILE,
    fps=5,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

with open("results.json", "w") as f:
    json.dump(results, f)
    #print(f)
'''

with open('results.json', 'r') as archivo:
    results = json.load(archivo)    

frame_offset = results["frame_offset"]
model_results = results[PROJECT_NAME]

def callback(scene: np.ndarray, index: int) -> np.ndarray:

    if index in frame_offset:
        detections = sv.Detections.from_inference(
            model_results[frame_offset.index(index)]
        )
        indexToSearch = index
    else:
        nearest = min(frame_offset, key=lambda x: abs(x - index))
        detections = sv.Detections.from_inference(
            model_results[frame_offset.index(nearest)]
        )
        indexToSearch = nearest      


    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)


    #labels = [
    #    model_results[frame_offset.index(index)]["class_name"]
    #    for _
    #    in detections.class_id
    #]    

    labels = [
        model_results[frame_offset.index(indexToSearch)]["predictions"]  
        for _
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(scene=scene,    detections=detections)    

    annotated_image = label_annotator.annotate( 
        scene=annotated_image,
        detections=detections
        )

    return annotated_image


sv.process_video(
    source_path=VIDEO_FILE,
    target_path="output.mp4",
    callback=callback,
)