import json
import os
from roboflow import Roboflow
import numpy as np
import supervision as sv
import datetime
import types

PROJECT_NAME = "logotracker"
VIDEO_FILE = "C:\\Projects\\AdsHunter\\src\\sesentaseg.mp4"
ANNOTATED_VIDEO = "output.mp4"
FPS = 5
results = []

#def __init__(self, name ='', times = 1, h=1, w=1):
class Prediction (object):
    predictionList = {}
    def __init__(self, name,hits,times,h,w):
        self.name = name
        self.hits = hits
        self.times = times #HITS
        self.h = h
        self.w = w
        Prediction.predictionList[name] = self

    """ def __str__(self):
        return self.name +" ["+self.times+"]" """
    
    def get_sreen_percentage(self):
        return self.h * self.w


def getDataProcess(sendRequest):
    os.chdir('C:\\Projects\\AdsHunter')
    #print("Trabajando en el entorno ->" + os.getcwd())
    global results
    if sendRequest == True:
        # Initialize the Roboflow object with your API key
        rf = Roboflow(api_key="GShvoXqnCrQbVtWOrVcs")

        # Retrieve your current workspace and proje ct name
        print("Trabajando en:"+ str(rf.workspace()))  
        rf = Roboflow(api_key='GShvoXqnCrQbVtWOrVcs')
        project = rf.workspace().project(PROJECT_NAME)
        model = project.version(2).model
        job_id, signed_url, expire_time = model.predict_video(
            VIDEO_FILE,
            fps=FPS,
            prediction_type="batch-video",
            #additional_models = ['clip'],#We can process multi Models 
        )
        results = model.poll_until_video_results(job_id)
        with open("results.json", "w") as f:
            json.dump(results, f)
    else:        
        with open('results.json', 'r') as archivo:
            results = json.load(archivo)
            #print(f)    
    return results

def callback(scene: np.ndarray, index: int) -> np.ndarray:

   # poll_until_video_results
    results = getDataProcess(False)
    frame_offset = results["frame_offset"]
    model_results = results[PROJECT_NAME]

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

    labels = [
        model_results[frame_offset.index(indexToSearch)]["predictions"]  
        for _
        in detections.class_id
    ]    

    #annotated_image = bounding_box_annotator.annotate(scene=scene,    detections=detections)

    #annotated_image = label_annotator.annotate(
    #    scene=annotated_image,
    #    detections=detections
    #    )

    

    #detections = sv.Detections()    
    
    dot_annotator = sv.DotAnnotator(sv.Color.BLACK)
    color_annotator = sv.ColorAnnotator()

    dotColor = np.array([0,0,0])
    #annotated_image = bounding_box_annotator.annotate(  scene=scene,    detections=detections)
    annotated_image = dot_annotator.annotate( scene, detections)
    annotated_image = color_annotator.annotate( scene,  detections)
    


    return annotated_image

def writeReport():

    os.chdir('C:\\Projects\\AdsHunter')
    print("Trabajando en el entorno ->" + os.getcwd())
    global results
    if len(results) == 0:
        results = getDataProcess(False)
    
    frame_offset = results["frame_offset"]
    time_offset  = results["time_offset"]
    model_results = results[PROJECT_NAME]

    # import xlsxwriter module
    import xlsxwriter
    
    workbook = xlsxwriter.Workbook('Report.xlsx')
    wsResume = workbook.add_worksheet("Resume")
    wsRaw = workbook.add_worksheet("Raw")
    
    # Start from the first cell.
    # Rows and columns are zero indexed.
    row = 0
    column = 0    
    # iterating through content list
    # result = list(filter(lambda x: (x % 13 == 0), my_list))

    validPredictions = list(filter(lambda x: ( len(x['predictions']) > 0 ), model_results))   
    
    wsResume.write(row, column, 'Propiedades')
    wsResume.write(row, column + 1, 'Hits')
    wsResume.write(row, column + 2, 'Duración')
    wsResume.write(row, column + 3, '% Promedio de impresión')    
    row += 1

    for predictionParent in validPredictions :        
        #anchodelaimagen=validPredictions['image']['with']
        #altodelaimagen=validPredictions['image']['height']

        
        for prediction in predictionParent['predictions']:
            if( prediction['class'] not in Prediction.predictionList ):
                #Adding data for objects Class def __init__(self, name,hits,times,h,w):
                pred = Prediction(prediction['class'],1,predictionParent['time'],prediction['height'],prediction['width'])

            #if( anchodelaimagen<540=columnaizquierda):
            #if( anchodelaimagen)    
                
            
        

            pred = Prediction.predictionList[prediction['class']]
            pred.hits += 1
            pred.times += predictionParent['time']
            
            
    for p in Prediction.predictionList.values():        
        wsResume.write(row, column, p.name)
        wsResume.write(row, column+1, int(p.hits/FPS))
        nsec = int(p.times/FPS)
        wsResume.write(row, column+2, str(datetime.timedelta(seconds = nsec)))
        row += 1
    
    #Creating Raw Report______________________________________________________________
    row = 0
    wsRaw.write(row, column, 'Exp Time')   
    wsRaw.write(row, column + 1, 'Group Frames')
    wsRaw.write(row, column + 2, 'Frame Offset')
    wsRaw.write(row, column + 3, 'Time Offset')
    wsRaw.write(row, column + 4, 'Class')
    wsRaw.write(row, column + 5, 'Width')
    wsRaw.write(row, column + 6, 'Height')
    wsRaw.write(row, column + 7, 'Screen Coverage')
    row += 1

    for key ,result in enumerate(model_results):
        totalScreenPx = result['image']['height'] * result['image']['width']
        wsRaw.write(row, column, result['time'])        
        wsRaw.write(row, column + 1, key)
        wsRaw.write(row, column + 2, frame_offset[key])
        #wsRaw.write(row, column + 3, time_offset[key])
        nSeconds =int(time_offset[key])
        wsRaw.write(row, column + 3, str(datetime.timedelta(seconds = nSeconds)))
        for res in result['predictions']:
            wsRaw.write(row, column + 4, res['class'])
            wsRaw.write(row, column + 5, res['width'])
            wsRaw.write(row, column + 6, res['height'])
            screenCov = ((res['width']*res['height'])/totalScreenPx)*100
            wsRaw.write(row, column + 7, str(screenCov)+"%")
            
        row += 1
        #str(datetime.timedelta(seconds=666))
        
    workbook.close()
    print ("Reporte Creado Exitosamente")

def processVideo():
    sv.process_video(
        source_path=VIDEO_FILE,
        target_path=ANNOTATED_VIDEO,
        callback=callback,
    )

#processVideo()

#Create Report
writeReport()