from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
#prediction.setModelTypeAsResNet()
prediction.setModelTypeAsInceptionV3()
prediction.setModelPath("idenprof/models/model.h5")
prediction.setJsonPath("idenprof/json/model_class.json")
prediction.loadModel(num_objects=3)

predictions, probabilities = prediction.predictImage("image.jpg", result_count=3)

for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction , " : " , eachProbability)
