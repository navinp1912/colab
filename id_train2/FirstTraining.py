from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsInceptionV3()
model_trainer.setDataDirectory("idenprof")
#model_trainer.trainModel(num_objects=10, num_experiments=200, enhance_data=True, batch_size=32, show_network_summary=True)
model_trainer.trainModel(num_objects=3, num_experiments=4, enhance_data=True, batch_size=32, show_network_summary=True)
