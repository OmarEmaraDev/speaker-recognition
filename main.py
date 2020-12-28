import numpy
from loader import loadMFCCData, getSpeakerNames
from model import computeModel, getProbabilityModel

print("Computing MFCCs from dataset ...")
print("")
(trainingSamples, trainingLabels), (testingSamples, testingLabels) = loadMFCCData()

print("Getting data set metadata ...")
print("")
speakerNames = getSpeakerNames()

print("Computing model ...")
print("")
model = computeModel(trainingSamples, trainingLabels, len(speakerNames))
print("")

print("Evaluating model ...")
print("")
model.evaluate(testingSamples, testingLabels, verbose = 2)
print("")

print("Getting probability model ...")
print("")
probabilityModel = getProbabilityModel(model)

print("Computing predictions ...")
print("")
predictions = probabilityModel.predict(testingSamples)
predictedLabels = numpy.argmax(predictions, axis = 1)
