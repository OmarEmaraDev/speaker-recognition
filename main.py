import numpy
import loader
import tensorflow as tf

(trainingSamples, trainingLabels), (testingSamples, testingLabels) = loader.loadMFCCData()
speakerNames = loader.getSpeakerNames()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = trainingSamples.shape[1:]),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(len(speakerNames))
])

model.compile(optimizer = "adam",
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ["accuracy"])

model.fit(trainingSamples, trainingLabels, epochs = 10)

testLoss, testAccuracy = model.evaluate(testingSamples,  testingLabels, verbose = 2)

probabilityModel = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probabilityModel.predict(testingSamples)

print(speakerNames[numpy.argmax(predictions[0])] == speakerNames[testingLabels[0]])
