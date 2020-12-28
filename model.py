import tensorflow

def computeModel(trainingSamples, trainingLabels, numberOfSpeakers):
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Flatten(input_shape = trainingSamples.shape[1:]),
        tensorflow.keras.layers.Dense(128, activation = "relu"),
        tensorflow.keras.layers.Dense(numberOfSpeakers)
    ])

    model.compile(optimizer = "adam",
                  loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ["accuracy"])

    model.fit(trainingSamples, trainingLabels, epochs = 10)

    return model

def getProbabilityModel(model):
    return tensorflow.keras.Sequential([model, tensorflow.keras.layers.Softmax()])
