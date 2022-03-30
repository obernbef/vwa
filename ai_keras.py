import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical

from game import GameState, Board


class ConnectFourModel:
    def __init__(
        self,
        number_pf_inputs: int,
        number_of_outputs: int,
        batch_size: int,
        epochs: int,
    ) -> None:
        self.number_pf_inputs: int = number_pf_inputs
        self.number_of_outputs: int = number_of_outputs
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.model = self.create_model()
        # self.model.compile(loss='categorical_crossentropy',
        #                    optimizer="rmsprop",
        #                    metrics=['accuracy'])

    def create_model(self) -> Sequential:
        model: Sequential = Sequential()
        model.add(Dense(42, activation=activations.sigmoid, input_shape=(self.number_pf_inputs, )))
        model.add(Dense(42, activation=activations.sigmoid))
        model.add(Dense(42, activation=activations.sigmoid))
        model.add(Dense(42, activation=activations.sigmoid))
        model.add(Dense(42, activation=activations.sigmoid))
        model.add(Dense(self.number_of_outputs, activation='softmax'))

        return model

    def train(self, dataset: list[tuple[GameState, Board]]) -> None:
        inputs: list[Board] = []
        outputs: list[GameState] = []
        for datapoint in dataset:
            inputs.append(datapoint[1])
            outputs.append(datapoint[0].numerator)

        X = np.array(inputs).reshape((-1, self.number_pf_inputs))
        Y = to_categorical(outputs, num_classes=3)
        limit = int(0.8 * len(X))
        X_train = X[:limit]
        X_test = X[limit:]
        Y_train = Y[:limit]
        Y_test = Y[limit:]
        self.model.fit(
            X_train,
            Y_train,
            validation_data=(X_test, Y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
        )

    def predict(self, board: Board, index: int):
        return self.model.predict(np.array(board).reshape(-1, self.number_pf_inputs))[0][index]