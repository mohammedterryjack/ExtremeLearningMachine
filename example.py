############   NATIVE IMPORTS  ###########################
############ INSTALLED IMPORTS ###########################
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
############   LOCAL IMPORTS   ###########################
from src.extreme_learning_machine import ELMClassifier
from src.utils import decoder, encoder
##########################################################

dataset = load_digits()

X_train, X_test, y_train_classes, y_test_classes = train_test_split(
    dataset.images.reshape((len(dataset.images), -1)), 
    dataset.target, 
    test_size=0.5, 
    shuffle=False
)

y_train = encoder(
    class_indexes=y_train_classes,
    n_classes=max(y_test_classes)
)
classifier = ELMClassifier(input_layer_width=X_train.shape[-1])

#classifier.fit(inputs=X_train,outputs=y_train)
classifier.load_trained_weights()

y_predicted = classifier.infer(inputs=X_test)

y_predicted_classes = decoder(y_predicted)

print(confusion_matrix(y_test_classes, y_predicted_classes))