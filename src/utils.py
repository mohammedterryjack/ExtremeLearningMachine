############   NATIVE IMPORTS  ###########################
from typing import List
############ INSTALLED IMPORTS ###########################
from numpy import argmax
############   LOCAL IMPORTS   ###########################
##########################################################

def onehot(class_index:int, vector_size:int) -> List[float]:
    vector = [0.]*(vector_size+1)
    vector[class_index] = 1.
    return vector

def encoder(class_indexes:List[int],n_classes:int) -> List[List[float]]:
    return list(
        map(
            lambda class_index: onehot(class_index,n_classes),
            class_indexes
        )
    )

def decoder(vectors:List[List[float]]) -> List[int]:
    return list(map(argmax,vectors))