import csv
import numpy as np
from typing import Set,Tuple, List
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torchvision
NoneType = type(None)
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models import vgg11
from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms
import time

#EXERCISE #1 

## Answer to question 1: 
#  - The problem in the exercise is that Sets do not garantee that a order exists, so when trying to acess the index...

## Answer to question 2:
#  - There are 2 possible solutions, the first one is changing the set into a List, that auto implies a order into the elements. 
#  - So when trying to acess to a index, it should return the element at the correct index

#  - The second solution is creating a mapping to each element of the set, and by making sure that the mapping has the same 
#  - number of elements as the set, we can acess directly to the map to get the right index

def id_to_fruit(fruit_id: int, fruits: Set[str]) -> str:
    """
    This method returns the fruit name by getting the string at a specific index of the set.
    :param fruit_id: The id of the fruit to get
    :param fruits: The set of fruits to choose the id from
    :return: The string corrosponding to the index ``fruit_id``

    | ``1   It does not print the fruit at the correct index, why is the returned result wrong?``
    | ``2   How could this be fixed?``

    This example demonstrates the issue:
    name1, name3 and name4 are expected to correspond to the strings at the indices 1, 3, and 4:
    'orange', 'kiwi' and 'strawberry'..
    """

    fruit_mapping = ["apple", "orange", "melon", "kiwi", "strawberry"]
    if fruit_id < len(fruit_mapping):
        return fruit_mapping[fruit_id]
    else:
        raise RuntimeError(f"Fruit with id {fruit_id} does not exist")
    
def id_to_fruit_list(fruit_id: int, fruits: List[str]) -> str:
    """
    This method returns the fruit name by getting the string at a specific index of the list.
    
    :param fruit_id: The id of the fruit to get
    :param fruits: The list of fruits to choose the id from
    :return: The string corresponding to the index ``fruit_id``
    """
    if fruit_id < len(fruits):
        return fruits[fruit_id]
    else:
        raise RuntimeError(f"Fruit with id {fruit_id} does not exist")


name1 = id_to_fruit(1, {"apple", "orange", "melon", "kiwi", "strawberry"})
name3 = id_to_fruit(3, {"apple", "orange", "melon", "kiwi", "strawberry"})
name4 = id_to_fruit(4, {"apple", "orange", "melon", "kiwi", "strawberry"})

name5 = id_to_fruit_list(1, ["apple", "orange", "melon", "kiwi", "strawberry"])
name6 = id_to_fruit_list(3, ["apple", "orange", "melon", "kiwi", "strawberry"])
name7 = id_to_fruit_list(4, ["apple", "orange", "melon", "kiwi", "strawberry"])

print(name1)
print(name3)
print(name4)
print(name5)
print(name6)
print(name7)