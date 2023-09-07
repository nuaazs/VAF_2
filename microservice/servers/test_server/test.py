import numpy as np
from scipy.spatial.distance import cosine


for i in range(10):
    try:
        a=2
        if a==1:
            print("a")
        else:
            continue
    except Exception as e:
        print(e)
    finally:
        print("finally")