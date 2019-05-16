""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    minimum = min(arr)
    maximum = max(arr)
    denom = maximum - minimum
    if denom == 0:
        return arr
    return list(map(lambda x: float(x-minimum)/denom, arr))

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)
