import json
name = 'unet'
f = open('./hyperparameters/'+ name + '.json')
config = json.load(f)
print (config["dataset"]["pre-filtered"])