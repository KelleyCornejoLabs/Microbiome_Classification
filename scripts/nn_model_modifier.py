import torch
import nn_classifier

def getOpt(prompt: str, options: list[str]):
    while True:
        opt = input(f"{prompt} ({'/'.join(options)}): ")

        if opt in options:
            return opt
        
        print("Invalid option")



# Load model
while True:
    # Prompt user for the path
    path = input("Please input the path to the model (q or quit to quit): ")

    # Handle quit
    if path == "quit" or path == "q":
        exit(0)

    # Try to load the model, continue if successful
    try:
        model_info = torch.load(f=path)
        break
    except FileNotFoundError:
        print("Invalid path")

requried_keys = ["model", "structue", "features", "all_labels"]
list_intersection = lambda x,y:[z for z in x if z in y]

# Get all the dictionary elements and print their names
keys = list(model_info.keys())
print(f"All keys: {', '.join(keys)}")
print(f"Found required: {', '.join(list_intersection(requried_keys, keys))}")

def determine_structure(model_dict: dict):
    # Get the neurons per layer for every layer
    weights = [model_dict.get(layer) for layer in keys if '.weight' in layer]
    print(weights[0].size())
    neurons_per_layer = [str(weights[0].size()[1])] + [str(len(weight)) for weight in weights]
    print(f"Layers: {', '.join(neurons_per_layer)}")

    if len(neurons_per_layer) == 5:
        structure = "OLD"
    elif len(neurons_per_layer) == 4:
        structure = "NEW"
    else:
        structure = "None"
        print("Cannot determine type")

    print("Note: hidden layers only matter for NEW type, and linearity cannot be auto-detected")
    print(f"Detected {structure} <linearity> {neurons_per_layer[0]} -> {neurons_per_layer[1]} -> {neurons_per_layer[-1]}")

    modify = getOpt("Override model type?", ["yes", "y", "no", "n"])

    modify = modify == "yes" or modify == "y"

    if modify:
        structure = getOpt("What is the type?", ["NEW", "OLD"])

    linearity = getOpt("Is the model linear?", ["yes", "y", "no", "n"])

    linearity = linearity == "yes" or linearity == "y"

    structure += f" {'Linear' if linearity else 'Non-linear'} {neurons_per_layer[0]} -> {neurons_per_layer[1]} -> {neurons_per_layer[-1]}"
    return structure


structure = model_info.get("structue")
features = model_info.get("features")
all_labels = model_info.get("all_labels")
optim = model_info.get("optim")
optim_type = model_info.get("optim_type")
lr = model_info.get("lr")
model = model_info.get("model")

# If no required keys are found the model is veeeeery old and just stores model dict
if list_intersection(requried_keys, keys) == []:
    print("\nStructure not found.")
    model = model_info
    structure = determine_structure(model_info)
elif structure == None or len(structure.split(" ")) != 7:
    model = model_info.get("model")
    print("\nStructure invalid")
    if structure != None:
        print(f"Structure length {len(structure.split(' '))}")
    structure = determine_structure(model)

def print_attribs():
    print()
    print("===Critical=Attributes===")
    print(f"{model.keys()}")
    print(f"Model: {'OK' if model != None else 'BAD'}")
    print(f"Structure: {structure}")
    print("===Corectness=Attributes===")
    print(f"Features: [{', '.join(features) if features is not None else 'UNKNOWN'}]")
    print(f"All classes: [{', '.join(all_labels) if all_labels is not None else 'UNKNOWN'}]")
    print("===Training=Attributes===")
    print(f"Opttimizer type: {optim_type if optim_type != None else 'UNKNOWN'}")
    print(f"Opttimizer dict: {'OK' if optim != None else 'BAD'}")
    print(f"Learning rate: {lr if lr != None else 'UNKNOWN'}")

print_attribs()
        

# TODO: Figure out features and labels based on training data, allow user to modify

path = input("Where should new model be stored? ")

torch.save({"model": model,
            "structue": structure,
            "features": None,
            "all_labels": None,
            "optim_type": None,
            "lr": 0,
            "optim": None}, f=path)
