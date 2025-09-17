from torchvision import datasets

dataset = datasets.ImageFolder(root="~/asl_alphabet_train/asl_alphabet_train")
print(dataset.classes)
print(len(dataset))
