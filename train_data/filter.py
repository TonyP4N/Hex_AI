import os.path
import re

documents = []

path = os.path.realpath("")
files = os.listdir(path)
print(files)

for file in files:
    with open(f'train_data/{file}', "r") as f:
        document = ""
        for line in f:
            if re.search("SZ\[13\]", line):
                document += line
        if document:
            documents.append(document)

with open("train_dataset_test", "w") as f:
    for doc in documents:
        f.write(doc + "\n")

