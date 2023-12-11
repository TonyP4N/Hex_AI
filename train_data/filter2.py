with open("../train_dataset_test", "r") as f:
    lines = f.readlines()

    documents = []

    for line in lines:
        line = line.strip().split(";")
        documents.append(";".join(line[2:]))

    for i in range(len(documents)):
        documents[i] = documents[i].replace(")", "")

    while "" in documents:
        documents.remove("")

with open("filtered_dataset", "w") as f:
    for doc in documents:
        f.write(doc + "\n")
