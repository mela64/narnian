from nltk.tokenize import word_tokenize
import csv

# category labels are: ['people', 'chasing', 'communication', 'playing', 'food', 'health']

unique_labels = {}
stream_of_tokens = []
stream_of_labels = []

with open('dataset.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        text = row[0]
        labels = row[1:]

        for i, label in enumerate(labels):
            labels[i] = label.strip().lower()

        tokenized_text = word_tokenize(text)
        for token in tokenized_text:
            stream_of_tokens.append(token.strip().lower())
            stream_of_labels.append(labels)

        for label in labels:
            unique_labels[label] = True

# vocabulary
voc = {}
idx = 0
sorted_stream_of_tokens = sorted(stream_of_tokens)
for token in sorted_stream_of_tokens:
    if token not in voc:
        voc[token] = idx
        idx += 1

# convert each tokenized text to a list of indices
stream_of_token_indices = [voc[token] for token in stream_of_tokens]

# final data
print(voc)
print(stream_of_tokens)
print(stream_of_labels)
print(stream_of_token_indices)
print(unique_labels.keys())

csv_filename = 'stream_of_words.csv'

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    for i in range(len(stream_of_tokens)):
        row = [stream_of_tokens[i]] + stream_of_labels[i]  # combine file with the corresponding labels
        writer.writerow(row)
