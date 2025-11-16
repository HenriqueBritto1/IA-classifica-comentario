import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

url = "https://docs.google.com/spreadsheets/d/17aHYyRNfbmde8bVOR_HX_BmNUEdkygPuaGO4lJj26jg/export?format=csv"
df = pd.read_csv(url)

classes = ["onca", "fake news", "caseiro"]

df = df[["comment_text"] + classes].copy()
df = df.dropna(subset=["comment_text"])

df = df.drop_duplicates(subset=["comment_text"])

def combine_labels(row):
    for coluna in classes:
        if row[coluna] != "não":
            return row[coluna]
    return None

df["label"] = df.apply(combine_labels, axis=1)
df = df.dropna(subset=["label"])

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["label_id"] = encoder.fit_transform(df["label"])

train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label_id"])
train_df, val_df = train_test_split(train_df, test_size=0.1765, random_state=42, stratify=train_df["label_id"])

from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW

tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

class CommentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_ds = CommentDataset(train_df.comment_text, train_df.label_id)
val_ds   = CommentDataset(val_df.comment_text, val_df.label_id)
test_ds  = CommentDataset(test_df.comment_text, test_df.label_id)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)
test_loader  = DataLoader(test_ds, batch_size=16)

class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :] 
        x = self.dropout(cls_embedding)
        return self.fc(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BERTClassifier(num_classes=len(encoder.classes_)).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()


train_losses = []
val_losses = []

for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(ids, mask)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_losses.append(total_loss / len(train_loader))

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(ids, mask)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}/10 | Train acc={train_acc:.3f} | Val acc={val_acc:.3f}")

plt.plot(train_losses, label="Treino")
plt.plot(val_losses, label="Validação")
plt.legend()
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Evolução do Loss")
plt.show()


model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for batch in test_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(ids, mask)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=encoder.classes_))


test_df = test_df.reset_index(drop=True)

for i in range(len(y_true)):
    if y_true[i] != y_pred[i]:
        print("TEXTO:", test_df.comment_text[i])
        print("REAL:", encoder.classes_[y_true[i]])
        print("PREVISTO:", encoder.classes_[y_pred[i]])
        print("-"*80)
