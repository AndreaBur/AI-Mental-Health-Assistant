#libreria para dataset de emotions
from datasets import load_dataset
#libreria para tokens de BERT
from transformers import AutoTokenizer 


# Cargar el dataset de emociones
dataset = load_dataset("emotion")

# Mostrar resumen del dataset
print(dataset)

# Mostrar el primer ejemplo del dataset de entrenamiento
ejemplo = dataset["train"][0]
print(ejemplo)

# Obtener las etiquetas posibles
etiquetas = dataset["train"].features["label"].names
print("Etiquetas posibles:", etiquetas)

# Cargar el tokenizador de BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Función para tokenizar cada ejemplo
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

# Aplicar tokenización al dataset completo
tokenized_dataset = dataset.map(tokenize)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

# Cargar modelo BERT con 6 etiquetas (emociones)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=6
)

# Métrica personalizada: precisión (accuracy)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Definir entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Entrenar el modelo 🚀
trainer.train()




