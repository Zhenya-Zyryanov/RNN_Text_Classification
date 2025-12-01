import torch
from torch.utils.data import DataLoader
from RNN_model import RNN, TextDataset, clean_text, tokenize
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(23)
language = 'english'

lstm_df = pd.DataFrame(columns=["Accuracy", "F1_score", "Сред. длинна текста", "Количество классов", "Эпох обучения"])
lstm_df.index.name = "Датасет"
gru_df = pd.DataFrame(columns=["Accuracy", "F1_score", "Сред. длинна текста", "Количество классов", "Эпох обучения"])
gru_df.index.name = "Датасет"
rnn_df = pd.DataFrame(columns=["Accuracy", "F1_score", "Сред. длинна текста", "Количество классов", "Эпох обучения"])
rnn_df.index.name = "Датасет"

global_accuracy = []
global_f1_score = []
rnn_types = ["LSTM", "GRU", "RNN"]
datasets = [("hf://datasets/architrawat25/Balanced_hate_speech18/balanced_dataset.csv", "text", "label", "Анализ ненависти в тексте"),  # 13 bin
            ("hf://datasets/jahjinx/IMDb_movie_reviews/IMDB_train.csv", "text", "label", "Положительные & Отрицательные отзывы"), # 118 bin
            ("hf://datasets/galileo-ai/20_Newsgroups_Fixed/train.csv", "text", "label", "Определение темы новостей"),  # 118 mult 10k
            ("Datasets/Text Multiclassification.csv", "Text", "Label", "Определение темы текста №1"),  # 219 mult
            ("hf://datasets/Ateeqq/AI-and-Human-Generated-Text/train.csv", "abstract", "label", "Human vs AI"), # 120 bin 21k
            ("hf://datasets/valurank/Topic_Classification/topic_classification.csv", "article_text", "topic", "Определение темы текста №2" ), # 462 mult 22k
            ("hf://datasets/Faith1712/Allsides_political_bias_proper/allsides_data_unstructured.zip", "text", "label", "Политическая предвзятость текста")]  # 488 mul 11k
for rnn_type in rnn_types:
    for dataset , text, label, name in datasets:
        df = pd.read_csv(dataset)
        df = df.dropna(subset=[label])
        train_df, test_df = train_test_split(df, test_size=0.33, random_state=42, stratify=df[label])

        le = LabelEncoder()
        le.fit(train_df[label].values)

        train_texts = train_df[text].fillna("").values
        train_cleaned = [clean_text(t) for t in train_texts]
        train_tokenized = [tokenize(t) for t in train_cleaned]
        all_tokens = [tok for doc in train_tokenized for tok in doc]

        freq = Counter(all_tokens)
        V = 1000000
        most_common = [w for w, _ in freq.most_common(V)]
        word2idx = {w: i + 2 for i, w in enumerate(most_common)}
        word2idx['<PAD>'] = 0
        word2idx['<UNK>'] = 1
        vocab_size = len(word2idx)

        lengths = [len(doc) for doc in train_tokenized]
        max_len = int(np.percentile(lengths, 98))
        if max_len < 15:
            max_len = 15


        train_ds = TextDataset(train_df[text].fillna("").values,
                               train_df[label].values,
                               word2idx, le, max_len)

        test_ds = TextDataset(test_df[text].fillna("").values,
                              test_df[label].values,
                              word2idx, le, max_len)

        num_classes = len(le.classes_)


        max_threshold = max_len + 40
        lengths_stat = [length for length in lengths if length <= max_threshold]
        k = int(1 + 3.332 * np.log2(len(lengths_stat)))
        n, bins, patches = plt.hist(lengths_stat, bins=k, alpha=0.7, color='blue', edgecolor='black')

        first_quantile = int(np.percentile(lengths_stat, 25))
        second_quantile = int(np.percentile(lengths_stat, 50))
        third_quantile = int(np.percentile(lengths_stat, 75))
        mean = int(np.round(np.mean(lengths_stat)))

        plt.axvline(first_quantile, color='brown', linestyle='--', linewidth=1, label=f'25%: {first_quantile}')
        plt.axvline(second_quantile, color='orange', linestyle='--', linewidth=1, label=f'50%: {second_quantile}')
        plt.axvline(third_quantile, color='red', linestyle='--', linewidth=1, label=f'75%: {third_quantile}')
        plt.axvline(mean, color='purple', linestyle='-', linewidth=2, label=f'Среднее: {mean}')

        plt.axvline(max_len, color='darkred', linestyle='-', linewidth=2, label=f'Макс. длинна 98%: {max_len}')

        bin_edges = bins
        right_bin_index = np.searchsorted(bin_edges, max_len, side='right')

        for i in range(right_bin_index, len(patches)):
            patches[i].set_facecolor('red')
            patches[i].set_alpha(0.8)

        plt.xlabel('Длина последовательности')
        plt.ylabel('Количество')
        plt.title(f'Распределение датасета "{name}"')
        plt.legend()

        stats_text = f'Общее количество: {len(lengths_stat)}\nMin: {min(lengths_stat)}\nMax: {max(lengths_stat)}'
        plt.annotate(stats_text, xy=(0.6205, 0.965), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                     verticalalignment='top')

        plt.legend(loc='upper left', bbox_to_anchor=(0.543, 0.83))

        plt.tight_layout()
        plt.savefig(f"Datasets_Distribution/{name}.png")


        batch_size = 32
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        embd_dim = 100
        lr = 1e-3
        rnn_hidden_size = 64
        num_layers = 2
        num_epochs = 10


        glove_input = "glove.6B.100d.txt"
        glove_word2vec_file = 'glove.6B.100d.word2vec.txt'
        glove2word2vec(glove_input, glove_word2vec_file)

        model = KeyedVectors.load_word2vec_format(glove_word2vec_file, binary=False)

        embedding_matrix = np.random.randn(len(word2idx), embd_dim)
        for word, idx in word2idx.items():
            if word in model:
                embedding_matrix[idx] = model[word]

        embedding_matrix_tensor = torch.tensor(embedding_matrix).to(device)


        model = RNN(vocab_size, embd_dim, rnn_hidden_size, num_classes, num_layers, rnn_type=rnn_type,
                    pretrained_embedding=embedding_matrix_tensor, freeze_embedding=False,
                    padding_idx=word2idx['<PAD>'], dropout=0.4, bidirectional=True)
        model.to(device)

        all_train_labels = []
        for _, labels in train_loader:
            all_train_labels.extend(labels.numpy())
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_train_labels), y=all_train_labels)
        weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
        max_grad_norm = 1.0

        best_acc = 0
        best_f1 = 0
        train_losses = []
        train_accuracies = []
        train_f1s = []
        val_losses = []
        val_accuracies = []
        val_f1s = []

        print(f"Обучение {rnn_type} сети на '{name}' датасете")
        for epoch in range(1, num_epochs + 1):
            print(f"\nЭпоха {epoch}/{num_epochs}")

            model.train()
            train_loss_sum = 0.0
            train_all_preds = []
            train_all_labels = []

            train_pbar = tqdm(train_loader, desc="Обучение", leave=False)
            for X_batch, y_batch in train_pbar:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                train_loss_sum += loss.item()
                preds = logits.argmax(dim=1)
                train_all_preds.extend(preds.cpu().numpy())
                train_all_labels.extend(y_batch.cpu().numpy())

            train_losses.append(train_loss_sum / len(train_loader))
            train_accuracies.append(accuracy_score(train_all_labels, train_all_preds))
            train_f1s.append(f1_score(train_all_labels, train_all_preds, average='micro', zero_division=0))


            model.eval()
            val_loss_sum = 0.0
            val_all_preds = []
            val_all_labels = []

            val_pbar = tqdm(val_loader, desc="Валидация", leave=False)
            with torch.no_grad():
                for X_val, y_val in val_pbar:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)

                    logits = model(X_val)
                    loss = criterion(logits, y_val)

                    val_loss_sum += loss.item()
                    preds = logits.argmax(dim=1)
                    val_all_preds.extend(preds.cpu().numpy())
                    val_all_labels.extend(y_val.cpu().numpy())

            val_losses.append(val_loss_sum / len(val_loader))
            val_accuracies.append(accuracy_score(val_all_labels, val_all_preds))
            val_f1s.append(f1_score(val_all_labels, val_all_preds, average='micro', zero_division=0))

            scheduler.step(val_losses[-1])

            print(f"Train loss: {train_losses[-1]:.2f}  Acc: {train_accuracies[-1]:.2f}  F1: {train_f1s[-1]:.2f} \n"
                  f" Val loss: {val_losses[-1]:.2f}  Acc: {val_accuracies[-1]:.2f}  F1: {val_f1s[-1]:.2f}")

        all_preds = []
        all_labels = []
        val_pbar = tqdm(val_loader, desc="Итоговая валидация", leave=False)
        with torch.no_grad():
            for x_batch, y_batch in val_pbar:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                preds = torch.argmax(y_pred, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        print("Результаты итоговой валидации:")
        print(f"Accuracy: {accuracy_score(all_labels, all_preds):.2f}")
        print(f"Max accuracy: {np.max(val_accuracies):.2f}, на {np.argmax(val_accuracies) + 1} эпохе")
        print(f"Max F1: {np.max(val_f1s):.2f}, на {np.argmax(val_f1s) + 1} эпохе")
        """print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, zero_division=0), '\n')"""
        plt.figure(figsize=(13, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Эпоха')
        plt.ylabel('Loss')
        plt.grid(axis='y', color='grey', linestyle='--', linewidth=0.65)
        plt.legend()

        epochs = len(train_losses)
        plt.xticks(range(0, epochs, 1), range(1, epochs + 1, 1))

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.plot(train_f1s, label='Train F1')
        plt.plot(val_f1s, label='Validation F1')
        plt.xlabel('Эпоха')
        plt.ylabel('Accuracy and F1')
        plt.legend()
        plt.grid(axis='y', color='grey', linestyle='--', linewidth=0.65)
        plt.xticks(range(0, epochs, 1), range(1, epochs + 1, 1))

        plt.suptitle(f'{rnn_type}_bindir на "{name}"')
        plt.tight_layout()
        plt.savefig(f"Data_Metrics/{rnn_type}_bidirectional сеть на  {name}.png")
        plt.close('all')

        if rnn_type == 'LSTM':
            lstm_df.loc[f"{name}"] = [val_accuracies[-1], val_f1s[-1], mean, num_classes, num_epochs]
        elif rnn_type == 'GRU':
            gru_df.loc[f"{name}"] = [val_accuracies[-1], val_f1s[-1], mean, num_classes, num_epochs]
        else:
            rnn_df.loc[f"{name}"] = [val_accuracies[-1], val_f1s[-1], mean, num_classes, num_epochs]


lstm_df.to_csv("LSTM_metrics_bidirectional.csv", index=True)
rnn_df.to_csv("RNN_metrics_bidirectional.csv", index=True)
gru_df.to_csv("GRU_metrics_bidirectional.csv", index=True)
