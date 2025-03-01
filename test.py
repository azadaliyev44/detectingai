import json
from similarity_scorer import SimScore
from sklearn.metrics import roc_auc_score

# CUDA requred, for cpu - change cuda to cpu
simscore = SimScore(device="cuda:0", checkpoint="facebook/bart-large-cnn")

input_file = "revised_texts.json"

human_original = []
human_revised = []
llm_original = []
llm_revised = []


with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Procesing dataset
for row in data:
    original_text = row.get("text", "")
    revised_text = row.get("revised_text", "")
    label = row.get("label", "")

    if label == "human":
        human_original.append(original_text)
        human_revised.append(revised_text)
    elif label == "llm":
        llm_original.append(original_text)
        llm_revised.append(revised_text)

# similarity calculation
chatgpt_scores = simscore.score(llm_revised, llm_original)
human_scores = simscore.score(human_revised, human_original)

# AUROC
y_true = [1] * len(chatgpt_scores) + [0] * len(human_scores)
y_score = chatgpt_scores + human_scores

# AUROC
auroc_score = roc_auc_score(y_true, y_score)

print(f"AUROC Score: {auroc_score}")
