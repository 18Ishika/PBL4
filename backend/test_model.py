from transformers import pipeline

# Load pretrained model
classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-mini-finetuned-age_news-classification"
)

# Test samples
titles = [
    "YOU WON'T BELIEVE What Happened Next 😱🔥",
    "Introduction to Data Structures in Java",
    "This One Trick Will Make You Rich Overnight",
    "How to Solve BFS Problems in Graphs"
]

# Predict
for title in titles:
    result = classifier(title)[0]
    print("Title:", title)
    print("Prediction:", result["label"])
    print("Confidence:", round(result["score"]*100, 2), "%")
    print("-"*40)