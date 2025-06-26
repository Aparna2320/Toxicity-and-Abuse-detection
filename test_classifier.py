from app.classifier import predict_toxicity

text = "You are disgusting and dumb"
result = predict_toxicity(text)

print(result)

