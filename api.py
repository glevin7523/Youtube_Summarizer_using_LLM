import google.generativeai as genai

genai.configure(api_key="AIzaSyA2N47LKWm5HSEf2HK-cJ0ErwNW2d4EFqE")

models = genai.list_models()
for model in models:
    print(model.name)
