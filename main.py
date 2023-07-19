from cgitb import text
from fastapi import FastAPI
import uvicorn
import pickle
import joblib
from flair.data import Sentence
from flair.models import SequenceTagger
from pydantic import BaseModel


app = FastAPI()

model = SequenceTagger.load("flair/ner-english-ontonotes-large")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/{text}")
def read_root(text: text):
    sentence = Sentence(text)
    model.predict(sentence)
   # for entity in sentence.get_spans('ner'):
       # print(entity)
    return sentence.get_spans('ner')



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)