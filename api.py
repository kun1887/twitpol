from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def index():
    # meter o model
    #meter model.predict(...)
    return {'ok': False}

@app.get("/predict")
def predict(url):
    # ml_model + predict
    return {'Political afil': url}
