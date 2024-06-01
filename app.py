from flask import Flask, request
import torch
import model
import numpy as np
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

embedding_dim = 384
hidden_dim = 512
num_layers = 1
output_dim = 180
num_epochs = 100
learning_rate = 0.001

lstm_model = model.LSTM(embedding_dim, hidden_dim, num_layers, output_dim)
lstm_model.load_state_dict(torch.load('lstm.pt'))

app = Flask(__name__)

def GeneratePosesJSON(input):
    with torch.no_grad():
        processed_text = torch.tensor(sentence_model.encode(input), dtype=torch.float)
        output_poses = lstm_model(processed_text.unsqueeze(0))

        people = output_poses.cpu().detach().numpy().reshape(5, 18, 2).tolist()
        newPeople = []
        for person in people:
            newPerson = []
            for keypoints in person:
                newPerson.append([keypoints[0], keypoints[1], 1])
            newPeople.append(newPerson)

        data = np.array(newPeople).reshape(5, 54).tolist()
        formatted_data = []
        for person in data:
            formatted_data.append({ "pose_keypoints_2d": person })
        return { 'people': data, 'animals': [], 'canvas_width': 900, 'canvas_height': 300 }
    
@app.route('/')
def hello():
    return "Hello, World!"

@app.route('/generate')
def generatePose():
    text = request.args.get('text')
    data = GeneratePosesJSON(text)
    return data

if __name__ == '__main__':
    app.run()