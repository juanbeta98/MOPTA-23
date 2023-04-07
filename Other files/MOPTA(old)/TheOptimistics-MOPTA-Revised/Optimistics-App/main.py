'Main Method'
from flask import Flask
from flask import request
import MOPTA_optimistics

app = Flask(__name__)

@app.route('/', methods=["GET"])

#connect AIMMS with the script that runs the H-SARA problem
def aimms_call():
    ourInput = request.get_json()
    MOPTA_optimistics.my_model(ourInput)
    return None

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000)
