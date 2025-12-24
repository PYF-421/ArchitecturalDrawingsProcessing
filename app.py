# app.py
from flask import Flask, jsonify
app = Flask(__name__)   # 变量名必须是 app

@app.route("/")
def index():
    return jsonify({"message": "healthy"})


if __name__ == "__main__":
    app.run(debug=True)
