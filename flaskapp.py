from flask import Flask, request, jsonify
from chatbot import prepare_rag_pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from the frontend

# Initialize the RAG pipeline (assuming mydata.txt is in the same directory)
qa_chain = prepare_rag_pipeline("mydata.txt")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Run the query through the RAG pipeline
        response = qa_chain.run(query)
        # Extract the answer after "Answer:" and remove duplicates
        answer_start = response.find("Answer:") + len("Answer:")
        answer = response[answer_start:].strip().split("\n")[0]
        
        return jsonify({'response': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)