from flask import Flask, request
from flask_restful import Resource, Api
import query_document_match

app = Flask(__name__)
api = Api(app)

matcher = query_document_match.QueryDocumentMatcher()

class HelloWorld(Resource):
    def post(self):
        question = request.form['question']
        top_doc, _ = matcher.query_document_match(question)
        return {'answer': top_doc}

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)