# @Time : 2018/8/20 23:02 
# @Author : Chicharito_Ron
# @File : predapi.py 
# @Software: PyCharm Community Edition

from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api
from dispute import DisputePrediction
import json

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 跨域
api = Api(app)


class Predict(Resource):
    def get(self):
        return '纠纷预测API'

    def post(self):
        db_url = request.form.get('dbbase')

        dp = DisputePrediction(db_url)
        pres = dp.main()

        return '更新成功'


api.add_resource(Predict, '/disputepred')

if __name__ == '__main__':
    app.run(port=6666, debug=True)
