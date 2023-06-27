from flask import Flask, render_template, request, redirect
import pickle
import sklearn
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':

        with open('knn_pickle', 'rb') as r:
            model = pickle.load(r)

        banyak_melahirkan = float(request.form['banyak_melahirkan'])
        kadar_glukosa = float(request.form['kadar_glukosa'])
        tekanan_darah = float(request.form['tekanan_darah'])
        tebal_kulit = float(request.form['tebal_kulit'])
        kadar_insulin = float(request.form['kadar_insulin'])
        bmi = float(request.form['bmi'])
        riwayat_diabetes = float(request.form['riwayat_diabetes'])
        umur = float(request.form['umur'])                                                                                  

        datas = np.array((banyak_melahirkan, kadar_glukosa, tekanan_darah, tebal_kulit, kadar_insulin, bmi, riwayat_diabetes, umur))
        datas = np.reshape(datas, (1, -1))

        isDiabetes = model.predict(datas)


        return render_template('hasil.html', finalData=isDiabetes)
    
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)