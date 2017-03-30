from flask import Flask, request, json
import dnn
import time

app = Flask(__name__, static_url_path='')

det = dnn.DNN("mmod_human_face_detector.dat")


@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/post', methods=['POST'])
def upload_file():
    if not request.method == 'POST':
        print("post it")
        return "[]"
       
    if 'file' not in request.files:
        print('No file part')
        return "[]"

    file = request.files['file']
    
    if file.filename == '':
        print('No selected file')
        return "[]"
    
    file.save("/tmp/img.img")
    
    print("processing...")
    
    start = time.time()
    
    ret = det.detect("/tmp/img.img")
    
    elapsed = time.time() - start
    
    print(elapsed, "sec, ", ret)

    return app.response_class(
        response=json.dumps(ret),
        status=200,
        mimetype='application/json'
    )




