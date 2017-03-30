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
       
    files = request.files.getlist("files")

    print(files)

    _files = []
    for (i, file) in enumerate(files):
        name = ("/tmp/img%d.img" % i)
        file.save(name)
        _files.append(name)
    
    print(_files)

    
    print("processing...", _files)
    
    start = time.time()
    
    ret = det.detect(_files)
    
    elapsed = time.time() - start
    
    print(elapsed, "sec, ", ret)

    return app.response_class(
        response=json.dumps(ret),
        status=200,
        mimetype='application/json'
    )




