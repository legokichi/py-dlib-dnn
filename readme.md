## setup

```
git submodule init
git submodule update

sudo apt-get install -y libopenblas-base libopenblas-dev liblapack-dev

source build.sh

source ./download.sh
pip install gunicorn flask

```

## gnicorn

```
gunicorn -w 4  -b 0.0.0.0:8888  server:app
```

## flask

```
env FLASK_APP=server.py flask run --host=0.0.0.0 --port 8888
```



