# ml_regression

create env 

```bash
conda create -n venv python=3.9 -y
```

activate env
```bash
conda activate venv
```

install the requirements
```bash
pip install -r requirements.txt
```

```bash
git init
```
```bash
dvc init 
```
```bash
dvc add notebook/data/student.csv
```
```bash
git add .
```
```bash
git commit -m "data version initiated"
```
```bash
git add . && git commit -m "update Readme.md"
```
```bash
git remote add origin https://github.com/ataharsk10/ml_regression.git
git branch -M main
git push origin main
```

tox command :
```bash
tox
```
tox command for rebuilding :
```bash
tox -r 
```
pytest command  :
```bash
pytest -v
```

setup commands :
```bash
pip install -e . 
```

build own package commands :
```bash
python setup.py sdist bdist_wheel
```