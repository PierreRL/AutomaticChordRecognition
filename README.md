# Automatic Chord Recognition with Deep Learning 🎼

This is my 4th year undergraduate project at the University of Edinburgh on Automatic Chord Recognition. For more information please read `thesis/thesis.pdf` in this repo.  

## How to Run

You can set up the python environment with:

```
python -m venv chord-recognition
pip install -r requirements.txt
```

Once you have the data under `data/processed`, you train and evaluate a CRNN with

```
python src/run.py
```

This will output the model, training history and evaluation metrics to a folder under `experiments/`. To customise which model is trained and how, please check the arguments you can pass to `src/run.py`.

## Repo Guide

- `/src` contains the core python scripts to run. This contains sub-folders and files:
    - `/src/models`: models and decoder
    - `/src/data`: data loading and generation
    - `run.py`: trains and evaluates a model. 
    - `train.py`
    - `eval.py`

    - `utils.py`: contains many useful functions.
- `/data` should contain the data relevant to this project. The data can be made available at request by emailing *first[dot]last[at]gmail.com*.
- `/notebooks`: python notebooks for analysing the data and results.
- `/scripts`: relevant files for running experiments on a cluster environment
- `/thesis`: latex to build the dissertation

## Citation

Please cite this work if you use it. Bibtex:

```
@misc{LeadSheetTranscription,
    author = {Pierre Lardet},
    title = {Automatic Chord Recognition with Deep Learning},
    year = {2024},
    doi = {https://github.com/PierreRL/LeadSheetTranscription}
}
```

- [pierre.wiki](https://pierre.wiki) 
- [Linkedin](https://www.linkedin.com/in/pierrelardet/)
- [Github](https://github.com/PierreRL)