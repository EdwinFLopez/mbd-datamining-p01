# mbd-datamining-p01
Práctica 01 - Minería de Datos - Master Big Data

## Introducción

Práctica de minería de datos sobre 3 tipos de algoritmos de predicción, usando aprendizaje de máquinas:

* Machine learning con **SVC**: en script `predictor-ml.py`
* Ensemble con **RandomForest**: en script `predictor-en.py`
* DeepLearning con **DNN**: en notebooks `predictor-dl.ipynb` y con la última ejecución: `predictor-dl-outputs.ipynb`  

El objetivo de la predicción es evaluar si un canto de ave pertenece a alguna de las especies observadas
en el entrenamiento.

Se utilizan los datos de audio y de espectrografía **Western Mediterranean Wetlands Bird Dataset**,
descargable desde: https://zenodo.org/record/7505820#.Y8U4f3bMKUk

## Preparación

1. Antes de iniciar, se debe contar con git, y descargar el código del repositorio github.
2. Se debe iniciar un ambiente virtual de python para instalar las dependencias.
3. Se proporciona un script para descargar el dataset y los archivos de soporte.
4. Se debe ejecutar el script adecuado, según se quiera ver la ejecución de cada caso.

## Ejecución paso a paso

### Descargar repositorio

Se debe contar con git instalado localmente para descargar el repo: 

```shell
git clone git@github.com:EdwinFLopez/mbd-datamining-p01.git
```

En caso de no tener git instalado, se puede descargar un zip de la siguiente url:
https://github.com/EdwinFLopez/mbd-datamining-p01/archive/refs/heads/main.zip

### Instalar ambiente virtual

Para ejecutar los scripts, se deben proporcionar las dependencias adecuadas.
Con este propósito debemos crear un ambiente virtual.

#### Si se usa conda:

```shell
$ conda create -n practica-mdd python=3.12
$ conda activate practica-mdd
```
Las dependencias quedarán instaladas en el folder `~/.conda/practica-mdd`.

Más información en: https://www.anaconda.com/open-source

#### Usando python venv

Se debe entrar en el folder donde se descargó el repo.

```shell
$ cd mbd-datamining-p01
$ python -m venv venv
$ source venv/bin/activate
```

### Instalar dependencias

Una vez creado el ambiente virtual, se deben instalar las dependencias:

```shell
$ pip install -r requirements.txt
```

### Descargar dataset

Una vez activado el ambiente virtual y las dependencias instaladas,
procedemos a descargar el dataset. Los archivos zip serán descomprimidos 
automáticamente por el script.

```shell
$ python download-all.py
```
Los archivos del dataset quedarán ubicados bajo el folder `data`.

```shell
$ tree data
├── readme.txt
├── metadata.csv
├── audio_files.zip
├── spectrograms.zip
├── audio_files
│   └── ... 
└── spectrograms
    └── ... 
```

### Ejecutar los scripts de predicción

#### Machine learning con **Support Vector Machine (SVM)**

```shell
$ python predictor-ml.py
```
La primera vez que se ejecuta, crea un archivo del modelo `./data/birds_features.csv`.
Este archivo se usa para evitar el reprocesamiento de los audios para extraer los features.

#### Ensemble con **RandomForest**

```shell
$ python predictor-en.py
```

La primera vez que se ejecuta, crea un archivo del modelo `./data/birds_data_rf.csv`.
Este archivo se usa para evitar el reprocesamiento de los audios para extraer los features.

#### DeepLearning con **DNN**

En este caso, se utiliza un notebook jupyter. El notebook es `predictor-dl.ipynb`. En nuestro caso
utilizamos tanto PyCharm como VSCode para procesarlo y ejecutarlo.
El modelo entrenado se almacena en la ruta: `./data/DL_model_birds.h5`.

Para ver los últimos cálculos generados, se debe cargar el notebook `predictor-dl-outputs.ipynb`. 
