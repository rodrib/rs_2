# recomendation_system1

# API de Recomendación de Películas

El sistema de recomendación se basa en el análisis de la sinopsis de las películas. 
Calculamos puntajes de similitud entre todas las películas en función de las descripciones de sus tramas y, posteriormente, recomendamos películas basadas en estos puntajes de similitud.
La información de la trama se obtiene de la función de descripción general presente en nuestro conjunto de datos.
En el presente caso usamos como metrica la similitud del coseno:

https://es.wikipedia.org/wiki/Similitud_coseno

# Dataset
Usamos 2 datasets los cuales fueron extraidos de aqui:

https://drive.google.com/drive/folders/1nvSjC2JWUH48o3pb8xlKofi8SNHuNWeu

Tambien estan los dataset parciales

https://drive.google.com/drive/folders/1-gruxBXBr8AUaDrX72f0UEDtJADOyuvV?usp=sharing


## Instalación

1. Clona este repositorio en tu máquina local.

git clone https://github.com/rodrib/recomendation_system1.git

2. Crea un ambiente virtual en Python.

python -m venv myenv


3. Activa el ambiente virtual.

source myenv/bin/activate


4. Instala las dependencias utilizando [pip](https://pip.pypa.io/en/stable/).

pip install -r requirements.txt


## Estructura del proyecto

El proyecto tiene la siguiente estructura:

recomendation_system1/
├── pycache/
├── rs/
├── static/
├── templates/
└── main.py


- La carpeta `recomendation_system1` es el directorio raíz del proyecto.
- `pycache/` es una carpeta generada automáticamente por Python y contiene archivos de caché.
- `rs/` contiene los archivos y módulos del sistema de recomendación.
- `static/` es la carpeta donde se almacenan los archivos estáticos, como hojas de estilo CSS.
- `templates/` contiene los archivos HTML utilizados para la interfaz del proyecto.
- `main.py` es el archivo principal que inicia la API y maneja las rutas y la lógica del sistema de recomendación.

## Transformaciones de los datasets y paso a paso:
Dan una idea del ETL,EDA y ML aplicado ademas las distintas notebooks y archivos que se usaron.
Esta en Miro como un mapa conceptual.

https://miro.com/welcomeonboard/THpvSVpPN3libHRxZU5vYnc4TFhEWU1FakxXREdUUklmWFdhd0U4MWNENncwVDVseWIzaWxKQnZ2M1k3d2RzYnwzMDc0NDU3MzU0NDEyMDg5MzgyfDI=?share_link_id=207722773558

## Notebooks usadas
RS1
https://colab.research.google.com/drive/1AjBtF6o23nrqFru_z_Ob9hT8yMAjUqCp?usp=drive_link

RS2
https://colab.research.google.com/drive/1qY1YLrkhSAur6TZPHwZyGthicfPNzvly?usp=drive_link

RS3
https://colab.research.google.com/drive/1amgSPMOVJDnZx_ICThwm0C8SncdBDwf3?usp=drive_link

EDA
https://colab.research.google.com/drive/1ocNMGdpKzw3B_17zp0-h_fp8Qhp0qdSN?usp=drive_link

RS_Ml
https://colab.research.google.com/drive/17hwh5wFHxIBw6bY88UrWrmFuadZXv0Dl?usp=drive_link



## Contribución

Si deseas contribuir a este proyecto, sigue estos pasos:

1. Haz un fork de este repositorio.

2. Crea una rama para tu contribución.

## Contacto

Si tienes alguna pregunta o sugerencia, no dudes en contactarme a través de mi correo electrónico: rodribogado50@gmail.com.


