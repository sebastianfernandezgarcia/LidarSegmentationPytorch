# TFT Sebastián Fernández García 👨‍🎓                                                                                        
## 📚Estudio y aplicación de arquitecturas de aprendizaje profundo para la segmentación y clasificación de nubesde puntos 3D 🚁☁️•••
                                                                                                                           
Este repositorio de **GitHub** tiene como finalidad alojar parte del código desarrollado por el estudiante **Sebastián Fernández García** durante su Trabajo Fin de Grado. Este trabajo consta de un estudio, aplicación, análisis y comparativas de diversas arquitecturas de aprendizaje profundo para la clasificación y segmentación de nubes de puntos 3D, así como métodos tradicionales y novedosos para procesar este tipo de datos.

## Breve descripción y flujo de trabajo:

**📁DATASETS**
1. *Datasets/Aerolaser/parte_dataset.py* -> Parte los raws originales .las en tamaño de segmento designado y guarda en .npy, no procesa, solo parte y guarda. Se pasa como parámetro la carpeta donde están los .las.

2. *Datasets/Aerolaser/procesa_dataset.py* -> Toma las nubes .npy particionadas y las normaliza, escala y voxeliza con parámetros designados. Puede normalizar entre 0 y 1, entre -1 y 1 y voxeliza con tamaño voxel 1 (ajustable) y modifica para poder guardar más de un punto por voxel hasta llegar a los puntos designados. Es necesario ejecutarlo sobre train, validation y test por separado. En test, el procesamiento es diferente porque guarda las transformaciones de las coordenadas para poder volver a las originales.

3. Los datasets no se encuentran en este repositorio al ser de uso privado e internos de la empresa colaboradora. Para hacer uso de ellos, contactar.
   
**🔧ENVIRONMENTS**
1. Contiene la exportación de los entornos con todos los paquetes usados. Con la sencilla orden de instalación, deja el entorno listo.

## **🌟POINTNET**
1. Cuenta con el código de ejecución de clasificación y segmentación con PointNet.

## **🌟POINTNET2**
1. Cuenta con el código de ejecución de clasificación y segmentación con PointNet++.

## **🌟RANDLANET**
1. *RandLA-Net/data.py* -> es el dataloader de los datos. Coge la carpeta donde se encuentran los .npy, los carga con tamaño de lote designado y devuelve puntos y etiquetas. Este será invocado en el entrenamiento para poder pasarle puntos al modelo. Recordar que la ruta se cambia en las últimas líneas. Se usa naive. Active learning da juego...

2. *RandLA-Net/trainAerolaser.py* -> Entrenamiento. La función *evaluate* sirve para validar cómo funciona el modelo. La función *train* es el núcleo principal. En ella:
   - Se definen los dataloaders desde *data.py*. Train y Val loader.
   - Posteriormente, se cuentan todas las etiquetas para obtener relativos.
   - *d_in* es el tamaño de entrada de la red (3 por xyz).
   - Se crea y define el modelo, se mueve a la GPU.
   - Posteriormente, se definen los pesos de la loss (inversamente proporcional a la aparición de clases) - jugar con esto.
   - En *args.load* se carga un modelo anterior y se reanuda el entrenamiento donde se quedó. O con un diccionario, se pueden modificar los parámetros del entrenamiento congelado.
   - Empieza el bucle de épocas. Se cargan puntos y etiquetas, se define la loss, el modelo predice, se calcula, etc. Si la época es múltiplo de 20, se reduce la tasa de aprendizaje (lr).
   - Se cogen puntuaciones y se envían a evaluar (validación). Se muestran métricas por pantalla. Se guarda el modelo.
   - Finalmente, está el *main* donde se generan los parámetros. Parámetros útiles: epochs, load, adam_lr, decimation (está como paper), datasampling (naive), scheduler_gamma, gpu 0 y workers 0. *savefreq* es la frecuencia con la que se guardan los pesos.
   - Los datasets irrelevantes se pasan en el interior del código por comodidad. BatchSize...
   - No se ha incluido el guardado solo cuando mejora ni el *early stopping* (se adjunta por separado). El código de *PointNet2* sí lo tiene, *RandLA-Net* no por experimentos. Pero se adjunta listo para usar si se requiere.
   -**Importante!!!** Dentro del código de *pytorchtools early stopping*, el modelo se guarda automáticamente si mejora. Si no, no hay que guardar nada fuera si se utiliza esta función.

3. *RandLA-Net/data_test.py* -> Es un dataloader como *data*, solo que para pruebas porque devuelve los puntos normalizados pero también los originales y sus etiquetas.

4. *RandLA-Net/testAerolaser.py* -> Evaluación por métricas y visualización, todo en uno. Archivo de registro donde se guardan las métricas. Se carga el peso que se va a usar. Se definen variables. En la función *metrics* están todas las métricas que se usan. Después se evalúa. Se guardan las métricas, los tiempos de inferencia, etc., y se realizan todos los cálculos. Se muestran por consola y se guardan en el archivo de registro. Finalmente, se procede a visualizar los resultados. Se pasan a las funciones los puntos originales con su clasificación, y estas funciones dibujan los puntos y se muestran en un visor de *Open3D*.

## ⚙️PRESENTACIÓN Y RESULTADOS
Imágenes, vídeos, pesos y métricas obtenidas de manera interna para la elaboración de los resultados, memoria y presentación.

## 🧪PRUEBAS_JUPYTER
Entorno experimental donde se realizaron distintas pruebas con código experimental, métodos, funciones y, en definitiva, una gran batería de pruebas.

# **Autor:** Sebastián Fernández García

- Github: [Sebastián Fernández García](https://github.com/sebastianfernandezgarcia)
- LinkedIn: [Sebastián Fernández García](https://www.linkedin.com/in/sebasfdezg/)

# Colaboradores 💪

### AEROLASER 🚁
- Web: [Aerolaser System](https://www.aerolaser.es/)
- LinkedIn: [Aerolaser System](https://www.linkedin.com/company/aerolaser-system-sl/mycompany/)

### ULPGC 📚
- Web: [Universidad de las Palmas de Gran Canaria](https://www.ulpgc.es/)
- LinkedIn: [Universidad de las Palmas de Gran Canaria](https://www.linkedin.com/school/universidad-de-las-palmas-de-gran-canaria/)

# Licencia de uso 📋
Este código se ha realizado en un ámbito privado empresarial, por lo que aun formando parte de un Trabajo Fin de Grado, se ruega no compartir bajo ningún concepto y no hacer ningún uso de el mismo, salvo los que se comprendan en el ámbito de las actividades de evaluación relacionadas con su defensa ante un tribunal de TFT.
