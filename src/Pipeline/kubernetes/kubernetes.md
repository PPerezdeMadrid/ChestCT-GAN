# Guía para Ejecutar el Pipeline en Kubernetes con Docker

Este repositorio contiene un pipeline en Python, `main_pipeline.py`, que se puede ejecutar en un contenedor Docker y desplegar en un cluster de Kubernetes.

## Pasos

### 1. Preparar el Entorno Local
Asegúrate de tener Docker y Minikube instalados en tu máquina.

- **Instalar Docker:**
  Si no tienes Docker instalado, sigue las instrucciones en [Docker Install](https://docs.docker.com/get-docker/).

- **Instalar Minikube:**
  Sigue las instrucciones para instalar Minikube en [Minikube Install](https://minikube.sigs.k8s.io/docs/).

- **Clonar el Repositorio:**
  ```bash
  git clone https://github.com/PPerezdeMadrid/ChestCT-GAN
  ```

### 2. Crear el Dockerfile

Crea un `Dockerfile` en el directorio de tu proyecto que describa cómo construir la imagen de Docker para tu aplicación. A continuación se muestra un ejemplo de `Dockerfile` para un proyecto en Python:

```Dockerfile
# Usa una imagen base de Python
FROM python:3.12

# Configura el directorio de trabajo
WORKDIR /app

# Copia el contenido del directorio actual al contenedor
COPY . /app

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Comando para ejecutar el archivo principal
CMD ["python", "main_pipeline.py", "run"]
```

### 3. Crear el Archivo `requirements.txt`

Asegúrate de que el archivo `requirements.txt` contenga todas las dependencias necesarias para que tu pipeline funcione correctamente. Un ejemplo de archivo `requirements.txt` podría ser:

```
metaflow
pandas
torch
torchvision
numpy
scipy
scikit-image
matplotlib
lpips
tqdm
pydicom
reportlab
fpdf
```

### 4. Construir la Imagen Docker

Para construir la imagen Docker, abre una terminal y navega al directorio de tu proyecto donde se encuentran el `Dockerfile` y `requirements.txt`. Luego, ejecuta:

```bash
docker build -t metaflow-pipeline .
```

Este comando construirá la imagen Docker con el nombre `metaflow-pipeline`.

### 5. Subir la Imagen a Docker Hub

Para usar la imagen en Kubernetes, primero necesitas subirla a Docker Hub.

- **Iniciar sesión en Docker Hub:**
  
  Si aún no tienes una cuenta en Docker Hub, crea una en [Docker Hub](https://hub.docker.com/). Luego, inicia sesión desde la terminal con:

  ```bash
  docker login
  ```

- **Etiquetar la Imagen:**

  Etiqueta la imagen para Docker Hub (reemplaza `<tu-usuario>` con tu nombre de usuario de Docker Hub):

  ```bash
  docker tag metaflow-pipeline <tu-usuario>/metaflow-pipeline:latest
  ```

- **Subir la Imagen:**

  Sube la imagen etiquetada a Docker Hub:

  ```bash
  docker push <tu-usuario>/metaflow-pipeline:latest
  ```

Esto subirá la imagen a tu repositorio en Docker Hub.

### 6. Crear un Deployment en Kubernetes

Ahora que la imagen está en Docker Hub, puedes crear un deployment en Kubernetes para ejecutar el pipeline.

- **Iniciar Minikube:**
  
  Inicia Minikube para crear un cluster local de Kubernetes:

  ```bash
  minikube start
  ```

- **Crear el Archivo de Deployment de Kubernetes:**

  Crea un archivo `deployment.yaml` para definir el deployment de tu pipeline. A continuación se muestra un ejemplo de archivo `deployment.yaml`:

  ```yaml
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: chestgan-pipeline
  spec:
    replicas: 1
    selector:
      matchLabels:
        app: chestgan-pipeline
    template:
      metadata:
        labels:
          app: chestgan-pipeline
      spec:
        containers:
          - name: chestgan-pipeline
            image: pperezdem/metaflow-pipeline:latest
            resources:
              requests:
                memory: "4Gi"
                cpu: "2"
              limits:
                memory: "8Gi"
                cpu: "4"
  ```

- **Aplicar el Deployment a Kubernetes:**

  Ejecuta el siguiente comando para aplicar el archivo de deployment:

  ```bash
  kubectl apply -f deployment.yaml
  ```

- **Verificar que el Pod se Está Ejecutando:**

  Puedes verificar que tu pod se haya creado y esté corriendo con:

  ```bash
  kubectl get pods
  ```

### 7. Limpiar los Recursos de Kubernetes

Cuando termines de usar el cluster de Kubernetes, puedes eliminar los recursos creados con:

```bash
kubectl delete -f deployment.yaml
```