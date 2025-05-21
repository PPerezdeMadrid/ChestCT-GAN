### **📌 README - Seguridad en AWS para S3 Upload desde una VM**  

#### **🔹 1. Introducción**  
Este documento describe las medidas de seguridad implementadas para permitir que una máquina virtual (VM) local cargue archivos de manera segura en un bucket de **Amazon S3**, evitando problemas de credenciales expuestas y asegurando el acceso restringido a los recursos.

---

### **🔹 2. Medidas de Seguridad Implementadas**  

#### ✅ **1. Uso de IAM con permisos mínimos**  
- Se ha creado un usuario de **AWS IAM** con los permisos **mínimos** necesarios para subir archivos al bucket.  
- Se adjuntó la política `AmazonS3FullAccess` para restringir el acceso a solo escritura en S3.  
- Se evita otorgar permisos innecesarios (como `s3:DeleteObject` o `s3:GetObject`).

#### ✅ **2. Uso de Credenciales Seguras con AWS CLI**  
- **No se almacenan credenciales en el código fuente**.  
- Se configuraron las credenciales en la VM mediante **AWS CLI**:  
  ```bash
  aws configure
  ```
  Se ingresaron los siguientes datos:
  - `AWS Access Key ID`
  - `AWS Secret Access Key`
  - `Default region name`
  - `Default output format` (JSON recomendado)

#### ✅ **3. Validación de Credenciales en el Código**  
- Antes de intentar subir archivos, el código **valida si las credenciales son correctas**.  
- Se usa `s3_client.list_buckets()` para comprobar acceso y evitar intentos fallidos.  

#### ✅ **4. Manejo de Errores de Seguridad**  
- Si no hay credenciales, el código retorna un mensaje de aviso en lugar de intentar subir archivos.  
- Se capturan errores específicos como `NoCredentialsError` para evitar mensajes de error inseguros.  

---

### **🔹 3. Flujo Seguro de Carga de Archivos**  
1️⃣ **La VM usa credenciales almacenadas localmente en `~/.aws/credentials`** (nunca en el código).  
2️⃣ **El script Python valida las credenciales antes de subir archivos** a S3.  
3️⃣ **Si hay un problema con las credenciales, el proceso se detiene** y muestra un aviso.  
4️⃣ **Si todo está correcto, los archivos se suben a S3 en la carpeta especificada.**  

---

### **📌 README - AWS Security for S3 Upload from a VM**

#### **🔹 1. Introduction**  
This document describes the security measures implemented to allow a local virtual machine (VM) to securely upload files to an **Amazon S3** bucket, preventing exposed credentials and ensuring restricted access to resources.

---

### **🔹 2. Implemented Security Measures**  

#### ✅ **1. Use of IAM with Least Privilege**  
- An **AWS IAM** user has been created with the **minimum** permissions necessary to upload files to the bucket.  
- The `AmazonS3FullAccess` policy was attached, limiting access to write-only in S3.  
- Unnecessary permissions (like `s3:DeleteObject` or `s3:GetObject`) are avoided.

#### ✅ **2. Use of Secure Credentials with AWS CLI**  
- **No credentials are stored in the source code.**  
- Credentials were configured on the VM using **AWS CLI**:  
  ```bash
  aws configure
````

The following information was entered:

* `AWS Access Key ID`
* `AWS Secret Access Key`
* `Default region name`
* `Default output format` (JSON recommended)

#### ✅ **3. Credentials Validation in Code**

* Before attempting to upload files, the code **validates whether the credentials are correct**.
* `s3_client.list_buckets()` is used to check access and prevent failed attempts.

#### ✅ **4. Handling Security Errors**

* If no credentials are found, the code returns a warning message instead of attempting to upload files.
* Specific errors like `NoCredentialsError` are caught to avoid insecure error messages.

---

### **🔹 3. Secure File Upload Flow**

1️⃣ **The VM uses credentials stored locally in `~/.aws/credentials`** (never in the code).
2️⃣ **The Python script validates credentials before uploading files** to S3.
3️⃣ **If there is a problem with the credentials, the process stops** and displays a warning.
4️⃣ **If all is correct, files are uploaded to S3 in the specified folder.**

---


