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


