# Interfaz Gráfica de ChestGan con Node.js y Express

Este proyecto es una interfaz gráfica construida con Node.js y Express. Se utiliza para interactuar con los servicios de AWS y ofrecer funcionalidades específicas según los requisitos de la aplicación.

## Requisitos

- **Node.js**: Asegúrate de tener Node.js instalado en tu sistema. Puedes descargarlo desde [aquí](https://nodejs.org/).
- **AWS Credentials**: Necesitarás una cuenta de AWS para obtener las credenciales de acceso.

## Instalación

Sigue estos pasos para configurar y ejecutar el proyecto:

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/PPerezdeMadrid/ChestCT-GAN
   cd src/interfazChestGAN
   ```

2. **Instala las dependencias**:
   ```bash
   npm install
   ```

3. **Crea el archivo `.env`**:
   En la raíz del proyecto, crea un archivo llamado `.env` y agrega tus credenciales de AWS:
   ```plaintext
   AWS_ACCESS_KEY_ID=tu_access_key_id
   AWS_SECRET_ACCESS_KEY=tu_secret_access_key
   AWS_REGION=eu-west-3
   SESSION_SECRET=e8c9fc5a8108eff77ece3f8a08e07d08d23ff0489a019131bf96ccec6f2774389be984aebd4254e2ee611bade5d8f54a9edc915850ae9c161bc43317320ebcd9
   ADMIN_PASSWD=tu_password
   ```

4. **Ejecuta el servidor**:
   Una vez que hayas configurado las dependencias y el archivo `.env`, puedes iniciar el servidor con el siguiente comando:
   ```bash
   npm start
   ```

5. Abre tu navegador y ve a `http://localhost:3000` para ver la interfaz gráfica.

## Configuración

- **AWS_ACCESS_KEY_ID**: Tu clave de acceso de AWS.
- **AWS_SECRET_ACCESS_KEY**: Tu clave secreta de AWS.
- **AWS_REGION**: La región de AWS que deseas usar (en este caso, `eu-west-3`).
- **SESSION_SECRET**: Secreto para las sesiones, puede ser cualquiera.
- **ADMIN_PASSWD**: Contraseña para que se registre un administrador

## Uso

- Al iniciar el servidor, podrás interactuar con la interfaz gráfica, que estará conectada a los servicios de AWS configurados en tu entorno.


## Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).

