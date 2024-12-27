# Interfaz Gráfica 

El **Express Generator** es una herramienta que te permite generar una estructura básica de un proyecto Express de manera rápida y organizada. Aquí te explico cómo usarlo para crear un servidor Express.

---

### **1. Instalar el Express Generator**

Primero, asegúrate de tener instalado Node.js y `npm`. Luego, instala el Express Generator de manera global:

```bash
npm install -g express-generator
```

---

### **2. Crear un proyecto con Express Generator**

Ejecuta el siguiente comando para generar un nuevo proyecto. Por ejemplo, si tu proyecto se llama `mi-app`:

```bash
npm install -g express-generator
```

---

### **3. Opciones del Express Generator**

Puedes personalizar el proyecto al generarlo con algunas opciones útiles:

- **Usar el motor de plantillas EJS** (por defecto usa Jade/Pug):  
  ```bash
  express --view=ejs interfazChestGAN
  ```

- **Incluir archivos CSS con Bootstrap o Sass:**  
  - Para usar CSS normal (por defecto):  
    ```bash
    express mi-app
    ```
  - Para usar Sass:  
    ```bash
    express --css=sass interfazChestGAN
    ```
  - Para usar LESS:  
    ```bash
    express --css=less interfazChestGAN
    ```

- **Evitar generar el archivo `public` con los archivos estáticos:**  
  ```bash
  express --no-public mi-app
  ```

Consulta más opciones con:
```bash
express --help
```

---

### **4. Instalar dependencias**

Después de generar el proyecto, accede al directorio del proyecto:

```bash
cd interfazChestGAN
```

Instala todas las dependencias definidas en el `package.json`:

```bash
npm install
```

```bash
npm install express-session
npm install ejs
npm install bcrypt sqlite3
```

---

### **5. Iniciar el servidor**

Ejecuta el siguiente comando para iniciar el servidor:

```bash
npm start
```

Por defecto, el servidor se ejecutará en [http://localhost:3000](http://localhost:3000).

---

### **6. Estructura del proyecto generado**

El Express Generator crea una estructura como esta:

```
mi-app/
├── app.js             # Archivo principal de la aplicación
├── bin/
│   └── www            # Archivo para inicializar el servidor
├── package.json       # Información del proyecto y dependencias
├── public/            # Archivos estáticos (CSS, JS, imágenes)
├── routes/            # Definición de rutas
│   ├── index.js       # Ruta principal
│   └── users.js       # Ruta adicional
├── views/             # Plantillas (EJS, Pug, etc.)
│   ├── error.ejs      # Página de error
│   └── index.ejs      # Página principal
```

---

