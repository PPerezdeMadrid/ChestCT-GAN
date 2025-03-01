var express = require('express');
var router = express.Router();
require('dotenv').config(); 
const sqlite3 = require('sqlite3').verbose();

const { getAllImageUrls } = require('../modules/awsManager'); 

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('lungCT', { user: req.session.user });
});


// Conectar a la base de datos SQLite
const db = new sqlite3.Database('./database.db', (err) => {
  if (err) {
      console.error('Error al conectar con la base de datos:', err.message);
  } else {
      console.log('Conexión exitosa a la base de datos.');
  }
});


/* GET URL images. */
router.get('/get-images', async (req, res) => {
  const bucketName = 'tfg-chestgan-bucket';
  const folderPaths = ['images_dcgan/', 'LungCT_real/']; // Ambas carpetas

  try {
    const imageUrls = await getAllImageUrls(bucketName, folderPaths);
    console.log(imageUrls)
    res.json(imageUrls);
  } catch (error) {
    console.error('Error obteniendo imágenes:', error);
    res.status(500).json({ error: 'No se pudieron obtener las imágenes' });
  }
});



/* POST guardar respuesta */
router.post('/save-response', (req, res) => {
  const { imageUrl, response } = req.body;
  const username = req.session.user.username;

  if (!username) {
      return res.status(401).json({ error: 'Usuario no autenticado' });
  }
  console.log('Response: ', response)
  console.log('URL: ', imageUrl)

  // Determinar si la imagen es real según la carpeta de origen
  let isReal = imageUrl.includes('LungCT_real/') ? 1 : 0;

  const query = `
      INSERT INTO eval_tomografias (username, image_url, response, is_real, created_at) 
      VALUES (?, ?, ?, ?, datetime('now'))
  `;

  db.run(query, [username, imageUrl, response, isReal], function (err) {
      if (err) {
          console.error('Error al guardar la respuesta:', err.message);
          return res.status(500).json({ error: 'Error en el servidor' });
      }
      res.status(200).json({ message: 'Respuesta guardada correctamente' });
  });
});



module.exports = router;
