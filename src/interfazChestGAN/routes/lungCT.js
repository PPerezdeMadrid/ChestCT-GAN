var express = require('express');
var router = express.Router();
require('dotenv').config(); 

const { getAllImageUrls } = require('../modules/awsManager'); 

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('lungCT', { user: req.session.user });
});

/* GET URL images. */
router.get('/get-images', async (req, res) => {
  const bucketName = 'tfg-chestgan-bucket';
  const folderPath = 'images_dcgan/'; // Carpeta dentro del bucket S3

  try {
    const imageUrls = await getAllImageUrls(bucketName, folderPath);
    res.json(imageUrls); // Devuelve las URLs de las imágenes como JSON
  } catch (error) {
    res.status(500).json({ error: 'No se pudieron obtener las imágenes' });
  }
});



module.exports = router;
