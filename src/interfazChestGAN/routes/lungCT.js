var express = require('express');
var router = express.Router();
require('dotenv').config(); 

const AWS = require('aws-sdk');
const s3 = new AWS.S3();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('lungCT', { user: req.session.user });
});

AWS.config.update({
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  region: process.env.AWS_REGION,
});

/* Listar imágenes en la carpeta "images" */
async function listImagesInFolder(bucketName, folderPath) {
  const params = {
    Bucket: bucketName,
    Prefix: folderPath, 
  };

  try {
    const data = await s3.listObjectsV2(params).promise();
    return data.Contents; // Lista de objetos (archivos) en el bucket
  } catch (error) {
    console.error("Error al obtener los objetos:", error);
    return [];
  }
}

// Función para obtener la URL firmada de una imagen
function getS3ImageUrl(bucketName, imagePath) {
  const params = {
    Bucket: bucketName,
    Key: imagePath,
    Expires: 60 // Tiempo de expiración en segundos para la URL firmada
  };

  // Generar la URL firmada (si las imágenes son privadas) o usar la URL pública si es accesible
  return s3.getSignedUrl('getObject', params);
}

/* GET URL images. */
router.get('/get-images', async (req, res) => {
  const bucketName = 'tfg-chestgan-bucket';
  const folderPath = 'images_dcgan/'; // Carpeta dentro del bucket S3

  try {
    const imageUrls = await getAllImageUrls(bucketName, folderPath);
    console.log(imageUrls)
    res.json(imageUrls); // Devuelve las URLs de las imágenes como JSON
  } catch (error) {
    res.status(500).json({ error: 'No se pudieron obtener las imágenes' });
  }
});

// Función principal para obtener las URLs firmadas de todas las imágenes en la carpeta
async function getAllImageUrls(bucketName, folderPath) {
  const images = await listImagesInFolder(bucketName, folderPath);

  // Obtener la URL firmada para cada imagen
  const imageUrls = images.map(image => {
    return getS3ImageUrl(bucketName, image.Key);
  });

  return imageUrls;
}

module.exports = router;
