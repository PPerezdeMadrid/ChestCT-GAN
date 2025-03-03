// services/aws.js
require('dotenv').config();
const AWS = require('aws-sdk');

// Configurar AWS con las credenciales desde el archivo .env
AWS.config.update({
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  region: process.env.AWS_REGION
});

const s3 = new AWS.S3();

// Función para listar los archivos en S3 dentro de la carpeta 'evaluation_dcgan'
const listFiles = async () => {
  const params = {
    Bucket: 'tfg-chestgan-bucket', // Reemplaza con el nombre de tu bucket
    Prefix: 'evaluation_dcgan/', // Carpeta que contiene los archivos
  };

  try {
    const data = await s3.listObjectsV2(params).promise();
    return data.Contents.map(file => file.Key);
  } catch (err) {
    console.error("Error listando los archivos", err);
    return [];
  }
};

// Generar una URL prefirmada (para archivos privados)
const getPresignedUrl = (key) => {
  const params = {
    Bucket: 'tfg-chestgan-bucket',
    Key: key,
    Expires: 60 * 60 // URL válida por 1 hora
  };

  return s3.getSignedUrl('getObject', params);
};

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
  
/* Función para obtener la URL firmada de una imagen */
function getS3ImageUrl(bucketName, imagePath) {
    const params = {
      Bucket: bucketName,
      Key: imagePath,
      Expires: 180 // Tiempo de expiración en segundos para la URL firmada
    };
  
    // Generar la URL firmada (si las imágenes son privadas) o usar la URL pública si es accesible
    return s3.getSignedUrl('getObject', params);
  }
  
/* Función principal para obtener las URLs firmadas de todas las imágenes en la carpeta */
async function getAllImageUrls(bucketName, folderPaths) {
  let allImages = [];

  // Obtener imágenes de cada carpeta
  for (const folderPath of folderPaths) {
      const images = await listImagesInFolder(bucketName, folderPath);
      allImages = allImages.concat(images);
  }

  // Filtrar solo imágenes que contengan ".png" en el nombre
  allImages = allImages.filter(image => image.Key.includes('.png'));

  // Obtener las URLs firmadas de todas las imágenes
  let imageUrls = allImages.map(image => getS3ImageUrl(bucketName, image.Key));

  // Mezclar aleatoriamente las imágenes
  imageUrls = imageUrls.sort(() => Math.random() - 0.5);

  return imageUrls;
}


  

module.exports = { listFiles, getPresignedUrl, getAllImageUrls };

