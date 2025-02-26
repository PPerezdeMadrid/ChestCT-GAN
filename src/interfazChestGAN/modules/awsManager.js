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

module.exports = { listFiles, getPresignedUrl };
