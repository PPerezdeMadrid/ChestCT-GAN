var express = require('express');
var router = express.Router();
const sqlite3 = require('sqlite3').verbose();

/* GET users listing. */
router.get('/', function(req, res, next) {
  // res.render('contact', {  username: req.session.username  });
  res.render('contact', {  user: req.session.user });
});


// Conectar a la base de datos
const db = new sqlite3.Database('./database.db', (err) => {
  if (err) {
    console.error('Error al conectar con la base de datos:', err.message);
  } else {
    console.log('Conexión exitosa a la base de datos.');
  }
});


router.post('/messageSend', (req, res) => {
  const { nombre, email, mensaje } = req.body;

  // Validar que los campos no estén vacíos
  if (!nombre || !email || !mensaje) {
    return res.render('contact', { 
      message: '✨ ¡Ups! Todos los campos son requeridos. Por favor, revisa el formulario. ✨' 
    });
  }

  // Insertar el mensaje en la base de datos
  const insertMessageQuery = `
    INSERT INTO mensajes_contacto (nombre, email, mensaje)
    VALUES (?, ?, ?)
  `;

  db.run(insertMessageQuery, [nombre, email, mensaje], function (err) {
    if (err) {
      console.error('Error al insertar el mensaje:', err.message);
      return res.render('contact', { 
        message: '🌟 ¡Oh no! Algo salió mal al procesar tu mensaje. Por favor, intenta nuevamente más tarde. 🌟'
      });
    }

    console.log(`Mensaje de ${nombre} guardado con éxito.`);
    res.render('contact', { 
      message: '✨ ¡Gracias por tu mensaje! Te responderemos pronto. ✨' 
    });
  });
});

module.exports = router;
