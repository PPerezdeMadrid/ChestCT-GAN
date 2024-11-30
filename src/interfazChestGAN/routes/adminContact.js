const express = require('express');
const router = express.Router();
const sqlite3 = require('sqlite3').verbose();


// Conectar a la base de datos
const db = new sqlite3.Database('./database.db');


router.get('/', (req, res) => {
  const getMessagesQuery = 'SELECT * FROM mensajes_contacto ORDER BY fecha DESC';
  
  db.all(getMessagesQuery, (err, rows) => {
    if (err) {
      console.error('Error al obtener los mensajes:', err.message);
      return res.render('adminViews/adminContact', {
        user: req.session.user,
        message: '🌟 ¡Oh no! Algo salió mal al cargar los mensajes. Inténtalo más tarde. 🌟'
      });
    }
    res.render('adminViews/adminContact', {
      user: req.session.user,
      messages: rows
    });
  });
});


// Borrar mensaje
router.post('/deleteMessage',  (req, res) => {
  const { id } = req.body;
  const deleteMessageQuery = 'DELETE FROM mensajes_contacto WHERE id = ?';

  db.run(deleteMessageQuery, [id], function(err) {
    if (err) {
      console.error('Error al borrar el mensaje:', err.message);
      return res.redirect('/adminContact');
    }
    console.log(`Mensaje con ID ${id} eliminado con éxito.`);
    res.redirect('/adminContact');
  });
});


module.exports = router;
