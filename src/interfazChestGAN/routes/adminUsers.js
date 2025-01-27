const express = require('express');
const router = express.Router();
const sqlite3 = require('sqlite3').verbose();


// Conectar a la base de datos
const db = new sqlite3.Database('./database.db');


router.get('/', (req, res) => {
  const getMessagesQuery = 'SELECT * FROM usuarios';
  
  db.all(getMessagesQuery, (err, rows) => {
    if (err) {
      console.error('Error al obtener los usuarios:', err.message);
      return res.render('adminViews/adminUsers', {
        user: req.session.user,
        message: '🌟 ¡Oh no! Algo salió mal al cargar los usuarios. Inténtalo más tarde. 🌟'
      });
    }
    res.render('adminViews/adminUsers', {
      user: req.session.user,
      messages: rows
    });
  });
});


router.get('/getUsers', (req, res) => {
  const getMessagesQuery = 'SELECT * FROM usuarios';

  db.all(getMessagesQuery, (err, rows) => {
    if (err) {
      console.error('Error al obtener los usuarios:', err.message);
      return res.status(500).json({ message: '🌟 ¡Oh no! Algo salió mal al cargar los usuarios. Inténtalo más tarde. 🌟' });
    }
    // Enviar los usuarios como JSON
    res.json(rows);
  });
});



// Borrar mensaje
router.post('/deleteUser',  (req, res) => {
  const { id } = req.body;
  const deleteMessageQuery = 'DELETE FROM usuarios WHERE id = ?';

  db.run(deleteMessageQuery, [id], function(err) {
    if (err) {
      console.error('Error al borrar el mensaje:', err.message);
      return res.redirect('/adminUsers');
    }
    console.log(`Mensaje con ID ${id} eliminado con éxito.`);
    res.redirect('/adminUsers');
  });
});


router.get('/editUser/:id', (req, res) => {
  const userId = req.params.id;
  const getUserQuery = 'SELECT * FROM usuarios WHERE id = ?';

  console.log()
  db.get(getUserQuery, [userId], (err, row) => {
    if (err) {
      console.error('Error al obtener el usuario:', err.message);
      return res.redirect('/adminUsers');
    }
    if (!row) {
      return res.redirect('/adminUsers'); // Si no se encuentra el usuario
    }
    res.render('adminViews/adminUsers', {
      user: req.session.user,
      userData: row
    });
  });
});


router.post('/editUser', (req, res) => {
  const { name, username, email, num_colegiado, is_admin: is_admin, id: userId } = req.body; 

    console.log('Datos recibidos para editar el usuario:', {
        userId, 
        name, 
        username, 
        email, 
        num_colegiado, 
        is_admin
    });

  // Verificar si el username o email ya existen en la base de datos
  const checkUserQuery = `SELECT * FROM usuarios WHERE (username = ? OR email = ?) AND id != ?`;

  db.get(checkUserQuery, [username, email, userId], (err, row) => {
    if (err) {
      console.error('Error al verificar la existencia de usuario:', err.message);
      return res.status(400).send({error: 'Error al verificar la existencia de usuario'});
    }

    if (row) {
      // return res.status(400).json({ error: 'El email o el nombre de usuario ya existe, debe elegir otro.' });
      return res.render('adminViews/adminUsers', {
        user: req.session.user,
        message: '❌ El email o el nombre de usuario ya existe, debe elegir otro. ❌',
      });
    }

    // Si no existe, proceder a actualizar el usuario
    const updateUserQuery = `UPDATE usuarios SET 
                              name = ?, 
                              username = ?, 
                              email = ?, 
                              num_colegiado = ?, 
                              is_admin = ? 
                              WHERE id = ?`;

    db.run(updateUserQuery, [name, username, email, num_colegiado, is_admin, userId], function(err) {
      if (err) {
        console.error('Error al actualizar el usuario:', err.message);
        return res.redirect(`/adminContact/editUser`);
      }
      console.log(`Usuario con ID ${userId} actualizado con éxito.`);
      req.session.user = {
        id: userId,
        name,
        username,
        email,
        num_colegiado,
        is_admin
      };
      res.json({ success: '✅ Usuario actualizado con éxito. ✅'});
    });
  });
});



module.exports = router;
