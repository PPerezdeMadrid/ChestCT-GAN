var express = require('express');
var router = express.Router();
const bcrypt = require('bcrypt');
const sqlite3 = require('sqlite3').verbose();


// Conectar a la base de datos
const db = new sqlite3.Database('./database.db', (err) => {
  if (err) {
    console.error('Error al conectar con la base de datos:', err.message);
  } else {
    console.log('Conexión exitosa a la base de datos.');
  }
});


/* GET home page. */
router.get('/', function(req, res, next) {
  if (!req.session.user) {
    return res.status(401).send('No has iniciado sesión');
  }

  const { email } = req.session.user;

  db.get('SELECT username, name, email, num_colegiado, is_admin FROM usuarios WHERE email = ?', 
  [email], (err, user) => {
    if (err) {
      console.error('Error al obtener los datos del usuario:', err.message);
      return res.status(500).send('Error interno del servidor');
    }

    if (!user) {
      return res.status(404).send('Usuario no encontrado');
    }
    
    // Guardar info en la sesión
    req.session.user = {
      ...req.session.user,
      num_colegiado: user.num_colegiado,
      isAdmin: user.is_admin
    };

    // Renderizar la vista de perfil con los datos del usuario en sesión
    res.render('profile', { 
      user: req.session.user,
    });
  });
});


/* Actualizar nombre de usuario */
router.post('/updateUsername', function(req, res) {
  const { newUsername } = req.body;
  const { email } = req.session.user;

  if (!newUsername) {
    return res.status(400).send('Nuevo nombre de usuario es obligatorio');
  }

  // Comprobar si el nuevo nombre de usuario ya existe en la base de datos
  const checkUsernameQuery = `SELECT * FROM usuarios WHERE username = ?`;

  db.get(checkUsernameQuery, [newUsername], function(err, row) {
    if (err) {
      console.error('Error al verificar el nombre de usuario:', err.message);
      return res.status(500).send({ error: 'Error al verificar el nombre de usuario' });
    }

    if (row) {
      console.log("Nombre de usuario ya en uso")
      return res.status(400).send({ error: 'El nombre de usuario ya está en uso. Por favor, elige otro.' });
    }

    // Si no existe, proceder con la actualización
    db.run('UPDATE usuarios SET username = ? WHERE email = ?', [newUsername, email], function(err) {
      if (err) {
        console.error('Error al actualizar el nombre de usuario:', err.message);
        return res.status(500).send({ error: 'Error al actualizar nombre de usuario' });
      }

      req.session.user.username = newUsername;

      res.send({ message: 'Nombre de usuario actualizado con éxito', newUsername });
    });
  });
});


/* GET img avatar */
router.get('/getAvatar', function (req, res) {
  const { email } = req.session.user;

  if (!email) {
      return res.status(401).send({ error: 'No se encontró el usuario en la sesión.' });
  }

  db.get('SELECT avatar FROM usuarios WHERE email = ?', [email], function (err, row) {
      if (err) {
          console.error('Error al obtener el avatar:', err.message);
          return res.status(500).send({ error: 'Error interno del servidor.' });
      }

      if (!row || !row.avatar) {
          return res.status(404).send({ error: 'No se encontró un avatar para este usuario.' });
      }

      res.status(200).send({ avatar: row.avatar});
  });
});


router.post('/saveAvatar', (req, res) => {
  const { userId, photoPath } = req.body;

  if (!userId || !photoPath) {
      return res.status(400).send({ error: 'Faltan datos requeridos: userId o photoPath.' });
  }

  // Actualizar la ruta del avatar en la base de datos
  const query = 'UPDATE usuarios SET avatar = ? WHERE id = ?';
  db.run(query, [photoPath, userId], function (err) {
      if (err) {
          console.error('Error al actualizar el avatar:', err.message);
          return res.status(500).send({ error: 'Error interno del servidor.' });
      }

      if (this.changes === 0) {
          return res.status(404).send({ error: 'No se encontró el usuario para actualizar el avatar.' });
      }

      res.status(200).send({ message: 'Avatar actualizado con éxito.' });
  });
});


/* Actualizar Nombre */
router.post('/updateName', function(req, res) {
  const { newName } = req.body;
  const { email } = req.session.user;

  if (!newName) {
    return res.status(400).send('Nuevo nombre es obligatorio');
  }

  db.run('UPDATE usuarios SET name = ? WHERE email = ?', [newName, email], function(err) {
    if (err) {
      console.error('Error al actualizar el nombre:', err.message);
      return res.status(500).send('Error al actualizar nombre');
    }

    req.session.user.name = newName;

    res.send({ message: 'Nombre actualizado con éxito', newName });
  });
});




module.exports = router;
