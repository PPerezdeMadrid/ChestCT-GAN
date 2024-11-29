var express = require('express');
var router = express.Router();
const bcrypt = require('bcrypt'); // Para comparar contraseñas
const sqlite3 = require('sqlite3').verbose();



/* GET users listing. */
router.get('/', function (req, res, next) {
  res.render('login', { user: req.session.user });
});

// Conectar a la base de datos
const db = new sqlite3.Database('./database.db', (err) => {
  if (err) {
    console.error('Error al conectar con la base de datos:', err.message);
  } else {
    console.log('Conexión exitosa a la base de datos.');
  }
});


router.post('/loginCliente', (req, res) => {
  const { username, passwd } = req.body;

  db.get('SELECT * FROM users WHERE username = ?', [username], (err, user) => {
    if (err) {
      console.error('Error al consultar la base de datos:', err.message);
      return res.status(500).send('Error interno del servidor');
    }

    if (!user) {
      return res.status(401).send('Usuario o contraseña incorrectos');
    }

    bcrypt.compare(passwd, user.password, (err, result) => {
      if (err) {
        console.error('Error al verificar la contraseña:', err.message);
        return res.status(500).send('Error interno del servidor');
      }

      if (result) {
        // Guardar los datos del usuario en la sesión.
        req.session.user = {
          name: user.name,
          username: user.username,
          email: user.email,
          isAdmin: user.is_admin
        };

        // Redirigir o renderizar la vista de perfil con los datos de sesión.
        res.render('profile', { user: req.session.user });
      } else {
        return res.render('login', { message: 'Usuario o contraseña incorrecto.' });
      }
    });
  });
});

router.post('/registerClient', (req, res) => {
  const { name, username, email, password, numColegiado, is_admin } = req.body;

  // Validar los datos (frontend ya lo hace)
  if (!name || !username || !email || !password || !numColegiado) {
    return res.render('login', { message: '✨ ¡Ups! Todos los campos son requeridos. Por favor, completa el formulario. ✨' });
  }

  // Verificar si el nombre de usuario ya está en uso
  const checkUserQuery = `SELECT * FROM users WHERE username = ?`;
  db.get(checkUserQuery, [username], (err, row) => {
    if (err) {
      console.error('Error al verificar el nombre de usuario:', err.message);
      return res.render('login', { message: '🌟 ¡Oh no! Algo salió mal. Por favor, intenta nuevamente más tarde. 🌟' });
    }

    if (row) {
      return res.render('login', { message: `¡Hola! Parece que el nombre de usuario "${username}" ya está en uso. ¡Prueba con otro! 😊` });
    }

    // Hash de la contraseña antes de guardarla
    bcrypt.hash(password, 10, (err, hashedPassword) => {
      if (err) {
        console.error('Error al hashear la contraseña:', err.message);
        return res.render('login', { message: '¡Oops! No pudimos procesar tu solicitud. Por favor, intenta más tarde.' });
      }

      // Insertar el nuevo usuario en la base de datos
      const insertUserQuery = `INSERT INTO users (username, name, email, password, num_colegiado, is_admin) VALUES (?, ?, ?, ?, ?, ?)`;
      db.run(insertUserQuery, [username, name, email, hashedPassword, numColegiado, is_admin ? 1 : 0], function (err) {
        if (err) {
          console.error('Error al insertar el usuario:', err.message);
          return res.render('login', { message: 'No pudimos registrar tu usuario. Intenta de nuevo.' });
        }

        // Crear sesión de usuario y redirigir al perfil
        req.session.user = {
          name: name,
          username: username,
          email: email,
          isAdmin: is_admin,
        };

        console.log(`Usuario ${username} registrado con éxito.`);
        res.render('profile', { user: req.session.user });
      });
    });
  });
});





module.exports = router;