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

  db.get('SELECT * FROM usuarios WHERE username = ?', [username], (err, user) => {
    if (err) {
      console.error('Error al consultar la base de datos:', err.message);
      return res.status(500).send('Error interno del servidor');
    }

    if (!user) {
       return res.render('login', {message:'Usuario o contraseña incorrectos'});
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
        // res.render('profile', { user: req.session.user });
        res.redirect('/profile');

      } else {
        return res.render('login', { message: 'Usuario o contraseña incorrecto.' });
      }
    });
  });
});


router.post('/registerClient', (req, res) => {
  const { name, username, email, password, numColegiado, is_admin } = req.body;

  // Validar datos del formulario
  if (!name || !username || !email || !password || (!is_admin && !numColegiado)) {
    return res.render('login', { 
      message: '✨ ¡Ups! Todos los campos requeridos deben ser completados. Por favor, revisa el formulario. ✨' 
    });
  }

  // Verificar si el nombre de usuario ya está en uso
  const checkUserQuery = `SELECT * FROM usuarios WHERE username = ?`;
  db.get(checkUserQuery, [username], (err, row) => {
    if (err) {
      console.error('Error al verificar el nombre de usuario:', err.message);
      return res.render('login', { 
        message: '🌟 ¡Oh no! Algo salió mal. Por favor, intenta nuevamente más tarde. 🌟' 
      });
      
    }

    if (row) {
      return res.render('login', { 
        message: `¡Hola! Parece que el nombre de usuario "${username}" ya está en uso. ¡Prueba con otro! 😊` 
      });
    }

    // Hash de la contraseña
    bcrypt.hash(password, 10, (err, hashedPassword) => {
      if (err) {
        console.error('Error al hashear la contraseña:', err.message);
        return res.render('login', { 
          message: '¡Oops! No pudimos procesar tu solicitud. Por favor, intenta más tarde.' 
        });
      }

      // Asignar valor predeterminado para `num_colegiado` si es administrador
      const numColegiadoValue = is_admin ? 'No tiene' : numColegiado;

      // Consulta de inserción
      const insertUserQuery = `
        INSERT INTO usuarios (username, name, email, password, num_colegiado, is_admin) 
        VALUES (?, ?, ?, ?, ?, ?)
      `;

      // Parámetros de la consulta
      const queryParams = [
        username,
        name,
        email,
        hashedPassword,
        numColegiadoValue,
        is_admin ? 1 : 0
      ];

      db.run(insertUserQuery, queryParams, function (err) {
        if (err) {
          console.error('Error al insertar el usuario:', err.message);
          if (err.message.includes('UNIQUE constraint failed')) {
            if(err.message.includes('usuarios.email')){
              return res.render('login', { 
                message: '⚠️ El correo electrónico ya está registrado. Intenta con otro correo. ⚠️' ,
                user: req.session.user
              });
            }else{
              return res.render('login', { 
                message: '⚠️ El usuario ya está registrado. Intenta con nombre de usuario ⚠️' ,
                user: req.session.user
              });
            }
          } else {
            return res.render('login', { 
              message: 'No hemos podido registrar tu usuario. Inténtalo de nuevo.' 
            });
          }
          
        }

        req.session.user = {
          id: this.lastID,
          name,
          username,
          email,
          isAdmin: !!is_admin,
        };

        console.log(`Usuario ${username} registrado con éxito.`);
        res.redirect('/profile');
      });
    });
  });
});







module.exports = router;