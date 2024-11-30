const express = require('express');
const router = express.Router();
const sqlite3 = require('sqlite3').verbose();

const db = new sqlite3.Database('./database.db');


router.get('/', function(req, res, next) {
    res.render('adminViews/adminBlog', { user: req.session.user});
  });


router.post('/add-entry', (req, res) => {
    const { titulo, resumen, contenido, fecha_publicacion } = req.body;

    // Verificar que todos los campos estén completos
    if (!titulo || !resumen || !contenido || !fecha_publicacion) {
        return res.render('adminViews/adminBlog', {
            user: req.session.user,
            message: 'Todos los campos deben ser rellenados'
        });
    }

    // Obtener el nombre de usuario del autor de la sesión
    const autor = req.session.user.username;

    // Consulta para insertar la entrada en la base de datos
    const query = `
        INSERT INTO entradas_blog (titulo, resumen, contenido, fecha_publicacion, autor)
        VALUES (?, ?, ?, ?, ?);
    `;

    db.run(query, [titulo, resumen, contenido, fecha_publicacion, autor], function (err) {
        if (err) {
            console.error('Error adding entry:', err);
            return res.render('adminViews/adminBlog', {
                user: req.session.user,
                message: '¡Oh no! Algo salió mal al añadir la entrada al blog. Inténtalo más tarde.'
            });
        }

        // Mensaje de éxito
        res.render('adminViews/adminBlog', {
            user: req.session.user,
            message: '🌟 Entrada añadida 🌟'
        });
    });
});



module.exports = router;
