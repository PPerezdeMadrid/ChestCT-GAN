var express = require('express');
var router = express.Router();
const sqlite3 = require('sqlite3').verbose();

// Conectar a la base de datos
const db = new sqlite3.Database('./database.db');

// Ruta para mostrar una entrada específica
router.get('/:id', (req, res) => {
    const entryId = req.params.id;
    db.get('SELECT * FROM entradas_blog WHERE id = ?', [entryId], (err, entry) => {
        if (err) {
            return res.render('entrada', {
                user: req.session.user,
                message: 'Error al recuperar la entrada.'
              });
        }
        if (!entry) {
            return res.render('entrada', {
                user: req.session.user,
                message: 'Entrada no encontrada.'
              });
        }
        // Renderiza la vista de la entrada y pasa la entrada como variable
        res.render('entrada', { entry });
    });
});

module.exports = router;
