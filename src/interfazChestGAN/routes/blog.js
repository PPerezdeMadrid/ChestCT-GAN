var express = require('express');
var router = express.Router();
const sqlite3 = require('sqlite3').verbose();

const db = new sqlite3.Database('./database.db');


/* GET home page. */
router.get('/', function(req, res, next) {
    db.all('SELECT id, titulo, resumen, fecha_publicacion, autor FROM entradas_blog ORDER BY fecha_publicacion DESC', function(err, rows) {
        if (err) {
            console.error('Error al obtener las entradas del blog:', err);
            return res.render('login', {message:'Error al cargar el blog.'});
        }
        
        res.render('blog', {
            user: req.session.user, 
            blogEntries: rows
        });
    });
});





module.exports = router;
