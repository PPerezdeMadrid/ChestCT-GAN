const sqlite3 = require('sqlite3').verbose();

function createDatabase() {
  const db = new sqlite3.Database('./database.db', (err) => {
    if (err) {
      console.error('Error al abrir la base de datos:', err.message);
    } else {
      console.log('Conexión a SQLite exitosa.');

      // Crear la tabla users si no existe
      db.run(`CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        num_colegiado TEXT NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
      )`, (err) => {
        if (err) {
          console.error('Error al crear la tabla users:', err.message);
        } else {
          console.log('Tabla users creada o ya existente.');
        }
      });

      db.run(`
        CREATE TABLE IF NOT EXISTS mensajes_contacto (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          nombre TEXT NOT NULL,
          email TEXT NOT NULL,
          mensaje TEXT NOT NULL,
          fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
      `, (err) => {
        if (err) {
          console.error('Error al crear la tabla users:', err.message);
        } else {
          console.log('Tabla mensajes_contacto creada o ya existente.');
        }
      });

      db.run(`
        CREATE TABLE IF NOT EXISTS entradas_blog (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          titulo TEXT NOT NULL,
          resumen TEXT NOT NULL,
          contenido TEXT NOT NULL,
          fecha_publicacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          autor TEXT NOT NULL
        )
      `, (err) => {
        if (err) {
          console.error('Error al crear la tabla users:', err.message);
        } else {
          console.log('Tabla entradas_blog creada o ya existente.');
        }
      });

      // Cerrar la conexión
      db.close((err) => {
        if (err) {
          console.error('Error al cerrar la conexión con la base de datos:', err.message);
        } else {
          console.log('Conexión a SQLite cerrada.');
        }
      });
    }
  });
}

module.exports = createDatabase;
