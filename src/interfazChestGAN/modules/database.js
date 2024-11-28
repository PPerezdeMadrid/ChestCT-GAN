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
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
      )`, (err) => {
        if (err) {
          console.error('Error al crear la tabla users:', err.message);
        } else {
          console.log('Tabla users creada o ya existente.');
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
