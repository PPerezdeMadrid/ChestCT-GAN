var createError = require('http-errors');
var express = require('express');
var session = require('express-session');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var createError = require('http-errors');
require('dotenv').config();
const sqlite3 = require('sqlite3').verbose();

var indexRouter = require('./routes/index');
var loginRouter = require('./routes/login');
var contactRouter = require('./routes/contact');
var proyectoRouter = require('./routes/proyecto');
var profileRouter = require('./routes/profile');
var lungCTRouter = require('./routes/lungCT')
var adminContactRouter = require('./routes/adminContact')
var adminBlogRouter = require('./routes/adminBlog')
var adminStatisticsRouter = require('./routes/adminStatistic')
var adminUsersRouter = require('./routes/adminUsers')
var blogRouter = require('./routes/blog')
var entradaRouter = require('./routes/entrada')

var app = express();

// Configurar las sesiones
/** 
app.use(session({
  secret: process.env.SESSION_SECRET,
  resave: false,
  saveUninitialized: true
}));
*/


const SQLiteStore = require('connect-sqlite3')(session);
app.use(session({
  store: new SQLiteStore({ db: 'sessions.db', dir: path.join(__dirname, 'modules') }),
  secret: process.env.SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: {
    maxAge: 1000 * 60 * 60 * 2 // 2 horas
  }
}));


const createDatabase = require('./modules/database');
createDatabase();

// view engine setup
app.set('view engine', 'ejs');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));


app.use('/', indexRouter);
app.use('/login', loginRouter);
app.use('/contact', contactRouter);
app.use('/proyecto', proyectoRouter);
app.use('/lungCT', checkAuthenticated,lungCTRouter);
app.use('/profile', checkAuthenticated, profileRouter);
app.use('/adminContact', checkAdmin, adminContactRouter);
app.use('/adminBlog', checkAdmin, adminBlogRouter);
app.use('/adminStatistics', checkAdmin, adminStatisticsRouter);
app.use('/adminUsers', checkAdmin, adminUsersRouter);
app.use('/blog', blogRouter);
app.use('/entrada', entradaRouter);

// Middleware para verificar si el usuario está logueado
function checkAuthenticated(req, res, next) {
  console.log(req.session); 
  if (!req.session.user) {  
    return res.redirect('/login'); 
  }
  next();  
}

// Middleware para verificar si el usuario es admin
function checkAdmin(req, res, next) {
  console.log(req.session); 
  if (!req.session.user && req.session.user.isAdmin) {  
    return res.redirect('/login'); 
  }
  next();
}

app.get('/logout', (req, res) => {
  req.session.destroy((err) => {
    if (err) {
      console.log("Ha habido un error: ", err) 
    }
    console.log("El usuario ha cerrado la sesión")
    res.redirect('/'); 
  });
});

app.post("/notify", (req, res) => {
  const db_2 = new sqlite3.Database('./database.db', (err) => {
    if (err) {
      console.error('Error al conectar con la base de datos:', err.message);
    } else {
      console.log('Conexión exitosa a la base de datos.');
    }
  });

  const { mensaje } = req.body;

  if (!mensaje) {
    return res.status(400).json({ error: "El mensaje es obligatorio" });
  }

  // Guardar en la base de datos
  const query = `INSERT INTO notificaciones_mlops (mensaje) VALUES (?)`;
  db_2.run(query, [mensaje], function (err) {
    if (err) {
      console.error("Error al insertar la notificación:", err.message);
      return res.status(500).json({ error: "Error al guardar en la base de datos" });
    }

    console.log(`✅ Notificación guardada con ID ${this.lastID}`);

    // ✅ Responder al cliente correctamente
    res.status(200).json({
      status: "ok",
      message: "Notificación guardada correctamente",
      id: this.lastID
    });
  });
});




app.use(function(req, res, next) {
  next(createError(404));
});

// Middleware de manejo de errores
app.use(function(err, req, res, next) {
  // Establecer el estado HTTP del error
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // Renderizar una página de error
  res.status(err.status || 500);
  res.render('error');  // Asumiendo que tienes una vista llamada "error.ejs"
});

// Listening on port 8080
const PORT = 8080;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

module.exports = checkAuthenticated;
module.exports = app;