var createError = require('http-errors');
var express = require('express');
var session = require('express-session');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var createError = require('http-errors');


var indexRouter = require('./routes/index');
var loginRouter = require('./routes/login');
var contactRouter = require('./routes/contact');
var proyectoRouter = require('./routes/proyecto');
var profileRouter = require('./routes/profile');
var lungCTRouter = require('./routes/lungCT')
var adminContactRouter = require('./routes/adminContact')
var adminBlogRouter = require('./routes/adminBlog')
var adminUsersRouter = require('./routes/adminUsers')
var blogRouter = require('./routes/blog')
var entradaRouter = require('./routes/entrada')


var app = express();

// Configurar las sesiones
app.use(session({
  secret: 'CLAVE',  // Usa una clave secreta para cifrar la sesión
  resave: false,
  saveUninitialized: true
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

// Listening on port 80
const PORT = 80;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

module.exports = checkAuthenticated;
module.exports = app;